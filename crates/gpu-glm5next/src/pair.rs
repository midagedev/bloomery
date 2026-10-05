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
//! A pass of slots ([`SlotRows`], `GpuModel::step_slots`,
//! `GpuModel::verify_slots`) walks its rows one layer apart the same way,
//! each row a row of its own slot's sequence: its embedding at its row's
//! position, its KDA launches on its slot's lanes (row base 0 in place on
//! its committed lane, a verify's row 1 from the lane its row 0 wrote into
//! the next), its latent rows and conv ring in its slot's stores, and its
//! final streams in its slot's own row ([`binding`]), so each row is bit for
//! bit its slot's row alone. A plain load runs a row a slot, a plain step; a
//! NextN load each slot's verify's two — a pass of two slots four rows in
//! flight — and each slot then waits for its commit ([`Lanes`]) as its own
//! verify does, `GpuModel::commit_slots` keeping its accepted rows slot by
//! slot (`SlotRows::keep_slot`). The rows in flight are the load's, so a
//! load of one lane runs a plain pass as a load of two does. The host half
//! plans each slot's rows from its own sequence ([`Body::plan_slot_rows`]):
//! the stores standing at its first position, its waiting cut carried out
//! once from its own checkpoints before its row 0, its held position and
//! rows' records moved.

use bloomery_gpu::checkpoint::Checkpoints;
use bloomery_gpu::head::Head;
use bloomery_gpu::hybrid::Chain;
use bloomery_gpu::linear::delta::row_lane;
use bloomery_gpu::model::{ChainBody, Rows, SlotRange, SlotRows};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError, capturing};
use cuda_core::{CudaStream, DeviceBuffer};
use runtime::seqstate::{Kept, Why};

use model::arch::glm5next::place::KdaLanes;

use super::nextn::{Held, Wrote};
use super::{
    Body, GlmSlot, LANES, Parts, Plant, RowBind, RowScratch, Scratch, SeqParts, StepInput, Store,
    cut_into, held_at, shape,
};
use crate::program;

/// The rows one load holds in flight, whatever its KDA lanes: every
/// load-sized term counts them — the pass's row buffers ([`Scratch`]), the
/// card experts' rows, the host boundary's, the expert tier's. A verify's
/// two (a row a lane of the one sequence), a plain pass's two (a row a
/// slot) and a NextN load's drafted pass's four (each slot's verify's
/// rows).
pub const LOAD_ROWS: usize = 4;

// The load's rows run the verifies of the most sequences a walk binds at
// once, a slot's rows a verify's, a row a lane.
const _: () =
    assert!(LOAD_ROWS == RowBind::SEQS * LANES && LANES >= 1 && LANES <= u32::MAX as usize);

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

    /// Whether the verify waiting for its commit is one of `rows` rows from
    /// `pos0`.
    fn waits_for(&self, pos0: u32, rows: usize) -> bool {
        self.waiting
            .is_some_and(|w| w.pos0 == pos0 && w.rows as usize == rows)
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
            .ok_or_else(|| shape(format!("row {row} of a pass on a load of {LOAD_ROWS} rows")))?;
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
        if tokens.len() != LANES {
            return Err(shape(format!(
                "a verify of {} rows; the lanes hold {LANES}",
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
        if carried < LANES {
            return Err(shape(format!(
                "a verify of {LANES} rows on a host boundary of {carried}: the load makes the \
                 boundary with a row for each of the verify's rows"
            )));
        }
        if pos as usize + LANES > self.ctx {
            return Err(shape(format!(
                "a verify of {LANES} rows at position {pos} in stores of {}",
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
            rows: LANES as u32,
        });
        self.wrote.pair = super::nextn::Held::at(pos, LANES as u32);
        Ok(())
    }

    /// Refused by name on a load of one KDA lane: a verify's row 1 writes a
    /// lane the load does not hold.
    fn refuse_one_lane(&self) -> Result<(), GpuError> {
        match self.lanes {
            KdaLanes::Two => Ok(()),
            KdaLanes::One => Err(shape(format!(
                "a verify of {LANES} rows on a load of one KDA lane: only a load that \
                 verifies holds the second (Body::open_placed_lanes at KdaLanes::Two, or the \
                 NextN load)"
            ))),
        }
    }

    /// Keep the first `pos − pos0` rows of the verify waiting for its commit
    /// ([`Body::commit_live`]). With no verify waiting, or a `pos` outside
    /// its rows, the cut [`Body::cut`] takes.
    pub(super) fn commit(&mut self, gpu: &Gpu, pos: u32) -> Result<(), GpuError> {
        if self.commit_live(gpu, pos)? {
            return Ok(());
        }
        self.cut(pos)
    }

    /// The live sequence's verify waiting for its commit kept up to `pos`
    /// ([`commit_lanes`]: the lane word to the lane the last kept row
    /// wrote, or where it stands when that is row 0's), the draft's records
    /// cut with it. Whether a verify waited with `pos` inside its rows;
    /// nothing moves otherwise.
    fn commit_live(&mut self, gpu: &Gpu, pos: u32) -> Result<bool, GpuError> {
        let count = self.lanes.count() as u32;
        let Body {
            s,
            held,
            wrote,
            nextn,
            ..
        } = self;
        let Scratch { lane, lanes, .. } = s;
        let done = commit_lanes(gpu.stream(), lane, lanes, held, wrote, pos, count)?;
        if let Some(n) = nextn.as_deref_mut().filter(|_| done) {
            n.cut(pos);
        }
        Ok(done)
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
    /// A verify of one sequence's positions: a row a lane ([`LANES`]), the
    /// drafted token's and the step's.
    const MAX_ROWS: usize = LANES;
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
                "{} heads; the verify runs {LANES}",
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
                .copy_from_device_async(&self.s.rows[0].streams[fin], gpu.stream())?;
        }
        self.planted(Plant::AfterLaunch)
    }
}

// ------------------------------------------------- a pass of several slots

/// The rows the last plan of a pass of several slots wrote, each row's slot
/// and position: what an enqueue outside a capture runs on, and only on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SlotsPlanned {
    at: [(usize, u32); LOAD_ROWS],
    n: usize,
}

impl SlotsPlanned {
    /// Each row's slot and position, the first [`LOAD_ROWS`] of them, and
    /// the pass's rows.
    fn of(rows: &[SlotRange]) -> SlotsPlanned {
        let mut at = [(0, 0); LOAD_ROWS];
        let each = rows
            .iter()
            .flat_map(|r| (r.pos0..).take(r.rows.len()).map(|p| (r.slot, p)));
        for (a, row) in at.iter_mut().zip(each) {
            *a = row;
        }
        SlotsPlanned {
            at,
            n: pass_rows(rows),
        }
    }
}

/// The rows of a pass of `rows`: where its last slot's range ends.
fn pass_rows(rows: &[SlotRange]) -> usize {
    rows.last().map_or(0, |r| r.rows.end)
}

/// The chain a pass of `rows` runs: [`program::slots_point`]'s point on a
/// page of one column a row — the step for one row, the pair for two, the
/// quad for four; `None` for a point the walk does not lay.
fn slots_chain(rows: &[SlotRange]) -> Option<Chain> {
    let o = program::slots_point(rows.len(), pass_rows(rows)).ok()?;
    Chain::of(o.units, o.cols, 1)
}

/// One exchange of handles of a pass of several slots' final-stream binding
/// ([`binding`]).
#[derive(Clone, Copy, Debug)]
enum Swap {
    /// Pass row `row`'s final streams with the live sequence's own row
    /// `own`'s — the load's row `own`'s, wherever the pass lays the live
    /// slot.
    Live { row: usize, own: usize },
    /// Pass row `row`'s final streams with the parked sequence `seq`'s own
    /// row `own`'s.
    Parked { row: usize, seq: usize, own: usize },
}

/// The exchanges that put each pass row's slot's own row of final streams
/// (`r − range.start`, the buffer its draft's walks and its sequence's state
/// read) in that row's buffer for a pass of `rows`, in order: the live
/// slot's own rows first when the pass does not lay it at row 0 — its rows
/// are the load's first, so they trade places with the rows laid there —
/// then each parked slot's rows into theirs. Pointer moves only;
/// [`exchange`] undoes them in reverse order. An exchange a row at most:
/// a pass past the load's [`LOAD_ROWS`] rows is refused by name.
fn binding(rows: &[SlotRange]) -> Result<[Option<Swap>; LOAD_ROWS], GpuError> {
    let live = rows
        .iter()
        .filter(|r| r.slot == 0 && r.rows.start > 0)
        .flat_map(|r| {
            (0..r.rows.len()).map(|own| Swap::Live {
                row: r.rows.start + own,
                own,
            })
        });
    let parked = rows
        .iter()
        .filter(|r| r.slot != 0)
        .enumerate()
        .flat_map(|(seq, r)| {
            (0..r.rows.len()).map(move |own| Swap::Parked {
                row: r.rows.start + own,
                seq,
                own,
            })
        });
    let mut out = [None; LOAD_ROWS];
    for (i, sw) in live.chain(parked).enumerate() {
        *out.get_mut(i).ok_or_else(|| {
            shape(format!(
                "a pass of {} rows on a load of {LOAD_ROWS}",
                pass_rows(rows)
            ))
        })? = Some(sw);
    }
    Ok(out)
}

/// Carry out the exchanges `swaps` ([`binding`]) between the rows' buffers
/// `s` and the pass's `parked` sequences on stream buffer `fin`, in order,
/// or undo them, in reverse when `undo`.
fn exchange(
    s: &mut Scratch,
    parked: &mut [&mut GlmSlot],
    fin: usize,
    swaps: &[Option<Swap>; LOAD_ROWS],
    undo: bool,
) -> Result<(), GpuError> {
    let mut one = |sw: Swap| -> Result<(), GpuError> {
        match sw {
            Swap::Live { row, own } => s.swap_streams(row, own, fin),
            Swap::Parked { row, seq, own } => {
                let r = s
                    .row_mut(row)
                    .ok_or_else(|| shape(format!("row {row} of a pass of several slots")))?;
                let buf = parked
                    .get_mut(seq)
                    .and_then(|p| p.rows.get_mut(own))
                    .ok_or_else(|| shape(format!("parked sequence {seq}'s own row {own}")))?;
                std::mem::swap(&mut r.streams[fin], buf);
                Ok(())
            }
        }
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

/// The commit at `pos` of the verify waiting on one sequence's `lanes`, its
/// KDA layers holding `count` lanes ([`Body::commit`]'s rule, its one
/// owner): the lane word `lane` to the lane the last kept row wrote
/// ([`row_lane`] of the kept rows, uploaded when it moves), the waiting
/// dropped, `held` to `pos` and every arena's record cut. Whether a verify
/// waited, ran whole and holds `pos` inside its rows; nothing moves
/// otherwise, and what that means is the caller's — the cut
/// [`Body::commit`] takes, the refusal `SlotRows::keep_slot` names.
fn commit_lanes(
    stream: &CudaStream,
    lane: &mut DeviceBuffer<u32>,
    lanes: &mut Lanes,
    held: &mut u32,
    wrote: &mut Wrote,
    pos: u32,
    count: u32,
) -> Result<bool, GpuError> {
    let Some(w) = lanes.waiting else {
        return Ok(false);
    };
    let end = w.pos0 + w.rows;
    if pos <= w.pos0 || pos > end || *held != end {
        return Ok(false);
    }
    let c = lanes.committed;
    let at = row_lane(c, pos - w.pos0 - 1, count);
    if at != c {
        lane.copy_from_host(stream, &[at])?;
    }
    *lanes = Lanes {
        committed: at,
        waiting: None,
    };
    *held = pos;
    wrote.cut(pos);
    Ok(true)
}

/// One sequence's host side a pass of several slots plans its rows of, lent
/// apart: the live sequence's fields or a parked [`GlmSlot`]'s.
struct SeqHost<'a> {
    stores: &'a mut [Store],
    lanes: &'a mut Lanes,
    held: &'a mut u32,
    ckpt: &'a mut Checkpoints,
    wrote: &'a mut Wrote,
}

impl<'a> SeqHost<'a> {
    fn parked(p: &'a mut GlmSlot) -> SeqHost<'a> {
        SeqHost {
            stores: &mut p.stores,
            lanes: &mut p.lanes,
            held: &mut p.held,
            ckpt: &mut p.ckpt,
            wrote: &mut p.wrote,
        }
    }
}

impl Body {
    /// Refused by name before anything moves, a pass of several slots of
    /// `rows` whose busy slots other than slot 0 are `parked`: with the taps
    /// armed or a route trace attached; a slot of other rows than the load
    /// runs a slot — one, its plain step, or on a NextN load two, its
    /// verify's; at a point the walk does not lay ([`slots_chain`]); more
    /// slots than the walk binds sequences ([`RowBind`]); and `parked` not
    /// the pass's other sequences. A verify waiting on the live sequence is
    /// refused by the plan, and by the enqueue unless it is the pass's own
    /// ([`SlotRows::enqueue_slots`]).
    fn refuse_slots(&self, parked: &[&mut GlmSlot], rows: &[SlotRange]) -> Result<(), GpuError> {
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
        let (each, load) = if self.nextn.is_some() {
            (
                LANES,
                format!(
                    " on a NextN load: each slot's rows are its verify's {LANES}, its step's and \
                     its drafted token's"
                ),
            )
        } else {
            (
                1,
                ": each slot's row is one plain step; a verify of one slot's rows is \
                 Rows::step_rows"
                    .to_string(),
            )
        };
        if let Some(r) = rows.iter().find(|r| r.rows.len() != each) {
            return Err(shape(format!(
                "slot {}'s {} rows in a pass of several slots{load}",
                r.slot,
                r.rows.len()
            )));
        }
        if slots_chain(rows).is_none() {
            return Err(shape(format!(
                "a pass of {} slots' {} rows: the walk lays a plain pass's one or two slots, a \
                 row a slot, or a NextN load's one or two slots' verify rows, {LANES} a slot, one \
                 column a row",
                rows.len(),
                pass_rows(rows)
            )));
        }
        if rows.len() > RowBind::SEQS {
            return Err(shape(format!(
                "a pass of {} slots: the walk binds {} sequences, the live one's and one parked \
                 or two parked",
                rows.len(),
                RowBind::SEQS
            )));
        }
        let want = rows.iter().filter(|r| r.slot != 0).count();
        if parked.len() != want || parked.iter().any(|p| p.rows.is_empty()) {
            return Err(shape(format!(
                "{} parked sequences for a pass whose other busy slots are {want}",
                parked.len()
            )));
        }
        Ok(())
    }

    /// Refused by name while a verify waits on the live sequence: a pass
    /// would run a slot on lanes the commit has not named.
    fn refuse_live_waiting(&self) -> Result<(), GpuError> {
        self.s
            .lanes
            .refuse_if_waiting(self.held)
            .map_err(|e| prefixed(e, "a pass of several slots"))
    }

    /// Each slot's ids in the pass's `ids`, every id in the vocabulary,
    /// every row's position in the stores and its sequence standing at its
    /// first position with no verify waiting ([`held_at`]), refused by name
    /// naming the slot.
    fn refuse_slot_rows(
        &self,
        parked: &[&mut GlmSlot],
        rows: &[SlotRange],
        ids: &[u32],
    ) -> Result<(), GpuError> {
        let mut parked = parked.iter();
        for r in rows {
            let named = |e| prefixed(e, &format!("slot {}", r.slot));
            let own = ids.get(r.rows.clone()).ok_or_else(|| {
                named(shape(format!(
                    "rows {:?} past the pass's {} ids",
                    r.rows,
                    ids.len()
                )))
            })?;
            for (o, &t) in own.iter().enumerate() {
                if t as usize >= self.embd.n_vocab {
                    return Err(named(shape(format!(
                        "token {t} is past the {} embedding rows",
                        self.embd.n_vocab
                    ))));
                }
                if r.pos0 as usize + o >= self.ctx {
                    return Err(named(shape(format!(
                        "a step at position {} in stores of {}",
                        r.pos0 as usize + o,
                        self.ctx
                    ))));
                }
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

    /// One slot's rows of a pass planned from its own sequence (`None` the
    /// live one's), [`SlotRows::plan_slots`]'s per-slot entry. Refused by
    /// name before anything moves: other than one row or the verify's
    /// [`LANES`], other than an id a row, rows past the stores, the sequence
    /// not standing at `r.pos0` or a verify of it waiting ([`held_at`]).
    /// Then its waiting cut carried out once before its row 0 ([`cut_into`]
    /// on its sequence), each row's inputs written into its pass row's
    /// buffers (bound, [`binding`]) and its `held` moved past them, its step
    /// row's record moved to `r.pos0` — and for a verify's rows
    /// [`Body::plan_pair`]'s rule on its own sequence: its lanes left
    /// waiting for the pass's commit and its verify rows' record named.
    fn plan_slot_rows(
        &mut self,
        stream: &CudaStream,
        seq: Option<&mut GlmSlot>,
        r: &SlotRange,
        ids: &[u32],
    ) -> Result<(), GpuError> {
        let n = r.rows.len();
        if n != 1 && n != LANES {
            return Err(shape(format!(
                "slot {}'s {n} rows in a pass: a slot's rows are its plain step's one or its \
                 verify's {LANES}",
                r.slot
            )));
        }
        if ids.len() != n {
            return Err(shape(format!(
                "slot {}'s {} ids for its {n} rows",
                r.slot,
                ids.len()
            )));
        }
        let pos0 = r.pos0;
        if pos0 as usize + n > self.ctx {
            return Err(shape(format!(
                "slot {}'s {n} rows from position {pos0} in stores of {}",
                r.slot, self.ctx
            )));
        }
        let Body {
            embd,
            s,
            stores,
            held,
            ckpt,
            wrote,
            ..
        } = self;
        let Scratch { rows, lanes, .. } = s;
        let seq = match seq {
            Some(p) => SeqHost::parked(p),
            None => SeqHost {
                stores,
                lanes,
                held,
                ckpt,
                wrote,
            },
        };
        held_at(seq.lanes, *seq.held, pos0)
            .map_err(|e| prefixed(e, &format!("slot {}", r.slot)))?;
        cut_into(
            stream,
            seq.ckpt,
            seq.stores,
            seq.lanes.committed(),
            *seq.held,
        )?;
        for (o, (&t, p)) in ids.iter().zip(pos0..).enumerate() {
            embd.fill(t)?;
            let row = r.rows.start + o;
            let buf = rows.get_mut(row).ok_or_else(|| {
                shape(format!("row {row} of a pass on a load of {LOAD_ROWS} rows"))
            })?;
            write_row(stream, buf, &embd.streams, p)?;
        }
        *seq.held = pos0 + n as u32;
        seq.wrote.step = Held::at(pos0, 1);
        if n == LANES {
            seq.lanes.waiting = Some(Waiting {
                pos0,
                rows: n as u32,
            });
            seq.wrote.pair = Held::at(pos0, n as u32);
        }
        Ok(())
    }

    /// Each slot's rows of `rows` planned in pass order
    /// ([`Body::plan_slot_rows`]), each on its own sequence — slot 0's the
    /// live one, every other the next of `parked` — the first refusal
    /// ending the plan.
    fn plan_each_slot(
        &mut self,
        stream: &CudaStream,
        parked: &mut [&mut GlmSlot],
        rows: &[SlotRange],
        ids: &[u32],
    ) -> Result<(), GpuError> {
        let mut parked = parked.iter_mut();
        for r in rows {
            let seq = match r.slot {
                0 => None,
                slot => Some(
                    parked
                        .next()
                        .map(|p| &mut **p)
                        .ok_or_else(|| shape(format!("slot {slot}'s parked sequence")))?,
                ),
            };
            let own = ids.get(r.rows.clone()).ok_or_else(|| {
                shape(format!(
                    "slot {}'s rows {:?} past the pass's {} ids",
                    r.slot,
                    r.rows,
                    ids.len()
                ))
            })?;
            self.plan_slot_rows(stream, seq, r, own)?;
        }
        Ok(())
    }

    /// Each slot's verify row 0's final streams into its own pair-0 copy at
    /// the pass's end, after the rows' buffers went home ([`exchange`]'s
    /// undo) — the per-slot form of the verify's copy ([`Rows::enqueue_rows`]):
    /// a NextN load keeps them past the step that writes the row's buffers
    /// next. Nothing on a plain load.
    fn copy_pair0(
        &mut self,
        gpu: &Gpu,
        parked: &mut [&mut GlmSlot],
        rows: &[SlotRange],
        fin: usize,
    ) -> Result<(), GpuError> {
        let Body { nextn, s, .. } = self;
        let Some(nx) = nextn.as_deref_mut() else {
            return Ok(());
        };
        let mut parked = parked.iter_mut();
        for r in rows {
            if r.slot == 0 {
                nx.pair0_mut()
                    .copy_from_device_async(&s.rows[0].streams[fin], gpu.stream())?;
                continue;
            }
            let GlmSlot {
                rows: own, draft, ..
            } = parked
                .next()
                .map(|p| &mut **p)
                .ok_or_else(|| shape(format!("slot {}'s sequence in the pass", r.slot)))?;
            let row0 = own
                .first()
                .ok_or_else(|| shape(format!("slot {}'s own row 0", r.slot)))?;
            let d = draft.as_mut().ok_or(GpuError::State {
                what: "glm5next Body::enqueue_slots",
                missing: "the slot's NextN side (a NextN load's sequences each carry one)",
            })?;
            d.pair0_mut().copy_from_device_async(row0, gpu.stream())?;
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
            rows: s_rows, lane, ..
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
        let [row0, a, b, c] = s_rows;
        let parts = Parts {
            k,
            d: dims,
            cfg,
            names,
            s: row0,
            idle: [a, b, c],
            idle_at: [1, 2, 3],
            row: 0,
            lane: first.lane,
            stores: first.stores,
            bind: RowBind::of_pass(rows, other),
            slots,
            card,
            taps: None,
        };
        program::walk_slots(gpu, w, parts, hybrid, rows.len(), heads)
    }
}

impl SlotRows for Body {
    /// The load's rows in flight ([`LOAD_ROWS`]): the plain pass's two, a
    /// row a slot, a NextN load's drafted pass's four, each slot's verify's
    /// two.
    const MAX_ROWS: usize = LOAD_ROWS;
    /// The commit settles a partial keep ([`Body::keep_slot`]: each slot's
    /// lane word, held count and draft cut to its kept rows).
    const SETTLES_PARTIAL_KEEP: bool = true;

    /// Refused by name on a NextN load, a pass that keeps every row as it
    /// runs (`GpuModel::step_slots`) with a slot of several rows: a slot's
    /// rows there are its verify's, which wait for its commit, and a kept
    /// pass tells no slot's draft what it ran. A slot of one row is
    /// [`Body::refuse_slots`]' to refuse; nothing on a plain load.
    fn refuse_kept(&self, ranges: &[SlotRange]) -> Result<(), GpuError> {
        let Some(r) = ranges
            .iter()
            .find(|r| r.rows.len() > 1)
            .filter(|_| self.nextn.is_some())
        else {
            return Ok(());
        };
        Err(shape(format!(
            "a pass of several slots on a NextN load that keeps every row as it runs \
             (GpuModel::step_slots), slot {}'s {} rows among them: each slot's rows are its \
             verify's, and a kept pass tells no slot's draft what it ran; run them through \
             GpuModel::verify_slots and keep them at GpuModel::commit_slots",
            r.slot,
            r.rows.len()
        )))
    }

    /// The chain of the point the walk lays ([`slots_chain`]): the step for
    /// one row, the pair for two, the quad for four. A point the walk does
    /// not lay is refused by name before any capture or launch of it
    /// ([`Body::refuse_slots`]), so no point the walk does not lay reaches
    /// here; one that does is a named panic.
    fn chain_of(rows: &[SlotRange]) -> Chain {
        slots_chain(rows).unwrap_or_else(|| {
            panic!(
                "glm5next SlotRows::chain_of: a pass of {} slots and {} rows, a point the walk \
                 does not lay; Body::refuse_slots refuses it before any capture or launch",
                rows.len(),
                pass_rows(rows)
            )
        })
    }

    /// Each slot's rows planned from its own sequence (module doc): refused
    /// by name before anything moves as [`Body::refuse_slots`] and each
    /// slot's rows ([`Body::refuse_slot_rows`]) refuse, while a verify waits
    /// on the live sequence, and on a failure planted before the launch;
    /// then each slot's rows' inputs written and its sequence moved
    /// ([`Body::plan_slot_rows`]), every row's final streams its own slot's
    /// own row's while they are written.
    fn plan_slots(
        &mut self,
        stream: &CudaStream,
        parked: &mut [&mut GlmSlot],
        rows: &[SlotRange],
        ids: &[u32],
    ) -> Result<(), GpuError> {
        self.refuse_slots(parked, rows)?;
        self.refuse_live_waiting()?;
        self.refuse_slot_rows(parked, rows, ids)?;
        self.planted(Plant::BeforeLaunch)?;
        let fin = program::final_streams(self.cfg.len());
        let swaps = binding(rows)?;
        exchange(&mut self.s, parked, fin, &swaps, false)?;
        let wrote = self.plan_each_slot(stream, parked, rows, ids);
        exchange(&mut self.s, parked, fin, &swaps, true)?;
        wrote?;
        self.planned = Some(SlotsPlanned::of(rows));
        Ok(())
    }

    /// The pass's walk, each row bound to its own slot's lane word, stores
    /// and own row, row `r` into `heads[r]`. Outside a capture it needs its
    /// plan ([`SlotRows::plan_slots`] of the same rows); a capture records
    /// the launches, which read every per-row value from the rows' words and
    /// every slot's buffers by address. On a NextN load each slot's
    /// verify row 0's final streams are then copied into its own pair-0
    /// copy ([`Body::copy_pair0`]). Refused by name as [`Body::refuse_slots`]
    /// refuses, while a verify other than the pass's own (slot 0's rows of
    /// its plan) waits on the live sequence, and for other than a head a
    /// row.
    fn enqueue_slots(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        heads: &mut [Head],
        parked: &mut [&mut GlmSlot],
        rows: &[SlotRange],
    ) -> Result<(), GpuError> {
        self.refuse_slots(parked, rows)?;
        // A verify waiting on the live sequence is the pass's own when its
        // plan laid slot 0's verify rows: those lanes are the ones it runs.
        let own = rows
            .iter()
            .find(|r| r.slot == 0)
            .is_some_and(|r| self.s.lanes.waits_for(r.pos0, r.rows.len()));
        if !own {
            self.refuse_live_waiting()?;
        }
        let total = pass_rows(rows);
        if heads.len() != total {
            return Err(shape(format!(
                "{} heads for a pass of {total} rows",
                heads.len()
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
        let swaps = binding(rows)?;
        exchange(&mut self.s, parked, fin, &swaps, false)?;
        let walked = self.walk_slot_rows(gpu, w, heads, parked, rows);
        exchange(&mut self.s, parked, fin, &swaps, true)?;
        walked?;
        self.copy_pair0(gpu, parked, rows, fin)?;
        if !captured {
            self.planned = None;
        }
        self.planted(Plant::AfterLaunch)
    }

    /// A pass's rows kept: a slot of one row keeps it as it ran (a plain
    /// step, nothing waiting); a slot of two — its verify's rows, standing
    /// in its lanes until the commit — keeps its first `kept` by
    /// [`Body::commit`]'s rule on its own sequence ([`commit_lanes`];
    /// [`Body::commit_live`] on the live one), the slot's draft's records
    /// cut with it (`nextn::NextnSeq::cut` on a parked one). A count outside
    /// its rows, or rows not waiting for their commit, is refused by name.
    fn keep_slot(
        &mut self,
        gpu: &Gpu,
        seq: Option<&mut GlmSlot>,
        r: &SlotRange,
        kept: usize,
    ) -> Result<(), GpuError> {
        let n = r.rows.len();
        if kept == 0 || kept > n {
            return Err(shape(format!(
                "slot {} keeping {kept} of its {n} rows (row 0 always)",
                r.slot
            )));
        }
        if n == 1 {
            return Ok(());
        }
        let pos = r.pos0 + kept as u32;
        let done = match seq {
            None => self.commit_live(gpu, pos)?,
            Some(GlmSlot {
                lane,
                lanes,
                held,
                wrote,
                draft,
                ..
            }) => {
                let count = self.lanes.count() as u32;
                let done = commit_lanes(gpu.stream(), lane, lanes, held, wrote, pos, count)?;
                if let Some(d) = draft.as_mut().filter(|_| done) {
                    d.cut(pos);
                }
                done
            }
        };
        if !done {
            return Err(shape(format!(
                "slot {} keeping {kept} of its {n} rows: its rows are not waiting for their \
                 commit (GpuModel::commit_slots after the pass that ran them)",
                r.slot
            )));
        }
        Ok(())
    }
}
