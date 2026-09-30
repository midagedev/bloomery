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

use bloomery_gpu::head::Head;
use bloomery_gpu::hybrid::Chain;
use bloomery_gpu::linear::delta::row_lane;
use bloomery_gpu::model::{ChainBody, Rows};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError, capturing};
use cuda_core::CudaStream;
use runtime::seqstate::{Kept, Why};

use model::arch::glm5next::place::KdaLanes;

use super::{Body, LANES, Plant, StepInput, shape};
use crate::program;

/// The rows a verify runs: a row a lane.
pub const PAIR_ROWS: usize = 2;

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
        let rows = 1 + usize::from(self.s.row1.is_some());
        let r = self
            .s
            .row_mut(row)
            .ok_or_else(|| shape(format!("row {row} of a pass on a load of {rows} rows")))?;
        r.streams[0].copy_from_host(stream, &self.embd.streams)?;
        r.pos.copy_from_host(stream, &[p])?;
        r.vis.copy_from_host(stream, &[0, p + 1])?;
        r.cnt.copy_from_host(stream, &[p + 1])?;
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
