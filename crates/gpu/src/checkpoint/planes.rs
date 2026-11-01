//! The sequence state of a body whose every store is a plane of positions
//! ([`PlaneSeq`]): what a server's prompt cache holds for a session it
//! switches away from, and puts back when the session returns.
//!
//! Such a body keeps no recurrent store and no checkpoint. The K and V rows of
//! every layer, and any other store indexed by position, of the held positions
//! are the whole of its sequence, and a prefix of any length is a valid cut
//! (`app::Keep` keeps every held position). The body lists its stores
//! ([`PlaneBody::lend`]) and nothing more; this owner copies them through the
//! shared state ([`SeqState::save_rows`], [`SeqState::save_rows_async`],
//! [`SeqState::load_rows`]) and holds the model-level rules every such body
//! shares:
//! - a save is refused by name on a poisoned model, at 0 positions, and when
//!   the body's own record of what its stores hold is short of the model's
//!   position;
//! - a resume is refused by name, nothing copied, for another model's state
//!   (file, card, context) or store layout ([`Identity`]), for a prefix of no
//!   positions, and while the state's copy is still running; the model is
//!   reset first unless it is empty, then stands at the positions put back
//!   ([`PlaneBody::stand`]).
//!
//! A [`PlaneSeq`] is of one body ([`PlaneBody`]): the state of another body is
//! another type, so a seat that is handed a state it did not take refuses it
//! at the cache's downcast.

use std::marker::PhantomData;
use std::sync::Arc;

use cuda_core::CudaStream;
use model::arch::place_hybrid::KvLayout;

use super::saved::{Identity, Lent, Pending, SaveLane, SeqState, SeqView, Spans, Streams};
use super::staging::NoRoom;
use crate::model::ChainBody;
use crate::{DeviceTensor, GpuError, GpuModel};

/// What a body gives for its sequence to be saved and put back.
pub trait PlaneBody: ChainBody + 'static {
    /// What the state's refusals name.
    const WHAT: &'static str;

    /// What a state of this load belongs to.
    fn who(&self) -> &Identity;

    /// The rows of the first `n` positions of every store the sequence holds,
    /// in an order that is the body's alone and the same on every call.
    /// Refused by name when a store is shorter than `n` positions.
    fn lend(&mut self, n: usize) -> Result<Vec<Spans<'_>>, GpuError>;

    /// The positions the body's own record says its stores hold, `None` for a
    /// body that keeps none (the model's position is then the only one).
    fn written(&self) -> Option<u32> {
        None
    }

    /// After `n` positions were put back on an empty model: the body's record
    /// stands at `n`.
    fn stand(&mut self, _n: u32) {}
}

/// A sequence state of the body `B` on the host ([`seq_save`]): the shared
/// state of the module doc's rows. Its kind is the body's.
pub struct PlaneSeq<B> {
    state: Arc<SeqState<()>>,
    body: PhantomData<fn() -> B>,
}

impl<B> PlaneSeq<B> {
    fn new(state: Arc<SeqState<()>>) -> PlaneSeq<B> {
        PlaneSeq {
            state,
            body: PhantomData,
        }
    }

    /// The positions it holds.
    #[must_use]
    pub fn positions(&self) -> u32 {
        self.state.positions()
    }

    /// The host bytes it holds.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.state.bytes()
    }

    /// The longest prefix of at most `n` positions a model holds right after
    /// the state is put back: every position the state holds up to `n`.
    #[must_use]
    pub fn keep_point(&self, n: u32) -> u32 {
        self.state.keep_point(n)
    }
}

impl<B: 'static> SeqView for PlaneSeq<B> {
    fn positions(&self) -> u32 {
        PlaneSeq::positions(self)
    }

    fn bytes(&self) -> usize {
        PlaneSeq::bytes(self)
    }

    fn keep_point(&self, n: u32) -> u32 {
        PlaneSeq::keep_point(self, n)
    }
}

fn shape<B: PlaneBody>(detail: impl Into<String>) -> GpuError {
    GpuError::Shape {
        what: B::WHAT,
        detail: detail.into(),
    }
}

/// The rows of the first `n` positions of one head-major plane pair's store
/// `t` (head `h`, position `p` at row `h · ctx + p`, `width` values a row):
/// `heads` runs of `n · width` values, `ctx · width` apart.
pub fn head_major(
    t: &mut DeviceTensor<u16>,
    (heads, ctx, width): (usize, usize, usize),
    n: usize,
) -> Result<Spans<'_>, GpuError> {
    Spans::by_position(t.buf_mut(), heads, ctx * width, n, width)
}

/// What every save takes before its copy: the model's position as the state's
/// positions, the stores lent for them, and `send` (the copy, on the engine
/// stream or on a lane) run on them. A step launched ahead is read first
/// ([`GpuModel::settle_ahead`]), its fault the save's error.
fn take<B: PlaneBody, R>(
    m: &mut GpuModel<B>,
    send: impl FnOnce(&CudaStream, Identity, u32, &Lent<'_, '_, '_>) -> Result<R, GpuError>,
) -> Result<R, GpuError> {
    m.settle_ahead(B::WHAT)?;
    if let Some(fault) = m.poisoned() {
        return Err(GpuError::Poisoned {
            what: B::WHAT,
            fault,
        });
    }
    let pos = m.pos();
    let (gpu, _, body) = m.body_parts(B::WHAT)?;
    if pos == 0 {
        return Err(shape::<B>(
            "a sequence of 0 positions has no rows to save".to_owned(),
        ));
    }
    if let Some(end) = body.written()
        && end < pos
    {
        return Err(shape::<B>(format!(
            "a sequence of {pos} positions whose stores hold {end}"
        )));
    }
    let who = body.who().clone();
    let mut spans = body.lend(pos as usize)?;
    let lent = Lent {
        positional: &mut spans,
        recurrent: &mut [],
    };
    send(gpu.stream(), who, pos, &lent)
}

/// The sequence state on the host ([`PlaneSeq`], the module doc): the rows of
/// the positions below `m`'s position, copied on the engine stream. Refused by
/// name as the module doc says.
pub fn seq_save<B: PlaneBody>(m: &mut GpuModel<B>) -> Result<PlaneSeq<B>, GpuError> {
    let state = take(m, |stream, who, pos, lent| {
        SeqState::save_rows(stream, who, pos, lent)
    })?;
    Ok(PlaneSeq::new(Arc::new(state)))
}

/// The host bytes a sequence save would hold, without reading the card.
pub fn seq_save_bytes<B: PlaneBody>(m: &mut GpuModel<B>) -> Result<u64, GpuError> {
    let pos = m.pos();
    let (_, _, body) = m.body_parts(B::WHAT)?;
    let mut spans = body.lend(pos as usize)?;
    Lent {
        positional: &mut spans,
        recurrent: &mut [],
    }
    .bytes(0)
}

/// A state [`seq_save_async`] saved with the copy still in flight, or [`NoRoom`].
pub type SavedAsync<B> = Result<(PlaneSeq<B>, Pending<()>), NoRoom>;

/// [`seq_save`] whose copy runs on `lane`'s stream into its pinned staging and
/// whose call returns once it is enqueued ([`SeqState::save_rows_async`]): the
/// state and its copy in flight, which the model's rows are not cut, reset or
/// replaced before; or [`NoRoom`], nothing begun, when the staging holds no
/// free extent for it. Refused by name as [`seq_save`] refuses.
pub fn seq_save_async<B: PlaneBody>(
    m: &mut GpuModel<B>,
    lane: &SaveLane,
) -> Result<SavedAsync<B>, GpuError> {
    let pending = take(m, |engine, who, pos, lent| {
        SeqState::save_rows_async(Streams { lane, engine }, who, pos, lent)
    })?;
    Ok(pending.map(|pending| (PlaneSeq::new(pending.shared()), pending)))
}

/// A save lane on `m`'s card sized to `limit` pinned bytes of states
/// ([`SaveLane::new`]): load-time only.
pub fn seq_lane<B: PlaneBody>(m: &mut GpuModel<B>, limit: u64) -> Result<SaveLane, GpuError> {
    let (gpu, _, _) = m.body_parts(B::WHAT)?;
    SaveLane::new(gpu.context(), limit)
}

/// Replace `m`'s sequence with `s`, which [`seq_save`] took of this load:
/// [`seq_resume_prefix`] of every position.
pub fn seq_resume<B: PlaneBody>(m: &mut GpuModel<B>, s: &PlaneSeq<B>) -> Result<(), GpuError> {
    seq_resume_prefix(m, s, s.positions()).map(drop)
}

/// [`seq_resume`] of the first `n` positions of `s` ([`PlaneSeq::keep_point`]):
/// the model stands at that many, which it returns, with only those
/// positions' rows moved to the card ([`SeqState::load_rows`]). Refused by
/// name as the module doc says, before the model is reset when the state or
/// the prefix is the cause. `s`'s copy must have landed.
pub fn seq_resume_prefix<B: PlaneBody>(
    m: &mut GpuModel<B>,
    s: &PlaneSeq<B>,
    n: u32,
) -> Result<u32, GpuError> {
    if let Some(fault) = m.poisoned() {
        return Err(GpuError::Poisoned {
            what: B::WHAT,
            fault,
        });
    }
    let at = s.keep_point(n);
    if at == 0 {
        return Err(shape::<B>(format!(
            "a prefix of 0 positions of a state of {} (asked {n}): reset the model instead",
            s.positions()
        )));
    }
    if m.pos() != 0 {
        m.reset()?;
    }
    m.run_rows(at as usize, B::WHAT, |gpu, _, body, _, _| {
        let who = body.who().clone();
        let held = {
            let mut spans = body.lend(s.positions() as usize)?;
            let mut lent = Lent {
                positional: &mut spans,
                recurrent: &mut [],
            };
            s.state.load_rows(gpu.stream(), &who, &mut lent, at)?.at
        };
        body.stand(held);
        Ok(false)
    })?;
    Ok(at)
}

/// The host bytes a state of `positions` positions holds on the load whose
/// stores `kv` sizes over `layers` layers: the stores are the plan's, f16
/// planes of positions, so a state is the plan's store bytes at that many
/// positions.
#[must_use]
pub fn seq_bytes(kv: &KvLayout, layers: usize, positions: usize) -> u64 {
    kv.bytes(0..layers, positions as u64)
}
