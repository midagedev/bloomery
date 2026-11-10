//! The K/V rooms' common owner for every body: the class rule that says where a by-position
//! buffer's pages may live ([`kv_backing`]), the constructor of a room-backed `[rows][width]`
//! tensor ([`RoomTensor`]), the free bytes a card keeps against its budget ([`CardKeep`]) and
//! the fill first → fit → free expert bytes → fit once more loop a body runs before it stages
//! rows ([`fit_rooms`]). A family declares its buffers as data
//! (`runtime::stores::PosLayout`, which the plan reads too) and attaches through these; nothing
//! here names a model.

use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use ::model::placement::Plan;
use ::model::placement::card_budget;
use ::model::placement::elastic::fill_first;
use bloomery_levers::KvHost;
use cuda_core::{CudaContext, CudaStream, DeviceCopy};
use runtime::stores::{Access, Class, PosBuf};

use super::{Backing, Roomed, Run, Views};
use crate::{DeviceTensor, GpuError};

const WHAT: &str = "kv rooms";

/// Where a by-position buffer's pages are backed: the public mirror of the room's own choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvBacking {
    /// The device's own memory.
    Card,
    /// Host memory on the device's NUMA node, reached over its link.
    Host,
    /// The card first, keeping `keep` of its free bytes; a page past that goes to the host.
    Spill { keep: CardKeep },
}

impl From<KvBacking> for Backing {
    fn from(b: KvBacking) -> Backing {
        match b {
            KvBacking::Card => Backing::Card,
            KvBacking::Host => Backing::Host,
            KvBacking::Spill { keep } => Backing::Spill { keep },
        }
    }
}

/// What a load keeps free of its stage card: the plan's `margin` (`Card::margin_bytes`) of the
/// free bytes the card's `budget` leaves it (`card_budget::of`, `None` for the card's own
/// bytes). A spill room's page and the fill's take-back ([`fit_rooms`]) read the card through
/// it, so under `BLOOMERY_CARD_BUDGET` the margin is kept against the budget, as the plan and
/// the elastic settle keep it, and never against the driver's free bytes alone. A budget the
/// plan lowered itself on a unified machine (`Plan::budget_from_pool`) is no keep: the card and
/// the host there are one pool, so a page moved to the host frees none of it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CardKeep {
    pub margin: u64,
    pub budget: Option<u64>,
    /// The card's free bytes when the load opened ([`crate::Gpu::opened_free`], the card
    /// reading's `before=`): what the budget is counted from. A keep under a budget with none
    /// is refused by name where it reads the card.
    pub opened: Option<u64>,
}

impl CardKeep {
    /// The keep of `plan`'s `card`: its margin, under the budget the plan's caller gave, counted
    /// from `opened`, the card's free bytes when the load opened.
    #[must_use]
    pub fn of(plan: &Plan<'_>, card: &::model::placement::Card, opened: Option<u64>) -> CardKeep {
        CardKeep {
            opened,
            margin: card.margin_bytes,
            budget: if plan.budget_from_pool {
                None
            } else {
                card_budget::of(&plan.card_budget, card)
            },
        }
    }

    /// The card's free bytes under the budget (`card_budget::free_under`), read on `ctx`.
    ///
    /// # Errors
    ///
    /// The driver's refusal to bind the context or read the device's memory.
    pub fn free(self, ctx: &CudaContext) -> Result<u64, GpuError> {
        ctx.bind_to_thread()?;
        self.free_bound()
    }

    /// The open reading a budgeted keep counts from: `None` with no budget, the reading under
    /// one, and a refusal by name for a budget with no reading.
    fn opened_under(self) -> Result<u64, GpuError> {
        match (self.budget, self.opened) {
            (None, o) => Ok(o.unwrap_or(0)),
            (Some(_), Some(o)) => Ok(o),
            (Some(_), None) => Err(GpuError::state(
                WHAT,
                "the card's free bytes at the load's open, which a card budget counts from",
            )),
        }
    }

    /// [`CardKeep::free`] on the context already bound to the thread.
    pub(crate) fn free_bound(self) -> Result<u64, GpuError> {
        let opened = self.opened_under()?;
        let (free, _) = crate::current_mem_info()?;
        Ok(card_budget::free_under(free as u64, opened, self.budget))
    }

    /// Whether one more card page of `page` bytes would leave the card under the margin
    /// (`card_budget::past_margin`), on the context already bound to the thread.
    pub(crate) fn past_margin(self, page: u64) -> Result<bool, GpuError> {
        let opened = self.opened_under()?;
        let (free, _) = crate::current_mem_info()?;
        Ok(card_budget::past_margin(
            free as u64,
            opened,
            self.budget,
            self.margin,
            page,
        ))
    }
}

/// The backing the `BLOOMERY_KV_HOST` lever's `word` gives a buffer read as `access`, the one
/// map of the lever word. A scanned buffer is read whole every step, so it is card-only
/// whatever the word; a selected buffer follows the word. `keep` is what a `spill` room keeps
/// free of the card ([`CardKeep`]).
#[must_use]
pub fn kv_backing(word: KvHost, access: Access, keep: CardKeep) -> KvBacking {
    match (access.class(), word) {
        (Class::Card, _) | (Class::Host, KvHost::Off) => KvBacking::Card,
        (Class::Host, KvHost::Spill) => KvBacking::Spill { keep },
        (Class::Host, KvHost::All) => KvBacking::Host,
    }
}

/// A `[rows][width]` tensor of `T` over a room: the buffer reads and writes as any
/// [`DeviceTensor`] (`Deref`), its pages backed as far as the positions a body fitted reach.
pub struct RoomTensor<T: DeviceCopy> {
    room: Roomed<Option<DeviceTensor<T>>>,
}

impl<T: DeviceCopy> RoomTensor<T> {
    /// A tensor of the rows `buf` holds over `positions` positions, each row `buf.row / size_of::<T>()`
    /// values wide, on `stream`'s device, backed where `backing` says, no row backed until a
    /// fit reaches it. `what` names the store for a fit's refusal. Refused by name for a buffer
    /// of more than one run (a tensor is one run of rows), a row that does not hold whole `T`s,
    /// or a size past usize.
    ///
    /// # Errors
    ///
    /// The refusals above, and the room's own (a device that cannot host a page).
    pub fn new(
        stream: &Arc<CudaStream>,
        buf: &PosBuf,
        positions: usize,
        what: &'static str,
        backing: KvBacking,
    ) -> Result<RoomTensor<T>, GpuError> {
        let size = size_of::<T>();
        let row =
            usize::try_from(buf.row).map_err(|_| GpuError::shape(WHAT, "a row past usize"))?;
        if buf.runs != 1 || size == 0 || row % size != 0 {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{}: a tensor is one run of rows of whole values ({} runs, {} B rows of {size} B values)",
                    buf.what, buf.runs, buf.row
                ),
            ));
        }
        let per = usize::try_from(buf.per).map_err(|_| GpuError::shape(WHAT, "per past usize"))?;
        let rows = positions.div_ceil(per.max(1));
        let (cols, len) = (
            row / size,
            rows.checked_mul(row / size)
                .ok_or_else(|| GpuError::shape(WHAT, format!("{rows} rows pass usize")))?,
        );
        let run = Run {
            at: 0,
            row,
            per,
            rows,
        };
        let room = Roomed::with_backing(
            Arc::clone(stream),
            len,
            &[run],
            what,
            backing.into(),
            // The buffer holds exactly `rows · cols` values, which `from_buf` checks.
            |[b]| DeviceTensor::from_buf(b, rows, cols).ok(),
        )?;
        if room.is_none() {
            return Err(GpuError::state(WHAT, "the room's tensor"));
        }
        Ok(RoomTensor { room })
    }

    /// The rows of the first `positions` positions backed ([`Roomed::fit`]). Never inside a capture.
    ///
    /// # Errors
    ///
    /// The room's named [`GpuError::NoRoom`].
    pub fn fit(&mut self, positions: usize) -> Result<(), GpuError> {
        self.room.fit(positions)
    }

    /// The bytes of new pages a fit of the first `positions` positions still asks.
    #[must_use]
    pub fn asks(&self, positions: usize) -> usize {
        self.room.asks(positions)
    }

    /// The pages a write from position `pos` on reaches made this sequence's own, given memory no
    /// other sequence's store maps. Never inside a capture.
    ///
    /// # Errors
    ///
    /// The room's refusal.
    pub fn own_from(&mut self, pos: usize) -> Result<(), GpuError> {
        self.room.own_from(pos)
    }

    /// Every page past the rows reached given back; the sequence's end.
    ///
    /// # Errors
    ///
    /// The room's refusal.
    pub fn settle(&mut self) -> Result<(), GpuError> {
        self.room.settle()
    }

    /// Bytes of the pages on the card and on the host the tensor's room backs with memory of
    /// its own.
    #[must_use]
    pub fn mapped(&self) -> (usize, usize) {
        (self.room.mapped(), self.room.host_mapped())
    }
}

impl<T: DeviceCopy> Deref for RoomTensor<T> {
    type Target = DeviceTensor<T>;

    fn deref(&self) -> &DeviceTensor<T> {
        self.room
            .as_ref()
            .expect("a RoomTensor holds its tensor from construction")
    }
}

impl<T: DeviceCopy> DerefMut for RoomTensor<T> {
    fn deref_mut(&mut self) -> &mut DeviceTensor<T> {
        self.room
            .as_mut()
            .expect("a RoomTensor holds its tensor from construction")
    }
}

/// What a body's rooms hold of the card's expert stacks after a fit asked them to give bytes
/// back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FreedBytes {
    pub bytes: u64,
    /// Slots a layer gave back.
    pub slots: usize,
    /// Layers that gave them.
    pub layers: usize,
}

/// What an elastic load's fill holds of the stage card for [`fit_rooms`]'s fill-first step:
/// the `held` bytes the fill backed above the plan and not yet given back, what the card keeps
/// ([`CardKeep`]) and the `page` the take-back adds past the shortfall.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FillHeld {
    pub held: u64,
    pub keep: CardKeep,
    pub page: u64,
}

/// A body whose stores are rooms: what [`fit_rooms`] needs of it.
pub trait KvRoomBody {
    /// The body's name for a refusal.
    const WHAT: &'static str;

    /// Every room fitted through `through` positions and its pages from `pos` on made this
    /// sequence's own, room by room.
    ///
    /// # Errors
    ///
    /// The first room's named [`GpuError::NoRoom`].
    fn fit_each(&mut self, pos: usize, through: usize) -> Result<(), GpuError>;

    /// The bytes of new pages a fit through `through` positions still asks, every room's.
    fn asks(&self, through: usize) -> usize;

    /// The residency machine's demand for `want` card bytes of expert stacks.
    ///
    /// # Errors
    ///
    /// The machine's refusal, or a body with no machine.
    fn free_expert_bytes(&mut self, want: u64) -> Result<FreedBytes, GpuError>;

    /// A give-back that [`fit_rooms`] asked for `asked` bytes and the machine made
    /// (`freed`): the body counts it and keeps it for its caller's record. Nothing by
    /// default, for a body that tallies no moves.
    fn gave_back(&mut self, _asked: u64, _freed: &FreedBytes) {}

    /// The fill this load holds above its plan ([`FillHeld`]); `None` by default, for a load
    /// that is not elastic.
    fn fill_held(&self) -> Option<FillHeld> {
        None
    }

    /// The stage card's free bytes under `keep` ([`CardKeep::free`]), read only for a fit that
    /// asks new pages on a load whose [`KvRoomBody::fill_held`] holds bytes.
    ///
    /// # Errors
    ///
    /// The driver's refusal; by default a body with no card to read.
    fn card_free(&self, _keep: CardKeep) -> Result<u64, GpuError> {
        Err(GpuError::state(
            Self::WHAT,
            "a stage card to read the free bytes of",
        ))
    }

    /// The fill-first take-back of `asked` bytes that the machine refused (`why`): the fit runs
    /// as it would have without one, and the body keeps the refusal for its caller's record.
    /// Nothing by default.
    fn fill_refused(&mut self, _asked: u64, _why: &GpuError) {}
}

/// The rooms of `body` fitted through `pos + rows` positions.
///
/// Fill first: on a load whose fill holds card bytes above the plan ([`KvRoomBody::fill_held`]),
/// a fit whose new pages would leave the card under its margin first takes back the shortfall
/// plus a page, at most what the fill holds (`placement::elastic::fill_first`), through the
/// machine ([`KvRoomBody::free_expert_bytes`], recorded by [`KvRoomBody::gave_back`]); a planned
/// expert is never asked for here. A refused take-back is kept by the body
/// ([`KvRoomBody::fill_refused`]) and the fit runs on. A fit that asks no new page reads nothing.
///
/// Then the fit: one that runs out of card bytes asks the body's residency machine for the
/// shortfall plus a page and fits once more; a room that ran out on host pages (a spill room's)
/// returns at once, since freed card bytes cannot make host memory. A second refusal, a refusing
/// machine or none is the named [`GpuError::NoRoom`] with what was freed. Never inside a capture.
///
/// # Errors
///
/// As above, and the card's free bytes unreadable for the fill-first step.
pub fn fit_rooms<B: KvRoomBody>(body: &mut B, pos: usize, rows: usize) -> Result<(), GpuError> {
    let through = pos + rows;
    take_fill_first(body, through)?;
    let first = match body.fit_each(pos, through) {
        Ok(()) => return Ok(()),
        Err(e) => e,
    };
    let GpuError::NoRoom {
        free, page, spill, ..
    } = &first
    else {
        return Err(first);
    };
    if *spill {
        return Err(first);
    }
    let page = *page;
    let free = match (*free).map_or_else(|| crate::current_mem_info().map(|m| m.0), Ok) {
        Ok(free) => free,
        Err(e) => {
            return Err(first.noting_room(format!(
                "the residency machine was not asked: the card's free bytes are unreadable ({e})"
            )));
        }
    };
    let want = body.asks(through).saturating_sub(free).saturating_add(page);
    let freed = match body.free_expert_bytes(want as u64) {
        Ok(f) => f,
        Err(e) => {
            return Err(first.noting_room(format!(
                "the residency machine freed 0 B of the {want} B the fit was short: {e}"
            )));
        }
    };
    body.gave_back(want as u64, &freed);
    body.fit_each(pos, through).map_err(|e| {
        e.noting_room(format!(
            "after the residency machine freed {} B of expert stacks ({} slots a layer over {} \
             layers)",
            freed.bytes, freed.slots, freed.layers
        ))
    })
}

/// [`fit_rooms`]'s fill-first step for a fit through `through` positions: nothing on a load with
/// no fill held, or for a fit that asks no new page; else the card read under its keep and the
/// take-back `fill_first` names asked of the machine. A body that opens a span in which the
/// machine refuses give-backs (a prompt call) runs this for the span's last position before it
/// opens, so the span's fits find the fill already taken back.
///
/// # Errors
///
/// The card's free bytes unreadable.
pub fn take_fill_first<B: KvRoomBody>(body: &mut B, through: usize) -> Result<(), GpuError> {
    let ask = match body.fill_held() {
        Some(h) if h.held > 0 => {
            let asks = body.asks(through) as u64;
            if asks == 0 {
                return Ok(());
            }
            fill_first(asks, body.card_free(h.keep)?, h.keep.margin, h.page, h.held)
        }
        _ => return Ok(()),
    };
    if ask == 0 {
        return Ok(());
    }
    match body.free_expert_bytes(ask) {
        Ok(freed) => body.gave_back(ask, &freed),
        Err(e) => body.fill_refused(ask, &e),
    }
    Ok(())
}

impl<T: DeviceCopy> Views for DeviceTensor<T> {
    fn dissolve(self) {
        self.into_buf().dissolve();
    }
}

impl<V: Views> Views for Option<V> {
    fn dissolve(self) {
        if let Some(v) = self {
            v.dissolve();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CardKeep, FillHeld, FreedBytes, GpuError, KvBacking, KvRoomBody, fit_rooms, kv_backing,
    };
    use bloomery_levers::KvHost;
    use runtime::stores::Access;

    const SELECT: Access = Access::Select { rows: 4 };

    /// The class rule: a scanned buffer is card-only whatever the lever word, a selected one
    /// follows it.
    ///
    /// FAIL-first: `(Class::Card, _)` dropped from the map sends a scanned buffer under `all`
    /// to the host and the first row goes red.
    #[test]
    fn a_scanned_buffer_is_card_only_and_a_selected_one_follows_the_word() {
        let keep = CardKeep {
            margin: 7,
            budget: Some(9),
            opened: Some(9),
        };
        for word in [KvHost::Off, KvHost::Spill, KvHost::All] {
            assert_eq!(
                kv_backing(word, Access::Scan, keep),
                KvBacking::Card,
                "{word:?}"
            );
        }
        assert_eq!(kv_backing(KvHost::Off, SELECT, keep), KvBacking::Card);
        assert_eq!(
            kv_backing(KvHost::Spill, SELECT, keep),
            KvBacking::Spill { keep }
        );
        assert_eq!(kv_backing(KvHost::All, SELECT, keep), KvBacking::Host);
    }

    /// A body of rooms that runs out until `freed` expert bytes were asked for.
    struct Body {
        calls: Vec<String>,
        /// `fit_each` refuses until the machine freed, `spill` marking the refusal.
        refuse: usize,
        spill: bool,
        machine: bool,
        /// What an elastic load's fill holds, and the card's free bytes a read returns.
        fill: Option<FillHeld>,
        free: u64,
        asks: usize,
        reads: std::cell::Cell<usize>,
    }

    fn no_room(spill: bool) -> GpuError {
        GpuError::NoRoom {
            what: "test rooms",
            detail: "a fit".to_owned(),
            source: Box::new(GpuError::state("test", "refused")),
            asked: 10,
            free: Some(3),
            page: 4,
            spill,
        }
    }

    impl KvRoomBody for Body {
        const WHAT: &'static str = "test body";

        fn fit_each(&mut self, pos: usize, through: usize) -> Result<(), GpuError> {
            self.calls.push(format!("fit {pos}..{through}"));
            if self.refuse > 0 {
                self.refuse -= 1;
                return Err(no_room(self.spill));
            }
            Ok(())
        }

        fn asks(&self, _through: usize) -> usize {
            self.asks
        }

        fn free_expert_bytes(&mut self, want: u64) -> Result<FreedBytes, GpuError> {
            self.calls.push(format!("free {want}"));
            if !self.machine {
                return Err(GpuError::state("test", "no machine"));
            }
            Ok(FreedBytes {
                bytes: want,
                slots: 2,
                layers: 3,
            })
        }

        fn gave_back(&mut self, asked: u64, _freed: &FreedBytes) {
            self.calls.push(format!("gave {asked}"));
        }

        fn fill_held(&self) -> Option<FillHeld> {
            self.fill
        }

        fn card_free(&self, keep: CardKeep) -> Result<u64, GpuError> {
            assert_eq!(keep.margin, 50, "the fill's own keep is read");
            self.reads.set(self.reads.get() + 1);
            Ok(self.free)
        }

        fn fill_refused(&mut self, asked: u64, _why: &GpuError) {
            self.calls.push(format!("refused {asked}"));
        }
    }

    fn body(refuse: usize, spill: bool, machine: bool) -> Body {
        Body {
            calls: Vec::new(),
            refuse,
            spill,
            machine,
            fill: None,
            free: 0,
            asks: 10,
            reads: std::cell::Cell::new(0),
        }
    }

    /// A body of an elastic load whose fill holds `held` bytes over a margin of 50, a 4-byte
    /// page, with `free` card bytes and every fit holding.
    fn filled(held: u64, free: u64, machine: bool) -> Body {
        Body {
            fill: Some(FillHeld {
                held,
                keep: CardKeep {
                    margin: 50,
                    budget: None,
                    opened: None,
                },
                page: 4,
            }),
            free,
            ..body(0, true, machine)
        }
    }

    /// Fill first: a fit of 10 new bytes on 55 free over a margin of 50 takes back the 5 it
    /// falls short plus a page from the fill, recorded, before it fits; one within the slack
    /// reads the card and takes nothing; a refused take-back is kept and the fit runs on.
    ///
    /// FAIL-first: a `fit_rooms` without the step ends with `["fit 5..7"]` on the first case; one
    /// that fails the fit on a refused take-back reds the last.
    #[test]
    fn a_fit_past_the_margin_takes_back_the_fill_before_it_fits() {
        let mut b = filled(100, 55, true);
        fit_rooms(&mut b, 5, 2).expect("the fit holds");
        assert_eq!(b.calls, ["free 9", "gave 9", "fit 5..7"]);
        let mut b = filled(100, 60, true);
        fit_rooms(&mut b, 0, 1).expect("the fit holds");
        assert_eq!(
            (b.calls.as_slice(), b.reads.get()),
            (&["fit 0..1".to_owned()][..], 1)
        );
        let mut b = filled(100, 55, false);
        fit_rooms(&mut b, 0, 1).expect("a refused take-back still fits");
        assert_eq!(b.calls, ["free 9", "refused 9", "fit 0..1"]);
    }

    /// A load with no fill, a fill that holds nothing and a fit that asks no page (a decode
    /// step within its backed pages) never read the card and never ask the machine.
    ///
    /// FAIL-first: a step that reads the card before it checks the asks reds the third case
    /// (one read).
    #[test]
    fn a_fit_with_nothing_held_or_asked_reads_nothing() {
        let mut unasked = filled(100, 0, true);
        unasked.asks = 0;
        for mut b in [body(0, true, true), filled(0, 0, true), unasked] {
            fit_rooms(&mut b, 0, 1).expect("the fit holds");
            assert_eq!(
                (b.calls.as_slice(), b.reads.get()),
                (&["fit 0..1".to_owned()][..], 0)
            );
        }
    }

    /// The retry order: fit, ask the machine for the shortfall (asked − free) plus a page, fit
    /// once more.
    ///
    /// FAIL-first: a loop that skips the second `fit_each` ends with two calls and the
    /// assertion goes red.
    #[test]
    fn a_refused_fit_asks_the_machine_once_and_fits_again() {
        let mut b = body(1, false, true);
        fit_rooms(&mut b, 5, 2).expect("the second fit holds");
        assert_eq!(b.calls, ["fit 5..7", "free 11", "gave 11", "fit 5..7"]);
    }

    /// A spill room's refusal is host memory, which freed card bytes cannot make: the machine is
    /// not asked.
    ///
    /// FAIL-first: without the `spill` check the machine is asked and `free 11` shows in the calls.
    #[test]
    fn a_spill_room_that_ran_out_does_not_ask_the_machine() {
        let mut b = body(1, true, true);
        let e = fit_rooms(&mut b, 0, 1).expect_err("the refusal stands");
        assert!(matches!(e, GpuError::NoRoom { spill: true, .. }));
        assert_eq!(b.calls, ["fit 0..1"]);
    }

    /// A second refusal and a machine that frees nothing are the named refusal, each noting what
    /// the machine did.
    #[test]
    fn a_second_refusal_and_a_refusing_machine_are_named() {
        let mut twice = body(2, false, true);
        let e = fit_rooms(&mut twice, 0, 1).expect_err("two refusals");
        assert!(e.to_string().contains("freed 11 B of expert stacks"), "{e}");
        let mut none = body(1, false, false);
        let e = fit_rooms(&mut none, 0, 1).expect_err("no machine");
        assert!(e.to_string().contains("freed 0 B of the 11 B"), "{e}");
    }
}
