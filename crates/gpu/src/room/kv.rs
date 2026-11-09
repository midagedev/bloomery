//! The K/V rooms' common owner for every body: the class rule that says where a by-position
//! buffer's pages may live ([`kv_backing`]), the constructor of a room-backed `[rows][width]`
//! tensor ([`RoomTensor`]) and the fit → free expert bytes → fit once more loop a body runs
//! before it stages rows ([`fit_rooms`]). A family declares its buffers as data
//! (`runtime::stores::PosLayout`, which the plan reads too) and attaches through these; nothing
//! here names a model.

use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use bloomery_levers::KvHost;
use cuda_core::{CudaStream, DeviceCopy};
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
    /// The card first, keeping `margin` bytes of it free; a page past that goes to the host.
    Spill { margin: u64 },
}

impl From<KvBacking> for Backing {
    fn from(b: KvBacking) -> Backing {
        match b {
            KvBacking::Card => Backing::Card,
            KvBacking::Host => Backing::Host,
            KvBacking::Spill { margin } => Backing::Spill { margin },
        }
    }
}

/// The backing the `BLOOMERY_KV_HOST` lever's `word` gives a buffer read as `access`, the one
/// map of the lever word. A scanned buffer is read whole every step, so it is card-only
/// whatever the word; a selected buffer follows the word. `margin` is the plan's card margin
/// (`Card::margin_bytes`), the free bytes a `spill` room keeps on the card.
#[must_use]
pub fn kv_backing(word: KvHost, access: Access, margin: u64) -> KvBacking {
    match (access.class(), word) {
        (Class::Card, _) | (Class::Host, KvHost::Off) => KvBacking::Card,
        (Class::Host, KvHost::Spill) => KvBacking::Spill { margin },
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
}

/// The rooms of `body` fitted through `pos + rows` positions. A fit that runs out of card bytes
/// asks the body's residency machine for the shortfall plus a page ([`KvRoomBody::free_expert_bytes`])
/// and fits once more; a room that ran out on host pages (a spill room's) returns at once, since
/// freed card bytes cannot make host memory. A second refusal, a refusing machine or none is the
/// named [`GpuError::NoRoom`] with what was freed. Never inside a capture.
///
/// # Errors
///
/// As above.
pub fn fit_rooms<B: KvRoomBody>(body: &mut B, pos: usize, rows: usize) -> Result<(), GpuError> {
    let through = pos + rows;
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
    use super::{FreedBytes, GpuError, KvBacking, KvRoomBody, fit_rooms, kv_backing};
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
        for word in [KvHost::Off, KvHost::Spill, KvHost::All] {
            assert_eq!(
                kv_backing(word, Access::Scan, 7),
                KvBacking::Card,
                "{word:?}"
            );
        }
        assert_eq!(kv_backing(KvHost::Off, SELECT, 7), KvBacking::Card);
        assert_eq!(
            kv_backing(KvHost::Spill, SELECT, 7),
            KvBacking::Spill { margin: 7 }
        );
        assert_eq!(kv_backing(KvHost::All, SELECT, 7), KvBacking::Host);
    }

    /// A body of rooms that runs out until `freed` expert bytes were asked for.
    struct Body {
        calls: Vec<String>,
        /// `fit_each` refuses until the machine freed, `spill` marking the refusal.
        refuse: usize,
        spill: bool,
        machine: bool,
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
            10
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
    }

    fn body(refuse: usize, spill: bool, machine: bool) -> Body {
        Body {
            calls: Vec::new(),
            refuse,
            spill,
            machine,
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
        assert_eq!(b.calls, ["fit 5..7", "free 11", "fit 5..7"]);
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
