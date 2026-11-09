//! The elastic expert tenant's common owner: experts are what the card gives
//! up when a later allocation does not fit and takes back when it has room.
//! A body with a residency machine offers the two moves through
//! [`ElasticBody`]; [`with_room`] runs an allocating operation, asks the
//! machine for the shortfall on a refusal and runs it once more; and
//! [`settle_to_margin`] is the move a load makes after its warm-up, once
//! every buffer serving uses exists: give back what the card holds past its
//! margin, or fill what it has free. [`super::kv::fit_rooms`] is the same
//! loop over the K/V rooms and stays its own entry for the bodies on it.
//!
//! Every give-back and every fill is counted ([`ElasticStats`]) and reported
//! to the caller ([`Elastic`]) so a binary prints it as a record line; a gate's
//! steady state after warm-up holds the give-back count at zero. Nothing here
//! names a model: a family implements the trait by naming its machine parts.

pub use ::model::placement::elastic::ElasticStats;
use ::model::placement::elastic::{fill_allowed, fill_slots};

use super::kv::{FreedBytes, KvRoomBody};
use crate::GpuError;

/// A body whose expert stacks a residency machine can shrink and grow: the
/// give-back is [`KvRoomBody::free_expert_bytes`], the one the K/V rooms'
/// fit already asks for, so the two loops move the same experts the same way.
pub trait ElasticBody: KvRoomBody {
    /// At most `room` device bytes of withdrawn expert slots backed again
    /// (`SwapMachine::fill_bytes`): the same `k` slots on every machine layer,
    /// the largest that fits.
    ///
    /// # Errors
    ///
    /// The machine's refusal, or a body with no machine.
    fn fill_expert_bytes(&mut self, room: u64) -> Result<FilledBytes, GpuError>;

    /// The machine's layer count and the bytes one slot takes on it: the
    /// quantum's terms. `None` for a body with no machine.
    fn expert_quantum(&self) -> Option<(u64, u64)>;

    /// The slots the machine has withdrawn on every layer it holds stacks of:
    /// the fewest over those layers, which is what one fill can restore a
    /// layer. 0 for a body with no machine or nothing withdrawn.
    fn withdrawn_slots(&self) -> usize;

    /// The counters this body's moves are tallied in.
    fn elastic_stats(&mut self) -> &mut ElasticStats;

    /// The moves made where no caller had a record to print them
    /// ([`KvRoomBody::gave_back`], a K/V fit's give-back), oldest first,
    /// emptied by the call.
    fn take_moves(&mut self) -> Vec<Elastic>;
}

/// What a fill backed: slots a layer, their device bytes, layers filled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FilledBytes {
    pub bytes: u64,
    pub slots: usize,
    pub layers: usize,
}

/// Why a settle held when the free bytes above the margin hold no whole quantum past the
/// keep: the common case of a card that is full enough, which a quiet boundary does not print.
pub const HELD_NO_QUANTUM: &str = "the free bytes above the margin hold no quantum past the keep";

/// One move of the tenant, reported to the caller for its record line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Elastic {
    /// Expert bytes given back at `site` so `asked` bytes could be made.
    GiveBack {
        site: &'static str,
        asked: u64,
        freed: FreedBytes,
    },
    /// Withdrawn slots filled from `free` bytes above `margin`, by the settle
    /// at `site`.
    Fill {
        site: &'static str,
        boundary: u64,
        free: u64,
        margin: u64,
        filled: FilledBytes,
    },
    /// Nothing moved at the settle at `site`, and why (a quiet settle is not a failure).
    Held { site: &'static str, why: String },
}

/// Whether `e` is the card's own memory running out: a room's named refusal
/// on a card-class room, or the driver's out-of-memory on a plain allocation.
fn out_of_card_memory(e: &GpuError) -> bool {
    match e {
        GpuError::NoRoom { spill, .. } => !*spill,
        GpuError::Driver { source, .. } => {
            source.0 == cuda_core::sys::cudaError_enum_CUDA_ERROR_OUT_OF_MEMORY
        }
        _ => false,
    }
}

/// `op` on `body`, once more after the machine gave `ask` bytes back when
/// the first run ran out of card memory ([`out_of_card_memory`]); `site`
/// names the allocation for the record and the refusal. A second refusal, a
/// refusing machine or none is the first refusal, noting what was freed.
/// Never inside a capture or an open prompt call (the machine refuses the
/// give-back by name there and names the next boundary). `report` receives
/// the give-back, when one ran, for the caller's record line.
///
/// # Errors
///
/// As above; an error of `op` that is not the card's memory is returned as it
/// is.
pub fn with_room<B: ElasticBody, T>(
    body: &mut B,
    site: &'static str,
    ask: u64,
    mut op: impl FnMut(&mut B) -> Result<T, GpuError>,
    mut report: impl FnMut(Elastic),
) -> Result<T, GpuError> {
    let first = match op(body) {
        Ok(t) => return Ok(t),
        Err(e) if out_of_card_memory(&e) => e,
        Err(e) => return Err(e),
    };
    let freed = match body.free_expert_bytes(ask) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::shape(
                B::WHAT,
                format!(
                    "{site}: {ask} B would not fit ({first}) and the residency machine freed 0 B \
                     of it: {e}"
                ),
            ));
        }
    };
    body.elastic_stats().gave_back(freed.bytes);
    report(Elastic::GiveBack {
        site,
        asked: ask,
        freed,
    });
    op(body).map_err(|e| {
        GpuError::shape(
            B::WHAT,
            format!(
                "{site}: {ask} B would not fit after the residency machine freed {} B of expert \
                 stacks ({} slots a layer over {} layers): {e}",
                freed.bytes, freed.slots, freed.layers
            ),
        )
    })
}

/// The card settled to its margin once every buffer serving uses exists
/// (after the warm-up, and at a quiet residency boundary): with `free` card
/// bytes and a `margin` to keep, a card below the margin gives the shortfall
/// back (`site` names the settle) and a card above it fills the withdrawn
/// slots the room past the margin and the keep allows
/// (`placement::elastic::fill_slots`), the fill held for the boundaries
/// after a give-back (`fill_allowed`), on the body's own boundary clock
/// ([`ElasticStats::boundary`]). The move, or why none ran, goes to `report`
/// and the counters; the card counts as read ([`ElasticStats::settled`]).
///
/// # Errors
///
/// The machine's refusal of a give-back the margin needs (a fill that is
/// refused is reported, not an error: the card keeps what it has).
pub fn settle_to_margin<B: ElasticBody>(
    body: &mut B,
    site: &'static str,
    free: u64,
    margin: u64,
    mut report: impl FnMut(Elastic),
) -> Result<(), GpuError> {
    body.elastic_stats().settled();
    if free < margin {
        let want = margin - free;
        let freed = body.free_expert_bytes(want)?;
        body.elastic_stats().gave_back(freed.bytes);
        report(Elastic::GiveBack {
            site,
            asked: want,
            freed,
        });
        return Ok(());
    }
    let Some((layers, slot_bytes)) = body.expert_quantum() else {
        report(Elastic::Held {
            site,
            why: "no residency machine".to_owned(),
        });
        return Ok(());
    };
    let stats = *body.elastic_stats();
    if !fill_allowed(stats.boundary, stats.last_give_back) {
        report(Elastic::Held {
            site,
            why: "a give-back ran within the fill's hold".to_owned(),
        });
        return Ok(());
    }
    let slots = fill_slots(free, margin, layers, slot_bytes, usize::MAX);
    if slots == 0 {
        report(Elastic::Held {
            site,
            why: HELD_NO_QUANTUM.to_owned(),
        });
        return Ok(());
    }
    let room = (slots as u64).saturating_mul(layers.saturating_mul(slot_bytes));
    match body.fill_expert_bytes(room) {
        Ok(filled) => {
            if filled.slots > 0 {
                body.elastic_stats().filled(filled.bytes);
            }
            report(Elastic::Fill {
                site,
                boundary: stats.boundary,
                free,
                margin,
                filled,
            });
        }
        Err(e) => report(Elastic::Held {
            site,
            why: format!("the residency machine refused the fill: {e}"),
        }),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Body {
        calls: Vec<String>,
        refuse: usize,
        machine: bool,
        stats: ElasticStats,
        quantum: Option<(u64, u64)>,
        withdrawn: usize,
    }

    fn oom() -> GpuError {
        GpuError::NoRoom {
            what: "test rooms",
            detail: "a fit".to_owned(),
            source: Box::new(GpuError::state("test", "refused")),
            asked: 10,
            free: Some(3),
            page: 4,
            spill: false,
        }
    }

    impl KvRoomBody for Body {
        const WHAT: &'static str = "test body";

        fn fit_each(&mut self, _pos: usize, _through: usize) -> Result<(), GpuError> {
            Ok(())
        }

        fn asks(&self, _through: usize) -> usize {
            0
        }

        fn free_expert_bytes(&mut self, want: u64) -> Result<FreedBytes, GpuError> {
            self.calls.push(format!("free {want}"));
            if !self.machine {
                return Err(GpuError::state("test", "no machine"));
            }
            Ok(FreedBytes {
                bytes: want,
                slots: 1,
                layers: 3,
            })
        }
    }

    impl ElasticBody for Body {
        fn fill_expert_bytes(&mut self, room: u64) -> Result<FilledBytes, GpuError> {
            self.calls.push(format!("fill {room}"));
            Ok(FilledBytes {
                bytes: room,
                slots: 1,
                layers: 3,
            })
        }

        fn expert_quantum(&self) -> Option<(u64, u64)> {
            self.quantum
        }

        fn withdrawn_slots(&self) -> usize {
            self.withdrawn
        }

        fn elastic_stats(&mut self) -> &mut ElasticStats {
            &mut self.stats
        }

        fn take_moves(&mut self) -> Vec<Elastic> {
            Vec::new()
        }
    }

    fn body(refuse: usize, machine: bool) -> Body {
        Body {
            calls: Vec::new(),
            refuse,
            machine,
            stats: ElasticStats::default(),
            quantum: Some((3, 100)),
            withdrawn: 4,
        }
    }

    fn op(b: &mut Body) -> Result<(), GpuError> {
        b.calls.push("op".to_owned());
        if b.refuse > 0 {
            b.refuse -= 1;
            return Err(oom());
        }
        Ok(())
    }

    /// The retry order: op, the machine asked for the bytes, op once more; the give-back counted
    /// and reported with its site.
    ///
    /// FAIL-first: a loop that skips the second `op` ends with two calls and the first assertion
    /// goes red; one that forgets the counter leaves `give_backs` at 0.
    #[test]
    fn a_refused_op_asks_the_machine_once_and_runs_again() {
        let mut b = body(1, true);
        let mut seen = Vec::new();
        with_room(&mut b, "slot", 11, op, |e| seen.push(e)).expect("the second op holds");
        assert_eq!(b.calls, ["op", "free 11", "op"]);
        assert_eq!(b.stats.give_backs, 1);
        assert_eq!(b.stats.give_back_bytes, 11);
        assert!(matches!(
            seen.as_slice(),
            [Elastic::GiveBack {
                site: "slot",
                asked: 11,
                ..
            }]
        ));
    }

    /// An error that is not the card's memory is returned as it is, the machine not asked.
    #[test]
    fn another_error_is_not_retried() {
        let mut b = body(0, true);
        let e = with_room(
            &mut b,
            "slot",
            11,
            |_| Err::<(), _>(GpuError::state("test", "a shape")),
            |_| {},
        )
        .expect_err("the error stands");
        assert!(e.to_string().contains("a shape"), "{e}");
        assert!(b.calls.is_empty());
    }

    /// A second refusal and a refusing machine are the named refusal, each noting what the
    /// machine did and the site.
    #[test]
    fn a_second_refusal_and_a_refusing_machine_are_named() {
        let mut twice = body(2, true);
        let e = with_room(&mut twice, "batch", 11, op, |_| {}).expect_err("two refusals");
        assert!(
            e.to_string().contains("batch: 11 B would not fit after"),
            "{e}"
        );
        assert!(e.to_string().contains("freed 11 B"), "{e}");
        let mut none = body(1, false);
        let e = with_room(&mut none, "batch", 11, op, |_| {}).expect_err("no machine");
        assert!(e.to_string().contains("freed 0 B"), "{e}");
        assert_eq!(none.stats.give_backs, 0);
    }

    /// A settle below the margin gives the shortfall back and marks the boundary; one above it
    /// fills what the room past the margin and the keep allows, in whole quanta; one within the
    /// hold after a give-back holds.
    ///
    /// FAIL-first: a settle that fills below the margin shows `fill` in the first calls; one
    /// that ignores the keep fills 12 quanta, not 10.
    #[test]
    fn a_settle_gives_back_below_the_margin_and_fills_above_it() {
        let mut b = body(0, true);
        let mut seen = Vec::new();
        settle_to_margin(&mut b, "warm", 90, 100, |e| seen.push(e)).expect("the give-back");
        assert_eq!(b.calls, ["free 10"]);
        assert_eq!(b.stats.last_give_back, Some(0));
        let mut b = body(0, true);
        settle_to_margin(&mut b, "warm", 100 + 12 * 300, 100, |e| seen.push(e)).expect("the fill");
        assert_eq!(b.calls, [format!("fill {}", 10 * 300)]);
        assert_eq!(b.stats.fills, 1);
        let mut held = body(0, true);
        held.stats.boundary = 6;
        held.stats.last_give_back = Some(5);
        settle_to_margin(&mut held, "warm", 100 + 12 * 300, 100, |e| seen.push(e))
            .expect("the hold");
        assert!(held.calls.is_empty());
        assert!(matches!(seen.last(), Some(Elastic::Held { .. })));
    }

    /// A body with no machine settles to a held report, never an error.
    #[test]
    fn a_body_without_a_machine_holds() {
        let mut b = body(0, false);
        b.quantum = None;
        let mut seen = Vec::new();
        settle_to_margin(&mut b, "warm", 1_000, 100, |e| seen.push(e)).expect("held");
        assert!(matches!(
            seen.as_slice(),
            [Elastic::Held { why, .. }] if why == "no residency machine"
        ));
    }
}
