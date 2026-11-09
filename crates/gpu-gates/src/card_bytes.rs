//! The card-byte invariant as the binaries apply it: each reading
//! ([`bloomery_gpu::card_bytes`]) printed as its record line, and, where the
//! run holds the card, a reading past the plan's count or one that could not
//! be measured refused by name.
//!
//! Whether the run holds the card is `BLOOMERY_CARD_HELD`, which
//! `tools/gpu-gate.sh` exports once it holds the card's gate lock and the
//! release sitting exports for the server it times: `1` holds it ([`Hold::Held`],
//! a refusal), unset or empty does not ([`Hold::Unheld`], the line alone, as in
//! plain serving, where another process may share the card), and any other
//! value is refused by name. A session takes its watch from [`watch`]; a binary
//! that opens its model its own way reads its card through [`check_card`].

use std::ffi::OsStr;

use app::{CardWatch, SessionError};
use bloomery_gpu::card_bytes::{PlanCard, card_reading};
use bloomery_gpu::{Gpu, GpuError};
use model::placement::card_made::{CardPhase, CardReading};

use crate::GateError;
use crate::record::{self, CARD_BYTES, CARD_UNMEASURED, Record};

/// The variable that says whether the run holds its card.
pub const HELD: &str = "BLOOMERY_CARD_HELD";

/// Whether the run holds its card: where it does, the card-byte invariant
/// refuses; where it does not, it only records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hold {
    Held,
    Unheld,
}

impl Hold {
    /// The hold `value` of [`HELD`] names: `1` held, unset or empty unheld;
    /// any other value is refused by name.
    pub fn of(value: Option<&OsStr>) -> Result<Hold, String> {
        match value.map(OsStr::to_str) {
            None | Some(Some("")) => Ok(Hold::Unheld),
            Some(Some("1")) => Ok(Hold::Held),
            Some(v) => Err(format!(
                "{HELD}={}: `1` (the run holds its card) or unset",
                v.unwrap_or("<not UTF-8>")
            )),
        }
    }

    /// The hold this process's [`HELD`] names.
    pub fn from_env() -> Result<Hold, String> {
        Hold::of(std::env::var_os(HELD).as_deref())
    }

    /// The word the record lines give the hold: what a reading does there.
    #[must_use]
    pub fn word(self) -> &'static str {
        match self {
            Hold::Held => "refuse",
            Hold::Unheld => "record",
        }
    }
}

/// `reading`'s record line under `hold`.
#[must_use]
pub fn line(reading: &CardReading, hold: Hold) -> Record {
    match reading {
        CardReading::Made(m) => Record::new(&CARD_BYTES)
            .w("phase", m.phase)
            .u("card", m.card)
            .w("held", hold.word())
            .w(
                "verdict",
                if m.past().is_some() { "past" } else { "within" },
            )
            .u("made", m.made())
            .u("planned", m.planned())
            .u("past", m.past().unwrap_or(0))
            .u("before", m.before)
            .u("now", m.now)
            .u("in_reading", m.in_reading)
            .csv(
                "terms",
                m.terms.iter().map(|(name, b)| format!("{name}={b}")),
            ),
        CardReading::Unmeasured(u) => Record::new(&CARD_UNMEASURED)
            .w("phase", u.phase)
            .u("card", u.card)
            .w("held", hold.word())
            .w("why", &u.why),
    }
}

/// `reading` printed, then refused by name under [`Hold::Held`] when it is
/// past its plan or not measured.
pub fn apply(reading: &CardReading, hold: Hold) -> Result<(), GpuError> {
    line(reading, hold).eprint();
    match hold {
        Hold::Held => reading
            .check()
            .map_err(|e| GpuError::plan("refuse_card_past_plan", e)),
        Hold::Unheld => Ok(()),
    }
}

/// A session's watch over its card ([`app::CardWatch`]) under one hold.
pub struct Watch {
    hold: Hold,
}

impl CardWatch for Watch {
    fn card(&mut self, reading: &CardReading) -> Result<(), SessionError> {
        Ok(apply(reading, self.hold)?)
    }
}

/// The watch a session's open hands its card to, under this process's
/// [`HELD`]: what every open's log returns ([`app::OpenLog::card_watch`]). A
/// process that registered its records and lacks these lines panics by name
/// here, at the open, not at the first reading.
pub fn watch() -> Result<Box<dyn CardWatch>, SessionError> {
    record::require(record::groups::card_bytes::WATCH, "card_bytes::watch");
    let hold = Hold::from_env().map_err(SessionError::Refused)?;
    Ok(Box::new(Watch { hold }))
}

/// For a binary that opens its model its own way: card `on` of its plan, read
/// on `gpu` (the `Gpu` the model on that card opened) at `phase`, printed and
/// refused as a session's watch does.
pub fn check_card(gpu: &Gpu, on: &PlanCard, phase: CardPhase) -> Result<(), GateError> {
    record::require(record::groups::card_bytes::WATCH, "card_bytes::check_card");
    let hold = Hold::from_env()?;
    apply(&card_reading(gpu, on, phase)?, hold)?;
    Ok(())
}
