//! The draft a V4.1 binary serves: which one (`BLOOMERY_DRAFT`), and the
//! DSpark draft's open on the loaded target, before its capture, with its
//! `load draft=dspark` record. `generate_ds41` and `bloomery-serve-ds41` open
//! it here; its file and card are `ds41_dspark.rs`'s.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use app::arch::deepseek41::{Ds41Draft, open_draft};
use app::{Loaded, Session};
use bloomery_gpu_deepseek41::body::Body;
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::generate::{DraftSeat, Place};
use bloomery_gpu_gates::record::{self, Record};
use bloomery_levers::{DraftWord, Levers};
use gguf::Split;
use model::arch::dspark::{DraftCardBytes, DraftHeld, DraftHparams};
use model::placement::workstation::{CardSpec, GRANULE};

use crate::dspark;

/// Which draft `BLOOMERY_DRAFT` serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Draft {
    /// Unset: the plain path.
    Off,
    /// The n-gram lookup.
    Lookup,
    /// The DSpark draft (`shared/ds41_dspark.rs`).
    Dspark,
}

impl Draft {
    /// `BLOOMERY_DRAFT` as the levers hold it: unset is the plain path,
    /// `lookup` and `dspark` the served drafts.
    pub fn from_levers(levers: &Levers) -> Result<Draft, GateError> {
        record::require(
            record::groups::ds41_draft::LOAD,
            "ds41_draft::Draft::from_levers",
        );
        Self::check_levers(levers)
    }

    /// The draft word without record registration or model access.
    pub fn check_levers(levers: &Levers) -> Result<Draft, GateError> {
        const WORDS: [DraftWord; 2] = [DraftWord::Lookup, DraftWord::Dspark];
        match levers.draft_word(&WORDS) {
            Ok(None) => Ok(Draft::Off),
            Ok(Some(DraftWord::Lookup)) => Ok(Draft::Lookup),
            Ok(Some(_)) => Ok(Draft::Dspark),
            Err(e) => Err(format!(
                "BLOOMERY_DRAFT is lookup, dspark or unset, not {:?}",
                e.word.as_str()
            )
            .into()),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Draft::Off => "unset",
            Draft::Lookup => "lookup",
            Draft::Dspark => "dspark",
        }
    }
}

/// The DSpark draft of `file` (read by [`dspark::draft_hparams`]) on its
/// card under `place` ([`dspark::draft_card`]), its taps attached to `loaded`
/// — the target opened from `target`, nothing captured yet — and its `load
/// draft=dspark` record for the caller to print; `what` names the caller to
/// the body. The load's own buffer count is held to the plan's formula term
/// by term (`DraftCardBytes::check_held`) whatever the placement. `seat`
/// is what the placement's plan set aside for the draft on its card
/// ([`dspark::draft_seat`]): the load opens on that card, else it is
/// refused by name before any allocation, and a draft whose resident bytes
/// are more is refused by name: it would sit in the card's margin. The
/// card's loss around the load is recorded as `taken` and never refuses:
/// another process's allocation on the card lands in it.
pub fn open_dspark(
    loaded: &mut Loaded<Body>,
    file: &(Split, DraftHparams),
    target: &Path,
    what: &'static str,
    place: Place,
    seat: DraftSeat,
) -> Result<(Ds41Draft, Record), GateError> {
    let t = Instant::now();
    let on = dspark::draft_card(place)?;
    place.check_draft_card(seat, &on)?;
    let (reserve, card) = (seat.reserve, on.spec);
    let (draft_split, hp) = file;
    let target =
        Arc::new(Split::open(target).map_err(|e| format!("open {}: {e}", target.display()))?);
    let mut d = open_draft(
        loaded,
        draft_split,
        hp,
        Arc::clone(&target),
        (card.name, card.device),
    )?;
    let resident = d.resident_bytes() as u64;
    hold_load(
        &dspark::plan_bytes(draft_split, &target)?,
        d.held(),
        resident,
        seat,
        (place, &card),
    )?;
    let (free, total) = d.mem_info()?;
    let b = loaded.model().body(what)?;
    let r = Record::new(&record::LOAD_DRAFT)
        .w("draft", "dspark")
        .w("card", d.card())
        .u("width", <Ds41Draft as runtime::Draft<Session<Body>>>::WIDTH)
        .list("target_layers", b.feature_layers().unwrap_or_default())
        .u("feature_width", b.feature_width())
        .u("resident", resident)
        .u("working", d.held().working)
        .u("reserve", reserve.bytes)
        .u("taken", d.taken_bytes() as u64)
        .u("draft_card_free", free)
        .u("draft_card_total", total)
        .f("load_s", t.elapsed().as_secs_f64());
    Ok((d, r))
}

/// A draft's load held to its plan, whichever family's draft it is: the
/// buffers the load counted of itself against the formula `planned` term by
/// term ([`DraftCardBytes::check_held`]), and its `resident` bytes against
/// what `seat`'s plan reserved for it on `card` under `place`. A draft whose
/// resident bytes are more would sit in the card's margin: refused by name.
pub fn hold_load(
    planned: &DraftCardBytes,
    held: &DraftHeld,
    resident: u64,
    seat: DraftSeat,
    (place, card): (Place, &CardSpec),
) -> Result<(), GateError> {
    planned.check_held(held, GRANULE, card.name)?;
    if resident > seat.reserve.bytes {
        return Err(format!(
            "the DSpark draft holds {resident} B on {}, past the {} B its plan reserved there \
             under --place {}",
            card.name,
            seat.reserve.bytes,
            place.name()
        )
        .into());
    }
    Ok(())
}
