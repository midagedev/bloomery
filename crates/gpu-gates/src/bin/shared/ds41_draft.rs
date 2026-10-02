//! The draft a V4.1 binary serves: which one (`BLOOMERY_DRAFT`), and the
//! DSpark draft's open on the loaded target, before its capture, with its
//! `load draft=dspark` record. `generate_ds41` and `bloomery-serve-ds41` open
//! it here; its file and card are `ds41_dspark.rs`'s.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use app::arch::deepseek41::CardDraft;
use app::{Loaded, Session};
use bloomery_gpu_deepseek41::body::Body;
use bloomery_gpu_deepseek41::draft::DraftBody;
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::generate::Place;
use bloomery_gpu_gates::record::{self, Record};
use bloomery_levers::Levers;
use gguf::Split;
use model::arch::dspark::DraftHparams;

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
        match levers.draft() {
            None => Ok(Draft::Off),
            Some("lookup") => Ok(Draft::Lookup),
            Some("dspark") => Ok(Draft::Dspark),
            Some(other) => {
                Err(format!("BLOOMERY_DRAFT is lookup, dspark or unset, not {other:?}").into())
            }
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
/// the body. With `reserve`, the bytes the placement's plan set aside for the
/// draft on its card ([`dspark::draft_reserve`]), a draft that took more is
/// refused by name: it would sit in the tier's margin.
pub fn open_dspark(
    loaded: &mut Loaded<Body>,
    file: &(Split, DraftHparams),
    target: &Path,
    what: &'static str,
    place: Place,
    reserve: Option<u64>,
) -> Result<(CardDraft<DraftBody>, Record), GateError> {
    let t = Instant::now();
    let card = dspark::draft_card(place)?;
    let (draft_split, hp) = file;
    let target =
        Arc::new(Split::open(target).map_err(|e| format!("open {}: {e}", target.display()))?);
    let mut d = CardDraft::open(loaded, draft_split, hp, target, (card.name, card.device))?;
    let (free, total) = d.mem_info()?;
    let resident = d.resident_bytes() as u64;
    if let Some(reserve) = reserve.filter(|&r| resident > r) {
        return Err(format!(
            "the DSpark draft took {resident} B on {}, past the {reserve} B its plan reserved \
             there under --place {}",
            card.name,
            place.name()
        )
        .into());
    }
    let b = loaded.model().body(what)?;
    let mut r = Record::new(&record::LOAD_DRAFT)
        .w("draft", "dspark")
        .w("card", d.card())
        .u(
            "width",
            <CardDraft<DraftBody> as runtime::Draft<Session<Body>>>::WIDTH,
        )
        .list("target_layers", b.feature_layers().unwrap_or_default())
        .u("feature_width", b.feature_width())
        .u("resident", resident);
    if let Some(reserve) = reserve {
        r = r.u("reserve", reserve);
    }
    let r = r
        .u("draft_card_free", free)
        .u("draft_card_total", total)
        .f("load_s", t.elapsed().as_secs_f64());
    Ok((d, r))
}
