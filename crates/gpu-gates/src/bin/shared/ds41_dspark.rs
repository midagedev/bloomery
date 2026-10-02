//! Where the DSpark draft a V4.1 run serves comes from: its file
//! (`BLOOMERY_DSPARK_MODEL`) and its card (`BLOOMERY_DSPARK_CARD`). The draft
//! itself is the session's (`app::arch::deepseek41::CardDraft`).
//!
//! The draft's card is one card as a `--place` list word names it, found
//! among this process's visible devices (`Place::draft_spec`): unset, the
//! device of the fewest usable bytes — the 3090, plan (a)'s idle card. It
//! may be the target's own card (the gate runs both on the 3090 under a card
//! budget). Under a placement with an expert tier card (`--place bp`) the
//! draft sits on the tier card, whose plan reserves its bytes
//! ([`draft_reserve`]): the lever may only name that device.

use std::path::PathBuf;

use std::path::Path;

use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::generate::Place;
use gguf::Split;
use model::arch::dspark::{self, DraftHparams};
use model::placement::workstation::{self, CardSpec};

/// The draft file: `$BLOOMERY_DSPARK_MODEL`, which the recipes export from
/// the V4.1 profile's `DSPARK_MODEL` (`tools/ref/models/deepseek41.sh`).
pub fn draft_path() -> Result<PathBuf, GateError> {
    match std::env::var_os("BLOOMERY_DSPARK_MODEL") {
        Some(p) if !p.is_empty() => Ok(PathBuf::from(p)),
        _ => Err(
            "BLOOMERY_DSPARK_MODEL unset — export the V4.1 profile's DSPARK_MODEL \
                  (`. tools/ref/ref-paths.sh` under BLOOMERY_MODEL=deepseek41), as the dspark \
                  recipes do"
                .into(),
        ),
    }
}

/// The draft file's hyperparameters, read from its header alone.
pub fn draft_hparams() -> Result<(Split, DraftHparams), GateError> {
    let path = draft_path()?;
    let draft = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let hp = DraftHparams::read(&draft)?;
    Ok((draft, hp))
}

/// The draft's card under `place` on this process's devices
/// ([`Place::draft_spec`] of `BLOOMERY_DSPARK_CARD` and the census): the
/// tier card when the placement has one, which the lever may name and no
/// other device; otherwise the lever's card, unset the device of the fewest
/// usable bytes (the 3090 here). The spec names its device.
pub fn draft_card(place: Place) -> Result<CardSpec, GateError> {
    let set = match std::env::var("BLOOMERY_DSPARK_CARD") {
        Err(std::env::VarError::NotPresent) => None,
        Ok(v) => Some(v),
        Err(e) => return Err(format!("BLOOMERY_DSPARK_CARD: {e}").into()),
    };
    place.draft_spec(set.as_deref(), &bloomery_gpu::census()?)
}

/// The reserve `place`'s plan makes for the DSpark draft `draft` on its
/// tier card: the draft's resident bytes from the two files' headers
/// (`model::arch::dspark::card_bytes`, the target's head and mask row read
/// from `target`), before any load; `None` under a placement with no tier
/// card.
pub fn draft_reserve(place: Place, draft: &Split, target: &Path) -> Result<Option<u64>, GateError> {
    if place.draft_card().is_none() {
        return Ok(None);
    }
    let t = Split::open(target).map_err(|e| format!("open {}: {e}", target.display()))?;
    let bytes = dspark::card_bytes(draft, &t, workstation::GRANULE)
        .map_err(|e| format!("the DSpark draft's card bytes: {e}"))?;
    Ok(Some(bytes.total()))
}
