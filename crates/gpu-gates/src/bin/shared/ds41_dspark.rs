//! Where the DSpark draft a V4.1 run serves comes from: its file
//! (`BLOOMERY_DSPARK_MODEL`) and its card (`BLOOMERY_DSPARK_CARD`). The draft
//! itself is the session's (`app::arch::deepseek41::CardDraft`).
//!
//! The draft's card is a placement card name, the 3090 when unset — plan
//! (a)'s idle card. It may be the target's own card (the gate runs both on
//! the 3090 under a card budget).

use std::path::PathBuf;

use bloomery_gpu_gates::GateError;
use gguf::Split;
use model::arch::dspark::DraftHparams;
use model::placement::workstation;

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

/// `BLOOMERY_DSPARK_CARD`: a placement card name; unset is the 3090.
pub fn draft_card() -> Result<&'static str, GateError> {
    let cards = [workstation::RTX_3090.name, workstation::A6000.name];
    match std::env::var("BLOOMERY_DSPARK_CARD") {
        Err(std::env::VarError::NotPresent) => Ok(workstation::RTX_3090.name),
        Ok(v) => cards
            .into_iter()
            .find(|c| *c == v)
            .ok_or_else(|| format!("BLOOMERY_DSPARK_CARD is one of {cards:?}, not {v:?}").into()),
        Err(e) => Err(format!("BLOOMERY_DSPARK_CARD: {e}").into()),
    }
}
