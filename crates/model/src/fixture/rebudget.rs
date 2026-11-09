//! The recorded card budget of a written fixture ([`KEY_CARD_BUDGET`]), rewritten in place to the
//! value the family's planner gives the file as it now stands: the one owner of that patch, for a
//! variant after its requant and for a trunk whose plan moved under it. The patch writes the eight
//! bytes of the u64 and no other byte of the set; a draft fixture beside the trunk records the same
//! value, and an r8 sidecar beside it records the digests of the headers the patch changed
//! (`r8file::record_headers`).

use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use gguf::{Split, Value};

use super::spec::{CardBudget, FixtureSpec};
use super::{FixtureError, KEY_CARD_BUDGET, KEY_SUBSET, KEY_VARIANT, io_err, sidecar};
use crate::r8file;

/// How [`rebudget`] came to the value it recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chosen {
    /// A family with a fixed budget: the recorded value stays as it is.
    Fixed,
    /// The planner's least cap for the written file.
    Planned,
    /// A host-only variant: no layer holds a card expert, so the header's value stays.
    HostOnly,
    /// The caller's value, as `generate`'s `--card-budget` gives one: no plan is run.
    Given,
}

/// What [`rebudget`] did.
#[derive(Clone, Debug)]
pub struct Rebudget {
    /// The value the shards held.
    pub was: u64,
    /// The value they hold now.
    pub now: u64,
    pub chosen: Chosen,
    /// Whether the file is a variant ([`KEY_VARIANT`]).
    pub variant: bool,
    /// The draft fixture beside the trunk that now records the same value, if the family has one
    /// and it exists.
    pub draft: Option<PathBuf>,
    /// The r8 sidecar beside the fixture and the count of source header digests it re-recorded,
    /// if the family has one and it exists.
    pub sidecar: Option<(PathBuf, usize)>,
}

fn refused(path: &Path, detail: impl std::fmt::Display) -> FixtureError {
    FixtureError::Budget(format!("{}: {detail}", path.display()))
}

/// The budget the family's planner gives the written `split`: a variant's by
/// [`super::budget::choose_written`] (it may be host-only), a trunk's by the generator's own search.
/// A file holding a subset of the planned tensors has no whole plan to choose from and is refused.
///
/// # Errors
/// A subset file, the planner's refusal, or a host-only plan of a trunk.
pub fn plan(budget: &CardBudget, split: &Split) -> Result<(u64, Chosen), FixtureError> {
    let planner = match budget {
        CardBudget::Fixed(_) => return Ok((recorded(split)?, Chosen::Fixed)),
        CardBudget::Planned(planner) => planner,
    };
    if split.value(KEY_SUBSET).is_some() {
        return Err(FixtureError::Budget(format!(
            "the file carries {KEY_SUBSET}: a subset holds no whole plan to choose the budget from"
        )));
    }
    let variant = split.value(KEY_VARIANT).is_some();
    let b = super::budget::choose_in(planner, split, variant)?;
    let chosen = if variant && super::budget::host_only(planner, split)? {
        Chosen::HostOnly
    } else {
        Chosen::Planned
    };
    Ok((b, chosen))
}

/// The value every shard of `split` holding [`KEY_CARD_BUDGET`] records, which they must agree on.
fn recorded(split: &Split) -> Result<u64, FixtureError> {
    let mut was = None;
    for i in 0..split.shard_count() {
        let (Some(shard), Some(path)) = (split.shard(i), split.shard_path(i)) else {
            continue;
        };
        match shard.value(KEY_CARD_BUDGET) {
            None => {}
            Some(Value::U64(v)) => match was {
                Some(w) if w != *v => {
                    return Err(refused(
                        path,
                        format!("holds {KEY_CARD_BUDGET} {v}, an earlier shard's {w}"),
                    ));
                }
                _ => was = Some(*v),
            },
            Some(other) => {
                return Err(refused(
                    path,
                    format!(
                        "holds {KEY_CARD_BUDGET} as {other:?}, not a u64 (the patch writes eight \
                         bytes)"
                    ),
                ));
            }
        }
    }
    was.ok_or_else(|| {
        FixtureError::Budget(format!(
            "no shard holds {KEY_CARD_BUDGET} to record the budget on"
        ))
    })
}

/// Record `budget` as [`KEY_CARD_BUDGET`] on every shard of `split` that holds the key, in place:
/// the value is a u64, so its eight bytes are written and no other byte of the file moves, and the
/// set is reopened to read the key back. Returns the value the shards held. Every shard is checked
/// before the first byte is written.
///
/// # Errors
/// No shard holding the key, a shard whose value is not a u64, shards that disagree on it, or a
/// file the patch cannot write or read back.
pub fn record(split: &Split, budget: u64) -> Result<u64, FixtureError> {
    let was = recorded(split)?;
    for i in 0..split.shard_count() {
        let (Some(shard), Some(path)) = (split.shard(i), split.shard_path(i)) else {
            continue;
        };
        let Some(at) = shard.value_offset(KEY_CARD_BUDGET) else {
            continue;
        };
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .map_err(|e| io_err(path, "open for the budget patch", e))?;
        file.seek(SeekFrom::Start(at))
            .and_then(|_| file.write_all(&budget.to_le_bytes()))
            .and_then(|()| file.sync_data())
            .map_err(|e| io_err(path, "write the budget", e))?;
    }
    // The patch moved eight bytes inside one value; the reopened set must read them back as the
    // budget it now holds.
    let first = split.shard_path(0).ok_or_else(|| {
        FixtureError::Budget(format!(
            "the set names no first shard to read {KEY_CARD_BUDGET} back from"
        ))
    })?;
    let again = Split::open(first)?;
    let now = recorded(&again)?;
    if now != budget {
        return Err(refused(
            first,
            format!("{KEY_CARD_BUDGET} reads back as {now}, want {budget}"),
        ));
    }
    Ok(was)
}

/// Rewrite the recorded budget of the fixture whose first shard is `first` to `given`, else to
/// [`plan`]'s value for it, then the draft fixture beside a trunk to the same value, then the r8 sidecar's record of the
/// source headers the patch moved. A sidecar the rewritten fixture would not open is restored and
/// the refusal returned. A family with a fixed budget keeps the recorded value, and its sidecar is
/// still re-recorded (a no-op when the headers did not move).
///
/// # Errors
/// [`plan`]'s and [`record`]'s, a draft fixture that is not a GGUF set, and the sidecar's refusal.
pub fn rebudget(
    spec: &FixtureSpec,
    first: &Path,
    given: Option<u64>,
) -> Result<Rebudget, FixtureError> {
    let split = Split::open(first)?;
    let variant = split.value(KEY_VARIANT).is_some();
    let (now, chosen) = match given {
        Some(b) => (b, Chosen::Given),
        None => plan(&spec.card_budget, &split)?,
    };
    let was = record(&split, now)?;
    drop(split);
    let dir = first.parent().unwrap_or(Path::new("."));
    let draft = match spec.draft {
        Some(ds) if !variant => {
            let p = dir.join(ds.file);
            if p.try_exists().map_err(|e| io_err(&p, "stat", e))? {
                record(&Split::open(&p)?, now)?;
                Some(p)
            } else {
                None
            }
        }
        _ => None,
    };
    let sidecar = match &spec.sidecar {
        Some(_) => {
            let path = sidecar::path_of(first)?;
            if path.try_exists().map_err(|e| io_err(&path, "stat", e))? {
                let fixture = Split::open(first)?;
                let n =
                    r8file::record_headers(&path, &fixture).map_err(|e| FixtureError::Sidecar {
                        path: path.clone(),
                        detail: e.to_string(),
                    })?;
                Some((path, n))
            } else {
                None
            }
        }
        None => None,
    };
    Ok(Rebudget {
        was,
        now,
        chosen,
        variant,
        draft,
        sidecar,
    })
}
