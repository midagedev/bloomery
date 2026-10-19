//! A session's persisted residency usage profile
//! ([`runtime::warmprof::WarmProfile`]): read at the load's end, the
//! residency reseeded to its rankings, so a load starts from the card set
//! earlier runs settled on instead of its placement's seed; written back with
//! this run's usage folded in. The load runs without it: a missing file is no
//! profile, and one written under another key or corrupt is refused by name.

use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use bloomery_gpu::host::swap::ResetReport;
use runtime::warmprof::{ProfileKey, WarmProfile};

use bloomery_gpu::model::ChainBody;

use crate::{Session, SessionError};

/// One profile file and the load it is keyed to ([`ProfileKey`]).
pub struct WarmFile {
    path: PathBuf,
    model: String,
    bytes: u64,
    mtime_ns: u128,
    place: String,
    cards: Vec<String>,
    /// The rule's usage counts the last [`WarmFile::save`] folded, so the
    /// next folds only what came after.
    folded: Vec<u64>,
}

/// What [`WarmFile::load`] did.
#[derive(Debug)]
pub enum WarmLoad {
    /// The model has no residency machine.
    NoResidency,
    /// No file at the path.
    Missing,
    /// The file is not a profile of this load; the reason names why.
    Refused(String),
    /// The residency reseeded to the profile's rankings.
    Reseeded(ResetReport),
}

/// What [`WarmFile::save`] did.
#[derive(Debug)]
pub enum WarmSave {
    /// The model has no residency machine.
    NoResidency,
    /// The profile written, `bytes` long; `replaced` names why a file at the
    /// path was not folded into (another key, corrupt).
    Written {
        bytes: usize,
        replaced: Option<String>,
    },
}

impl WarmFile {
    /// The profile at `path` for a load of `model` (its first shard) at
    /// placement `place` on `cards`. Refused by name: a model file whose
    /// length or modification time cannot be read.
    pub fn new(
        path: &Path,
        model: &Path,
        place: &str,
        cards: &[&str],
    ) -> Result<WarmFile, SessionError> {
        let refused = |e: std::io::Error| {
            SessionError::Refused(format!(
                "warm profile: the model file {} has no length or time: {e}",
                model.display()
            ))
        };
        let meta = std::fs::metadata(model).map_err(refused)?;
        let mtime_ns = meta
            .modified()
            .map_err(refused)?
            .duration_since(UNIX_EPOCH)
            .map_err(|e| refused(std::io::Error::other(e)))?
            .as_nanos();
        Ok(WarmFile {
            path: path.to_path_buf(),
            model: model.display().to_string(),
            bytes: meta.len(),
            mtime_ns,
            place: place.to_string(),
            cards: cards.iter().map(|c| (*c).to_string()).collect(),
            folded: Vec::new(),
        })
    }

    /// The profile's path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Reseed `s`'s residency to the profile's rankings, at the load's end
    /// before any request: every later residency reset returns there.
    /// Refused by name: a file that cannot be read for another reason than
    /// its absence, and what the reseed refuses.
    pub fn load<B: ChainBody>(&mut self, s: &mut Session<B>) -> Result<WarmLoad, SessionError> {
        let Some((layers, experts)) = shape(s) else {
            return Ok(WarmLoad::NoResidency);
        };
        let bytes = match std::fs::read(&self.path) {
            Ok(b) => b,
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(WarmLoad::Missing),
            Err(e) => {
                return Err(SessionError::Refused(format!(
                    "warm profile {}: {e}",
                    self.path.display()
                )));
            }
        };
        let profile = match self.decode(&bytes, layers, experts) {
            Ok(p) => p,
            Err(why) => return Ok(WarmLoad::Refused(why)),
        };
        let orders: Vec<Vec<u32>> = (0..layers as usize)
            .map(|l| {
                profile
                    .ranking(l)
                    .expect("a decoded profile has the rule's layers")
            })
            .collect();
        let order: Vec<&[u32]> = orders.iter().map(Vec::as_slice).collect();
        match s.residency_reseed(&order)? {
            Some(r) => Ok(WarmLoad::Reseeded(r)),
            None => Ok(WarmLoad::NoResidency),
        }
    }

    /// Fold `s`'s usage since the last save into the file's profile and
    /// write it whole (a sibling file renamed over it). A file that is not
    /// this load's profile is replaced by this run's usage alone, the
    /// report naming why. Refused by name: a read or write that fails.
    pub fn save<B: ChainBody>(&mut self, s: &mut Session<B>) -> Result<WarmSave, SessionError> {
        let Some((layers, experts)) = shape(s) else {
            return Ok(WarmSave::NoResidency);
        };
        let usage = s
            .residency_rule()
            .expect("shape found the rule")
            .usage()
            .to_vec();
        let io = |e: std::io::Error| {
            SessionError::Refused(format!("warm profile {}: {e}", self.path.display()))
        };
        let (mut profile, replaced) = match std::fs::read(&self.path) {
            Ok(b) => match self.decode(&b, layers, experts) {
                Ok(p) => (p, None),
                Err(why) => (self.empty(layers, experts), Some(why)),
            },
            Err(e) if e.kind() == ErrorKind::NotFound => (self.empty(layers, experts), None),
            Err(e) => return Err(io(e)),
        };
        let delta: Vec<u64> = match self.folded.len() {
            0 => usage.clone(),
            _ => usage
                .iter()
                .zip(&self.folded)
                .map(|(&u, &f)| u.checked_sub(f).expect("a rule's usage only grows"))
                .collect(),
        };
        profile
            .fold(&delta)
            .map_err(|e| SessionError::Refused(format!("warm profile: {e}")))?;
        let mut out = Vec::new();
        profile.encode(&mut out);
        let tmp = self.path.with_extension("tmp");
        std::fs::write(&tmp, &out).map_err(io)?;
        std::fs::rename(&tmp, &self.path).map_err(io)?;
        self.folded = usage;
        Ok(WarmSave::Written {
            bytes: out.len(),
            replaced,
        })
    }

    fn decode(&self, bytes: &[u8], layers: u32, experts: u32) -> Result<WarmProfile, String> {
        let cards: Vec<&str> = self.cards.iter().map(String::as_str).collect();
        let key = self.key(&cards);
        WarmProfile::decode(bytes, &key, layers, experts).map_err(|e| e.to_string())
    }

    fn empty(&self, layers: u32, experts: u32) -> WarmProfile {
        let cards: Vec<&str> = self.cards.iter().map(String::as_str).collect();
        WarmProfile::empty(&self.key(&cards), layers, experts)
    }

    fn key<'a>(&'a self, cards: &'a [&'a str]) -> ProfileKey<'a> {
        ProfileKey {
            model: &self.model,
            bytes: self.bytes,
            mtime_ns: self.mtime_ns,
            place: &self.place,
            cards,
        }
    }
}

/// The residency rule's `(layers, experts)`; `None` without a machine.
fn shape<B: ChainBody>(s: &mut Session<B>) -> Option<(u32, u32)> {
    let rule = s.residency_rule()?;
    let layers = rule.layers();
    let experts = rule.usage().len() / layers;
    Some((
        u32::try_from(layers).expect("a rule's layers fit u32"),
        u32::try_from(experts).expect("a rule's experts fit u32"),
    ))
}
