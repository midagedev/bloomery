//! The residency profile: per (layer, expert) of one model file, the count
//! of the engine's own router picks over the kept rows of every pass a
//! residency machine ended, undecayed. One file per model file
//! (`BLOOMERY_RES_PROFILE`, [`ProfileLever`]), which every seat and process
//! of that file reads at its plan and adds to while it runs. This module is
//! the format's one owner: the file's bytes ([`Profile::encode`],
//! [`Profile::decode`]), its read with the refusals, its atomic write and
//! the merge a running machine adds through ([`add_to_file`]), and the seed
//! a plan takes from it ([`seed`]).
//!
//! **The seed.** A plan that puts a layer's routed experts on cards takes,
//! per layer, the profile's experts by [`runtime::xsplit::rank_key`] (count
//! descending, id ascending): the stage card its first `n_l`, each tier card
//! the next ones, in the order the planner filled them, the host the rest.
//! An expert the file never counted ranks after every counted one, by id, so
//! a layer the file ranks fewer than its card holds keeps the id prefix for
//! the rest. Each card segment keeps its count, so every byte total of the
//! plan stands; only which ids sit where moves ([`super::routed_row_on`],
//! the one owner of a stack's split, rebuilds each row).
//!
//! **The file.** Little-endian: the magic [`MAGIC`], the version, the model
//! layers, the experts a layer, the key's byte length, the key text
//! ([`ModelKey::text`]: the model key, then each shard's name and bytes), the
//! counts layer by layer, and an FNV-1a 64 of every byte before it. A file
//! short of its header or its shape, of another magic or version, of another
//! shape or model file, or whose sum does not hold is refused by name, and
//! the plan keeps the id prefix ([`CardSeed::Prefix`] says why).

use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use gguf::Split;
use runtime::xsplit::rank_key;

use super::{Device, ExpertList, Format, PlacementError, Plan, Role, routed_row_on_as};

/// The first bytes of every profile file.
pub const MAGIC: [u8; 8] = *b"BLMRPRF1";

/// The format version this build reads and writes.
pub const VERSION: u32 = 1;

/// The profiles' directory under the user's cache directory when
/// `BLOOMERY_RES_PROFILE` is unset: `$XDG_CACHE_HOME` when it holds an
/// absolute path, else `$HOME/.cache` (the XDG base directory rule).
pub const CACHE_DIR: &str = "bloomery/profiles";

/// The passes a running machine counts between two additions to the file.
pub const FLUSH_PASSES: u64 = 2048;

/// Bytes before the key text: magic, version, layers, experts, key length.
const HEAD: usize = 8 + 4 * 4;

/// Why a profile was not read, written or keyed.
#[derive(Debug, thiserror::Error)]
pub enum ProfileError {
    #[error("{}: {op}: {source}", .path.display())]
    Io {
        path: PathBuf,
        op: &'static str,
        source: io::Error,
    },
    #[error("{}: {len} bytes, short of the {want} its {what} takes", .path.display())]
    Short {
        path: PathBuf,
        len: usize,
        want: usize,
        what: &'static str,
    },
    #[error("{}: {len} bytes, past the {want} its header's shape takes", .path.display())]
    Long {
        path: PathBuf,
        len: usize,
        want: usize,
    },
    #[error("{}: magic {found:?}, not a residency profile's {MAGIC:?}", .path.display())]
    Magic { path: PathBuf, found: Vec<u8> },
    #[error("{}: version {found}; this build reads version {VERSION}", .path.display())]
    Version { path: PathBuf, found: u32 },
    #[error(
        "{}: {layers} layers of {experts} experts; the model has {want_layers} of {want_experts}",
        .path.display()
    )]
    Shape {
        path: PathBuf,
        layers: u64,
        experts: u64,
        want_layers: u64,
        want_experts: u64,
    },
    #[error("{}: written for the model file {found:?}, not this one, {want:?}", .path.display())]
    Model {
        path: PathBuf,
        found: String,
        want: String,
    },
    #[error("{}: its key is not UTF-8 text", .path.display())]
    KeyText { path: PathBuf },
    #[error("{}: sum {found:016x}, its bytes sum to {want:016x}", .path.display())]
    Sum {
        path: PathBuf,
        found: u64,
        want: u64,
    },
    #[error("the model file's key: {0}")]
    Key(String),
}

impl ProfileError {
    /// The file is not there: the first run of a model file.
    #[must_use]
    pub fn not_found(&self) -> bool {
        matches!(self, ProfileError::Io { source, .. } if source.kind() == io::ErrorKind::NotFound)
    }
}

/// The model file a profile belongs to: its key ([`crate::r8::model_stem`]
/// of the first shard's name) and each shard's name and bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelKey {
    pub stem: String,
    pub shards: Vec<(String, u64)>,
}

impl ModelKey {
    /// The key of `split`, each shard's length read from its file. A shard
    /// with no file name, or whose length cannot be read, is refused by name.
    pub fn of(split: &Split) -> Result<ModelKey, ProfileError> {
        let mut shards = Vec::with_capacity(split.shard_count());
        for i in 0..split.shard_count() {
            let path = split
                .shard_path(i)
                .ok_or_else(|| ProfileError::Key(format!("shard {i} has no path")))?;
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .ok_or_else(|| {
                    ProfileError::Key(format!("{}: the path names no file", path.display()))
                })?;
            let bytes = std::fs::metadata(path)
                .map_err(|source| ProfileError::Io {
                    path: path.to_path_buf(),
                    op: "read the shard's length",
                    source,
                })?
                .len();
            shards.push((name, bytes));
        }
        let first = shards
            .first()
            .map(|(n, _)| n.clone())
            .ok_or_else(|| ProfileError::Key("a split of no shard".to_string()))?;
        Ok(ModelKey {
            stem: crate::r8::model_stem(&first).to_string(),
            shards,
        })
    }

    /// The key as the file holds it: the stem, then a line a shard of its
    /// name and bytes.
    #[must_use]
    pub fn text(&self) -> String {
        let mut out = self.stem.clone();
        for (name, bytes) in &self.shards {
            out.push('\n');
            out.push_str(name);
            out.push(' ');
            out.push_str(&bytes.to_string());
        }
        out
    }
}

/// What `BLOOMERY_RES_PROFILE` asks a plan for, resolved once at a binary's
/// lever parse ([`super::PlanLevers::from_levers`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ProfileLever {
    /// No profile read or written: `off`, and a plan made without the
    /// binary's levers (`PlanLevers::default()`, a gate's own plan).
    #[default]
    Off,
    /// The file at this path.
    File(PathBuf),
    /// Unset: the model's file, its key the name, in this directory.
    Dir(PathBuf),
    /// Unset with neither an absolute `XDG_CACHE_HOME` nor `HOME`: no
    /// directory to look in.
    NoCache,
    /// The binary does not act on `BLOOMERY_RES_PROFILE` (a gate, a tool):
    /// no profile read or written, set or not.
    NotActed,
}

impl ProfileLever {
    /// The lever from whether the binary `acts` on it, its parsed value
    /// `set`, and the `HOME` and `XDG_CACHE_HOME` the binary runs with
    /// ([`CACHE_DIR`]; a relative `XDG_CACHE_HOME` is not one, by the XDG
    /// rule).
    #[must_use]
    pub fn of(
        acts: bool,
        set: Option<bloomery_levers::ResProfile<'_>>,
        home: Option<OsString>,
        xdg_cache: Option<OsString>,
    ) -> ProfileLever {
        if !acts {
            return ProfileLever::NotActed;
        }
        match set {
            Some(bloomery_levers::ResProfile::Off) => ProfileLever::Off,
            Some(bloomery_levers::ResProfile::File(p)) => ProfileLever::File(p.to_path_buf()),
            None => match (xdg_cache, home) {
                (Some(c), _) if Path::new(&c).is_absolute() => {
                    ProfileLever::Dir(PathBuf::from(c).join(CACHE_DIR))
                }
                (_, Some(h)) if !h.is_empty() => {
                    ProfileLever::Dir(PathBuf::from(h).join(".cache").join(CACHE_DIR))
                }
                _ => ProfileLever::NoCache,
            },
        }
    }
}

/// The file a running machine adds its counts to ([`add_to_file`]), and the
/// model it counts for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileTarget {
    pub path: PathBuf,
    pub key: ModelKey,
    /// The model's layers and experts a layer: the file's shape.
    pub layers: usize,
    pub experts: usize,
}

/// Which experts a plan's card segments took.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CardSeed {
    /// Each layer's id prefix, and why no profile ranks them; `path` where
    /// the plan looked, when it did.
    Prefix { path: Option<PathBuf>, why: String },
    /// Each layer's top experts of the profile at `path`, which counts
    /// `pairs` (layer, expert) pairs.
    Profile { path: PathBuf, pairs: usize },
}

/// A plan's seed: what its card segments took, and the file a machine that
/// runs it adds its counts to (`None`: it writes no profile).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Seed {
    pub card: CardSeed,
    pub target: Option<ProfileTarget>,
}

impl Default for Seed {
    /// A plan no profile was asked of: the planner's own id prefix.
    fn default() -> Seed {
        Seed {
            card: CardSeed::Prefix {
                path: None,
                why: "the plan read no profile".to_string(),
            },
            target: None,
        }
    }
}

/// One model file's counts: `layers × experts`, layer by layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Profile {
    pub key: ModelKey,
    pub layers: usize,
    pub experts: usize,
    pub counts: Vec<u32>,
}

impl Profile {
    /// No pick counted yet.
    #[must_use]
    pub fn new(key: ModelKey, layers: usize, experts: usize) -> Profile {
        Profile {
            key,
            layers,
            experts,
            counts: vec![0; layers * experts],
        }
    }

    /// Layer `layer`'s counts, an expert each.
    #[must_use]
    pub fn layer(&self, layer: usize) -> &[u32] {
        &self.counts[layer * self.experts..(layer + 1) * self.experts]
    }

    /// The (layer, expert) pairs the file counts.
    #[must_use]
    pub fn pairs(&self) -> usize {
        self.counts.iter().filter(|&&c| c != 0).count()
    }

    /// Every expert of layer `layer`, by [`rank_key`] of its count: the
    /// order the seed fills the cards in.
    #[must_use]
    pub fn order(&self, layer: usize) -> Vec<u32> {
        let counts = self.layer(layer);
        let mut ids: Vec<u32> = (0..counts.len() as u32).collect();
        ids.sort_unstable_by_key(|&id| rank_key(counts[id as usize], id));
        ids
    }

    /// `other`'s counts added to these, each saturating at `u32::MAX`. The
    /// shapes must agree.
    fn add(&mut self, other: &[u32]) {
        for (c, &o) in self.counts.iter_mut().zip(other) {
            *c = c.saturating_add(o);
        }
    }

    /// The file's bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let key = self.key.text();
        let mut out = Vec::with_capacity(HEAD + key.len() + 4 * self.counts.len() + 8);
        out.extend_from_slice(&MAGIC);
        for v in [
            VERSION,
            u32::try_from(self.layers).unwrap_or(u32::MAX),
            u32::try_from(self.experts).unwrap_or(u32::MAX),
            u32::try_from(key.len()).unwrap_or(u32::MAX),
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(key.as_bytes());
        for c in &self.counts {
            out.extend_from_slice(&c.to_le_bytes());
        }
        let sum = fnv1a64(&out);
        out.extend_from_slice(&sum.to_le_bytes());
        out
    }

    /// The profile `bytes` hold, read from `path` (named in each refusal):
    /// any shape, any key — [`read`] holds them to a model's.
    pub fn decode(bytes: &[u8], path: &Path) -> Result<Profile, ProfileError> {
        let short = |want: usize, what: &'static str| ProfileError::Short {
            path: path.to_path_buf(),
            len: bytes.len(),
            want,
            what,
        };
        if bytes.len() < HEAD {
            return Err(short(HEAD, "header"));
        }
        if bytes[..8] != MAGIC {
            return Err(ProfileError::Magic {
                path: path.to_path_buf(),
                found: bytes[..8].to_vec(),
            });
        }
        let word = |at: usize| {
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        let version = word(8);
        if version != VERSION {
            return Err(ProfileError::Version {
                path: path.to_path_buf(),
                found: version,
            });
        }
        let (layers, experts, key_len) = (word(12) as usize, word(16) as usize, word(20) as usize);
        let want = layers
            .checked_mul(experts)
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_add(HEAD + key_len + 8))
            .ok_or_else(|| short(usize::MAX, "shape"))?;
        if bytes.len() < want {
            return Err(short(want, "shape"));
        }
        if bytes.len() > want {
            return Err(ProfileError::Long {
                path: path.to_path_buf(),
                len: bytes.len(),
                want,
            });
        }
        let body = &bytes[..want - 8];
        let mut tail = [0u8; 8];
        tail.copy_from_slice(&bytes[want - 8..]);
        let (found, sum) = (u64::from_le_bytes(tail), fnv1a64(body));
        if found != sum {
            return Err(ProfileError::Sum {
                path: path.to_path_buf(),
                found,
                want: sum,
            });
        }
        let text = std::str::from_utf8(&bytes[HEAD..HEAD + key_len]).map_err(|_| {
            ProfileError::KeyText {
                path: path.to_path_buf(),
            }
        })?;
        let key = parse_key(text).ok_or_else(|| ProfileError::KeyText {
            path: path.to_path_buf(),
        })?;
        let counts = bytes[HEAD + key_len..want - 8]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| u32::from_le_bytes(c))
            .collect();
        Ok(Profile {
            key,
            layers,
            experts,
            counts,
        })
    }
}

/// [`ModelKey::text`] back; `None` for a line that is not a name and bytes.
fn parse_key(text: &str) -> Option<ModelKey> {
    let mut lines = text.split('\n');
    let stem = lines.next()?.to_string();
    let shards = lines
        .map(|l| {
            let (name, bytes) = l.rsplit_once(' ')?;
            Some((name.to_string(), bytes.parse().ok()?))
        })
        .collect::<Option<Vec<_>>>()?;
    Some(ModelKey { stem, shards })
}

/// FNV-1a 64 of `bytes`.
fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// The profile at `path` for the model `key` of `layers × experts`; a file
/// that is not there, short, malformed, of another shape or of another model
/// file is refused by name ([`ProfileError::not_found`] tells the first).
pub fn read(
    path: &Path,
    key: &ModelKey,
    layers: usize,
    experts: usize,
) -> Result<Profile, ProfileError> {
    let bytes = std::fs::read(path).map_err(|source| ProfileError::Io {
        path: path.to_path_buf(),
        op: "read",
        source,
    })?;
    let p = Profile::decode(&bytes, path)?;
    if (p.layers, p.experts) != (layers, experts) {
        return Err(ProfileError::Shape {
            path: path.to_path_buf(),
            layers: p.layers as u64,
            experts: p.experts as u64,
            want_layers: layers as u64,
            want_experts: experts as u64,
        });
    }
    if p.key != *key {
        return Err(ProfileError::Model {
            path: path.to_path_buf(),
            found: p.key.text(),
            want: key.text(),
        });
    }
    Ok(p)
}

/// Write `p` at `path`: its bytes into a file of this write's own beside it
/// (the process and a count: two machines of one process may add to one
/// file at once), then renamed over `path`, so a reader sees the old file or
/// the new one whole. The directory is made when missing.
pub fn write(path: &Path, p: &Profile) -> Result<(), ProfileError> {
    let io = |op: &'static str| {
        move |source| ProfileError::Io {
            path: path.to_path_buf(),
            op,
            source,
        }
    };
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(io("make the directory"))?;
    }
    static WRITES: AtomicU64 = AtomicU64::new(0);
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(format!(
        ".tmp.{}.{}",
        std::process::id(),
        WRITES.fetch_add(1, Ordering::Relaxed)
    ));
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, p.encode()).map_err(io("write the temporary file"))?;
    std::fs::rename(&tmp, path).map_err(|source| {
        let _ = std::fs::remove_file(&tmp);
        ProfileError::Io {
            path: path.to_path_buf(),
            op: "rename the temporary file over it",
            source,
        }
    })
}

/// Add `delta`'s counts to the file at `target`: the file's counts plus
/// them when it holds this model's profile, else `delta` alone — a file not
/// there yet, or one [`read`] refuses (it was refused at the plan already),
/// is replaced. Another process's addition between this read and the rename
/// is lost; the file stays whole. The profile written.
pub fn add_to_file(target: &ProfileTarget, delta: &[u32]) -> Result<Profile, ProfileError> {
    let mut p = read(&target.path, &target.key, target.layers, target.experts)
        .unwrap_or_else(|_| Profile::new(target.key.clone(), target.layers, target.experts));
    p.add(delta);
    write(&target.path, &p)?;
    Ok(p)
}

/// Seed `plan`'s card segments from the profile `lever` names for the model
/// file `key`, and record which in [`Plan::seed`]: every routed stack a card
/// holds experts of takes its layer's [`Profile::order`], the cards in the
/// order the planner filled them, each keeping its count. The id prefix
/// stays, with why, for `off`, a binary run without a cache directory, a file without a
/// key, a plan that puts no routed expert on a card, and a profile [`read`]
/// refuses or that is not there yet; the target stays for every case past
/// the key, so a running machine writes the file.
pub fn seed(
    plan: &mut Plan<'_>,
    lever: &ProfileLever,
    key: &Result<ModelKey, String>,
) -> Result<(), PlacementError> {
    let prefix = |path: Option<&Path>, why: String| CardSeed::Prefix {
        path: path.map(Path::to_path_buf),
        why,
    };
    let path = match (lever, key) {
        (ProfileLever::Off, _) => {
            plan.seed = Seed {
                card: prefix(None, "BLOOMERY_RES_PROFILE=off".to_string()),
                target: None,
            };
            return Ok(());
        }
        (ProfileLever::NotActed, _) => {
            plan.seed = Seed {
                card: prefix(
                    None,
                    "this binary does not act on BLOOMERY_RES_PROFILE".to_string(),
                ),
                target: None,
            };
            return Ok(());
        }
        (ProfileLever::NoCache, _) => {
            plan.seed = Seed {
                card: prefix(
                    None,
                    "neither XDG_CACHE_HOME nor HOME is set: no profile directory".to_string(),
                ),
                target: None,
            };
            return Ok(());
        }
        (_, Err(e)) => {
            plan.seed = Seed {
                card: prefix(None, format!("no model key: {e}")),
                target: None,
            };
            return Ok(());
        }
        (ProfileLever::File(p), Ok(_)) => p.clone(),
        (ProfileLever::Dir(d), Ok(k)) => d.join(&k.stem),
    };
    let Ok(key) = key else {
        unreachable!("an error key returned above")
    };
    let model = plan.model;
    let (layers, experts) = (model.layers, model.experts as usize);
    let target = Some(ProfileTarget {
        path: path.clone(),
        key: key.clone(),
        layers,
        experts,
    });
    let on_card = plan.rows.iter().any(|r| {
        r.segments.iter().any(|s| {
            matches!(s.device, Device::Card(_)) && s.experts.as_ref().is_some_and(|e| !e.is_empty())
        })
    });
    let card = if !on_card {
        prefix(
            Some(&path),
            "the plan puts no routed expert on a card".to_string(),
        )
    } else {
        match read(&path, key, layers, experts) {
            Ok(p) => {
                relist(plan, &p)?;
                CardSeed::Profile {
                    path: path.clone(),
                    pairs: p.pairs(),
                }
            }
            Err(e) if e.not_found() => prefix(Some(&path), "no profile there yet".to_string()),
            Err(e) => prefix(Some(&path), e.to_string()),
        }
    };
    plan.seed = Seed { card, target };
    Ok(())
}

/// Each routed row with card segments rebuilt from `p`'s order of its
/// layer: the card segments, in the plan's order, take consecutive runs of
/// it of their own lengths.
fn relist(plan: &mut Plan<'_>, p: &Profile) -> Result<(), PlacementError> {
    let model = plan.model;
    for row in &mut plan.rows {
        let t = model.tensors.get(row.tensor).ok_or_else(|| {
            PlacementError::Experts(format!("plan row {} names no tensor", row.tensor))
        })?;
        if t.role != Role::RoutedExperts {
            continue;
        }
        let cards: Vec<(usize, usize, Format)> = row
            .segments
            .iter()
            .filter_map(|s| match s.device {
                Device::Card(c) => {
                    Some((c, s.experts.as_ref().map_or(0, |e| e.ids().len()), s.format))
                }
                _ => None,
            })
            .collect();
        let Some(&(_, _, format)) = cards.first() else {
            continue;
        };
        let Format::Card(card_format) = format else {
            return Err(PlacementError::Experts(format!(
                "{}: a card segment in {format}",
                t.name
            )));
        };
        if cards.iter().any(|c| c.2 != format) {
            return Err(PlacementError::Experts(format!(
                "{}: card segments of more than one format",
                t.name
            )));
        }
        let layer = t.layer.filter(|&l| l < p.layers).ok_or_else(|| {
            PlacementError::Experts(format!(
                "{}: a routed stack of no layer the profile holds",
                t.name
            ))
        })?;
        let order = p.order(layer);
        let mut at = 0usize;
        let mut lists = Vec::with_capacity(cards.len());
        for &(c, n, _) in &cards {
            let ids = order.get(at..at + n).ok_or_else(|| {
                PlacementError::Experts(format!(
                    "{}: {} experts on cards of a stack of {}",
                    t.name,
                    at + n,
                    order.len()
                ))
            })?;
            lists.push((c, ExpertList::new(ids.to_vec(), model.experts)?));
            at += n;
        }
        *row = routed_row_on_as(row.tensor, t, row.stage, lists, model, Some(card_format))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> ModelKey {
        ModelKey {
            stem: "Model-Q4".to_string(),
            shards: vec![
                ("Model-Q4-00001-of-00002.gguf".to_string(), 7),
                ("Model-Q4-00002-of-00002.gguf".to_string(), 9),
            ],
        }
    }

    fn profile() -> Profile {
        let mut p = Profile::new(key(), 2, 4);
        p.counts = vec![0, 5, 5, 1, 9, 0, 0, 0];
        p
    }

    /// The bytes round-trip, and every refusal names its cause: short,
    /// magic, version, shape past or short, sum, key text.
    #[test]
    fn the_file_round_trips_and_refuses_by_name() {
        let path = Path::new("/p");
        let p = profile();
        let bytes = p.encode();
        assert_eq!(Profile::decode(&bytes, path).expect("whole"), p);
        let err = |b: &[u8]| Profile::decode(b, path).expect_err("refused").to_string();
        assert!(
            err(&bytes[..10]).contains("short of the 24 its header takes"),
            "{}",
            err(&bytes[..10])
        );
        assert!(err(&bytes[..bytes.len() - 1]).contains("its shape takes"));
        let mut long = bytes.clone();
        long.push(0);
        assert!(err(&long).contains("past the"));
        let mut magic = bytes.clone();
        magic[0] = b'X';
        assert!(err(&magic).contains("magic"));
        let mut version = bytes.clone();
        version[8] = 2;
        assert!(err(&version).contains("version 2"));
        let mut flipped = bytes.clone();
        let at = flipped.len() - 9;
        flipped[at] ^= 1;
        assert!(err(&flipped).contains("sum"));
    }

    /// A read holds the file to the model's shape and key.
    #[test]
    fn a_read_refuses_another_shape_or_model() {
        let dir =
            std::env::temp_dir().join(format!("bloomery-profile-read-{}", std::process::id()));
        let path = dir.join("Model-Q4");
        write(&path, &profile()).expect("written");
        assert_eq!(read(&path, &key(), 2, 4).expect("read"), profile());
        let shape = read(&path, &key(), 3, 4).expect_err("shape").to_string();
        assert!(
            shape.contains("2 layers of 4 experts; the model has 3 of 4"),
            "{shape}"
        );
        let mut other = key();
        other.shards[1].1 = 10;
        let model = read(&path, &other, 2, 4).expect_err("model").to_string();
        assert!(model.contains("not this one"), "{model}");
        let missing = read(&dir.join("none"), &key(), 2, 4).expect_err("missing");
        assert!(missing.not_found());
        std::fs::remove_dir_all(&dir).expect("cleaned");
    }

    /// Additions sum into the file; a file refused at the plan is replaced.
    #[test]
    fn additions_sum_and_replace_a_refused_file() {
        let dir = std::env::temp_dir().join(format!("bloomery-profile-add-{}", std::process::id()));
        let target = ProfileTarget {
            path: dir.join("Model-Q4"),
            key: key(),
            layers: 2,
            experts: 4,
        };
        let delta = [1, 0, 0, 0, 0, 0, 0, 2];
        assert_eq!(add_to_file(&target, &delta).expect("first").counts, delta);
        assert_eq!(
            add_to_file(&target, &delta).expect("second").counts,
            [2, 0, 0, 0, 0, 0, 0, 4]
        );
        std::fs::write(&target.path, b"torn").expect("torn");
        assert_eq!(
            add_to_file(&target, &delta).expect("replaced").counts,
            delta
        );
        std::fs::remove_dir_all(&dir).expect("cleaned");
    }

    /// Count descending, id ascending; uncounted experts after, by id.
    #[test]
    fn the_order_is_rank_key_with_the_prefix_after() {
        let p = profile();
        assert_eq!(p.order(0), [1, 2, 3, 0]);
        assert_eq!(p.order(1), [0, 1, 2, 3]);
        assert_eq!(p.pairs(), 4);
    }

    /// The lever: off, a path, the cache directory (an absolute
    /// `XDG_CACHE_HOME`, else `HOME`'s `.cache`), none; and nothing, unset or
    /// set, for a binary that does not act on it.
    #[test]
    fn the_lever_resolves_unset_under_the_cache_directory() {
        use bloomery_levers::ResProfile;
        let os = |s: &str| Some(OsString::from(s));
        assert_eq!(
            ProfileLever::of(true, Some(ResProfile::Off), os("/h"), os("/c")),
            ProfileLever::Off
        );
        assert_eq!(
            ProfileLever::of(true, Some(ResProfile::File(Path::new("/x"))), None, None),
            ProfileLever::File(PathBuf::from("/x"))
        );
        assert_eq!(
            ProfileLever::of(true, None, os("/h"), None),
            ProfileLever::Dir(PathBuf::from("/h/.cache/bloomery/profiles"))
        );
        assert_eq!(
            ProfileLever::of(true, None, os("/h"), os("/run/c")),
            ProfileLever::Dir(PathBuf::from("/run/c/bloomery/profiles"))
        );
        assert_eq!(
            ProfileLever::of(true, None, None, os("/run/c")),
            ProfileLever::Dir(PathBuf::from("/run/c/bloomery/profiles"))
        );
        assert_eq!(
            ProfileLever::of(true, None, os("/h"), os("rel/c")),
            ProfileLever::Dir(PathBuf::from("/h/.cache/bloomery/profiles"))
        );
        assert_eq!(
            ProfileLever::of(true, None, None, os("")),
            ProfileLever::NoCache
        );
        assert_eq!(
            ProfileLever::of(false, None, os("/h"), os("/c")),
            ProfileLever::NotActed
        );
        assert_eq!(ProfileLever::default(), ProfileLever::Off);
    }
}
