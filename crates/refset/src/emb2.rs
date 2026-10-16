//! The EmbeddingGemma v1 and v2 sets `tools/ref/emb2/emb2.sh` writes ([`Emb2Set`]): `dump_emb2`'s text
//! and tower dumps and their CPU twins, and `server_vectors.py`'s llama-server
//! vectors. A family of these sets has [`Identity::Emb2`], which carries the
//! [`Kind`] of set and the [`Profile`] (the files every set is dumped from and
//! their digests); its reader runs the node dumps' checks, then the kind's.

use crate::RefError;
use crate::clefvis::one;
use crate::family::{Build, Family, Identity};
use crate::ik::RefManifest;
use std::path::Path;

/// The kind of set, which its `# emb2` and `# device` lines state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Text,
    TextCpu,
    Tower,
    TowerCpu,
    Server,
    ServerCpu,
}

impl Kind {
    /// The `# emb2` line's value: the dumper's mode, or `server`.
    #[must_use]
    pub const fn set_kind(self) -> &'static str {
        match self {
            Kind::Text | Kind::TextCpu => "text",
            Kind::Tower | Kind::TowerCpu => "tower",
            Kind::Server | Kind::ServerCpu => "server",
        }
    }

    /// The `# device` line's first field.
    #[must_use]
    pub const fn device(self) -> &'static str {
        match self {
            Kind::TextCpu | Kind::TowerCpu | Kind::ServerCpu => "cpu",
            Kind::Text | Kind::Tower | Kind::Server => "cuda",
        }
    }
}

/// The files every set of a model is dumped from. A family's
/// `Identity::Emb2` carries one.
#[derive(Debug, PartialEq, Eq)]
pub struct Profile {
    /// The text model file.
    pub model: &'static str,
    /// The sha256 of [`Profile::model`].
    pub model_sha256: &'static str,
    /// The optional projector path and SHA256 for tower and multimodal server sets.
    pub mmproj: Option<(&'static str, &'static str)>,
}

/// A set that passed its family's check.
#[derive(Debug, Clone)]
pub struct Emb2Set {
    pub man: RefManifest,
}

impl Emb2Set {
    /// Read the set at `dir` as a set of `family`, a family of identity [`Identity::Emb2`]: against the family
    /// ([`RefManifest::open`]: the trailer, the model file, the architecture and the mainline build), then against
    /// its kind. A set of another kind or device is [`RefError::Foreign`] on `emb2` or `device`; one whose text file or
    /// projector has another path or sha256 is [`RefError::Stale`]; a line that is missing or of another shape is
    /// [`RefError::Missing`] or [`RefError::Malformed`] naming it, and so is a family of another identity.
    /// A tower set without its first block tap is malformed.
    pub fn open(dir: &Path, family: &Family) -> Result<Emb2Set, RefError> {
        let at = format!("emb2: {}", dir.display());
        let Identity::Emb2(kind, profile) = family.identity else {
            return Err(RefError::malformed(
                &at,
                format!("the {} family is not an emb2 family", family.name),
            ));
        };
        let man = RefManifest::open(dir, family)?;
        let foreign = |field: &'static str, got: &str, want: &str| RefError::Foreign {
            set: dir.display().to_string(),
            family: family.name,
            field,
            got: got.to_string(),
            want: want.to_string(),
        };
        let stale =
            |what: &str, path: &str, digest: &str, want: &str, want_digest: &str| RefError::Stale {
                set: dir.display().to_string(),
                dumped_from: format!("{what} {path} sha256 {digest}"),
                runs: format!("{what} {want} sha256 {want_digest}"),
            };
        let line = one(&man, "emb2")?.join("\t");
        if line != kind.set_kind() {
            return Err(foreign("emb2", &line, kind.set_kind()));
        }
        let dev = one(&man, "device")?;
        let [device, "threads", _, "card", _] = dev[..] else {
            return Err(RefError::malformed(
                &at,
                format!("{dev:?}: want cuda|cpu\tthreads\t<n>\tcard\t<name>"),
            ));
        };
        if device != kind.device() {
            return Err(foreign("device", device, kind.device()));
        }
        // The text file: its own sha256 line in a text or server set, beside its path in a tower set (whose `# model` is
        // the projector).
        let (text, text_digest) = match kind {
            Kind::Tower | Kind::TowerCpu => match one(&man, "text_model")?[..] {
                [path, "sha256", digest] => (path, digest),
                ref other => {
                    return Err(RefError::malformed(
                        &at,
                        format!("{other:?}: want <path>\tsha256\t<digest>"),
                    ));
                }
            },
            Kind::Text | Kind::TextCpu | Kind::Server | Kind::ServerCpu => {
                let model = one(&man, "model")?;
                let [path] = model[..] else {
                    return Err(RefError::malformed(
                        &at,
                        format!("{model:?}: want <the text file's path>"),
                    ));
                };
                match one(&man, "model_sha256")?[..] {
                    [digest] => (path, digest),
                    ref other => {
                        return Err(RefError::malformed(
                            &at,
                            format!("{other:?}: want <digest>"),
                        ));
                    }
                }
            }
        };
        if text != profile.model || text_digest != profile.model_sha256 {
            return Err(stale(
                "model",
                text,
                text_digest,
                profile.model,
                profile.model_sha256,
            ));
        }
        if kind != Kind::Text
            && kind != Kind::TextCpu
            && let Some((want_path, want_digest)) = profile.mmproj
        {
            let mm = one(&man, "mmproj")?;
            let [path, "sha256", digest] = mm[..] else {
                return Err(RefError::malformed(
                    &at,
                    format!("{mm:?}: want <path>\tsha256\t<digest>"),
                ));
            };
            if path != want_path || digest != want_digest {
                return Err(stale("mmproj", path, digest, want_path, want_digest));
            }
        }
        if matches!(kind, Kind::Tower | Kind::TowerCpu)
            && !man
                .tensors
                .iter()
                .any(|row| row.name.ends_with("/layer_out-0"))
        {
            return Err(RefError::malformed(
                &at,
                "the tower has no layer_out-0 block tap",
            ));
        }
        Ok(Emb2Set { man })
    }
}

/// The mainline commit shared by the EmbeddingGemma oracle families.
// PIN(2026-10-14): 6868a52cb337450658006ad2f9281c2eb63f1666, a pinned master
// commit that contains the no-cache attention pad, the pooled-embedding
// LM-head skip and the t_embd expand removal on this model's path.
pub const LCPP_BUILD: &str = "6868a52cb";

pub(crate) const fn family(
    name: &'static str,
    sets: &'static [&'static str],
    kind: Kind,
    profile: &'static Profile,
    arch: &'static str,
    runs: fn() -> Result<String, RefError>,
    recipe: &'static str,
) -> Family {
    Family {
        name,
        sets,
        resolve: None,
        recipe,
        identity: Identity::Emb2(kind, profile),
        arch: Some(arch),
        build: Some(Build::Is(LCPP_BUILD)),
        runs: Some(runs),
        draft_runs: None,
        consumers: &[],
    }
}

#[cfg(test)]
pub(crate) mod tests;
