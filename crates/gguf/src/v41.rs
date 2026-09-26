//! Which DeepSeek-V4.1-Flash file this tree runs: the one Rust owner of the
//! path. It lives here because every crate that opens the file — the engine,
//! the engram tables, the tokenizer, the qdot and placement tests, the gate
//! binaries — depends on this crate, and this crate on none of them.
//!
//! The shell owner is the deepseek41 model profile
//! (`tools/ref/models/deepseek41.sh`, its `V41_MODEL`); `tools/box.sh`
//! exports that choice into every box command as `BLOOMERY_V41_MODEL` (and
//! its directory as `BLOOMERY_V41_DIR`), whatever profile the command picked,
//! so the two owners name one file. [`DEFAULT`] is what a process started
//! outside `box.sh` opens.
//!
//! The file is the public `Q3_K_M` set ([`PUBLIC`]), the default. The oracle
//! sets dumped from it carry the name suffix [`PUBLIC_SET_SUFFIX`]; a set of
//! any other file carries none.

use std::path::PathBuf;

/// The environment variable that names the file: shard 1 of a split set.
pub const ENV: &str = "BLOOMERY_V41_MODEL";

/// The public `Q3_K_M` file's first shard.
pub const PUBLIC: &str =
    "/models/DeepSeek-V4.1-Flash-Q3_K_M/DeepSeek-V4.1-Flash-Q3_K_M-00001-of-00009.gguf";

/// The file opened when [`ENV`] is unset or empty.
pub const DEFAULT: &str = PUBLIC;

/// The oracle set name suffix of the public file's sets.
pub const PUBLIC_SET_SUFFIX: &str = "_plain";

/// The V4.1 first shard to open: [`ENV`], else [`DEFAULT`]. An empty value
/// counts as unset, as in the profile.
#[must_use]
pub fn model() -> String {
    match std::env::var(ENV) {
        Ok(p) if !p.is_empty() => p,
        _ => DEFAULT.to_string(),
    }
}

/// The directory of [`model`]: where every shard of the split set lies.
#[must_use]
pub fn dir() -> PathBuf {
    let m = PathBuf::from(model());
    m.parent().map_or_else(|| PathBuf::from("."), PathBuf::from)
}

/// The oracle set name suffix of the sets dumped from `model`: the public
/// file's [`PUBLIC_SET_SUFFIX`], nothing for any other file. A set opened for
/// another file than it was dumped from is caught by its `# model` line, not
/// by its name.
#[must_use]
pub fn set_suffix_of(model: &str) -> &'static str {
    if model == PUBLIC {
        PUBLIC_SET_SUFFIX
    } else {
        ""
    }
}

/// The name of V4.1 oracle set `base` (`ref_deepseek41`,
/// `ref_deepseek41_d1_every_node`, …) for the file [`model`] names.
#[must_use]
pub fn set(base: &str) -> String {
    format!("{base}{}", set_suffix_of(&model()))
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT, PUBLIC, PUBLIC_SET_SUFFIX, set_suffix_of};

    /// The public file's sets carry the suffix, any other file's none. The
    /// default is the public file, so an unset variable opens the suffixed
    /// sets.
    #[test]
    fn set_suffix_follows_the_file() {
        assert_eq!(set_suffix_of(PUBLIC), PUBLIC_SET_SUFFIX);
        assert_eq!(set_suffix_of("/models/other/other-00001-of-00009.gguf"), "");
        assert_eq!(set_suffix_of(DEFAULT), PUBLIC_SET_SUFFIX);
    }
}
