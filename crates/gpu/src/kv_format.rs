//! The format a GQA load holds its K/V cache planes in, and the one rule that picks it.
//!
//! Every load whose attention keeps the GQA cache layout (`[n_kv][ctx][head]` a side, in f16
//! or in q8_0's two planes, [`q8_plane_lens`](crate::rope_neox::q8_plane_lens)) takes its
//! format from [`KvChoice::choose`]: `--cache-type-k`'s word, else the `BLOOMERY_QWEN3_KV`
//! lever's, else f16, whatever the file's container.

use crate::GpuError;

/// The format a load holds its KV cache planes in: f16, or q8_0's two-plane layout, 17/16
/// bytes a value against f16's 2, both planes (K and V) quantized together by the one choice
/// ([`KvChoice::choose`]); the append quantizes each row and the flash reads it through the
/// format's algebra.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvQ8 {
    /// `[n_kv][ctx][head]` f16 a side.
    #[default]
    F16,
    /// Per side a codes plane of `head/4` u32 a row and a scales plane of `head/32` u16
    /// ([`q8_plane_lens`](crate::rope_neox::q8_plane_lens)).
    Q8,
}

impl KvQ8 {
    /// The cache format selected by a layout's q8_0 bit.
    #[must_use]
    pub const fn of(q8: bool) -> KvQ8 {
        if q8 { KvQ8::Q8 } else { KvQ8::F16 }
    }

    /// The word the lever and `--cache-type-k` take (llama-server's spelling), or `None` on
    /// any other.
    #[must_use]
    pub fn parse(word: &str) -> Option<KvQ8> {
        match word {
            "f16" => Some(KvQ8::F16),
            "q8_0" => Some(KvQ8::Q8),
            _ => None,
        }
    }

    /// The word [`KvQ8::parse`] took.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            KvQ8::F16 => "f16",
            KvQ8::Q8 => "q8_0",
        }
    }
}

/// What chose a load's cache format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvFrom {
    /// `--cache-type-k`'s word.
    Flag,
    /// `BLOOMERY_QWEN3_KV`'s word, the flag unset.
    Lever,
    /// Neither set: f16.
    Default,
}

impl KvFrom {
    /// The word a load's record prints.
    #[must_use]
    pub const fn word(self) -> &'static str {
        match self {
            KvFrom::Flag => "flag",
            KvFrom::Lever => "lever",
            KvFrom::Default => "default",
        }
    }
}

/// A load's cache format and what chose it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvChoice {
    pub kv: KvQ8,
    pub from: KvFrom,
}

impl KvChoice {
    /// The format `flag` (`--cache-type-k`) names, else the one `lever` names; `None` when
    /// neither is set. Any other word is refused by name, from the flag or the lever: what a
    /// seat checks before the model is fetched.
    pub fn asked(flag: Option<&str>, lever: Option<&str>) -> Result<Option<KvChoice>, GpuError> {
        let (word, from) = match (flag, lever) {
            (Some(word), _) => (word, KvFrom::Flag),
            (None, Some(word)) => (word, KvFrom::Lever),
            (None, None) => return Ok(None),
        };
        match KvQ8::parse(word) {
            Some(kv) => Ok(Some(KvChoice { kv, from })),
            None => Err(GpuError::shape(
                "KvChoice::asked",
                format!("--cache-type-k takes f16 or q8_0, not {word:?}"),
            )),
        }
    }

    /// [`KvChoice::asked`], else f16.
    pub fn choose(flag: Option<&str>, lever: Option<&str>) -> Result<KvChoice, GpuError> {
        Ok(KvChoice::asked(flag, lever)?.unwrap_or(KvChoice {
            kv: KvQ8::F16,
            from: KvFrom::Default,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::{KvChoice, KvFrom, KvQ8};
    use crate::GpuError;

    // PIN(2026-11-02): the one rule replaces the folder-to-q8_0 cases; the file's container
    // no longer selects the default cache format.
    /// The cache format: `--cache-type-k`'s word wins over the lever's, the lever's over the
    /// default, f16. Any other word is refused by name, from the flag or the lever.
    #[test]
    fn the_cache_format_is_the_flags_then_the_levers_then_f16() {
        let choose = |flag, lever| KvChoice::choose(flag, lever).expect("a cache word");
        let set = |kv, from| KvChoice { kv, from };
        assert_eq!(choose(None, None), set(KvQ8::F16, KvFrom::Default));
        assert_eq!(KvChoice::asked(None, None).expect("unset"), None);
        assert_eq!(choose(Some("f16"), None), set(KvQ8::F16, KvFrom::Flag));
        assert_eq!(
            choose(Some("q8_0"), Some("f16")),
            set(KvQ8::Q8, KvFrom::Flag)
        );
        assert_eq!(choose(None, Some("q8_0")), set(KvQ8::Q8, KvFrom::Lever));
        assert_eq!(
            choose(Some("f16"), Some("fp8")),
            set(KvQ8::F16, KvFrom::Flag)
        );
        for (flag, lever) in [
            (Some("q8"), None),
            (None, Some("fp8")),
            (Some("q4"), Some("f16")),
            (Some(""), None),
        ] {
            let e: GpuError =
                KvChoice::choose(flag, lever).expect_err("a word the flag does not take");
            assert!(matches!(
                &e,
                GpuError::Shape {
                    what: "KvChoice::asked",
                    ..
                }
            ));
            assert!(
                e.to_string().contains("--cache-type-k takes f16 or q8_0"),
                "{e}"
            );
        }
    }
}
