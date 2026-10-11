//! Qwen3.8's MTP draft head: the one rule that picks the head a drafted load
//! scores with from `BLOOMERY_MTP_HEAD_ROWS`, and the row list the crate
//! ships for it unset.
//!
//! Unset, the head is [`SHIPPED`] — the Korean, Chinese and Japanese list
//! at 76,000 rows, `crates/model/data/` built in, so a load finds it with no
//! working directory and no data directory — read through the same reader
//! and the same refusals as a list given by path ([`parse_head_rows`]). Its bytes
//! are pinned ([`ListDef::sha256`]): a list whose bytes are not the pinned
//! ones is refused by name before it is read. On a target whose tokenizer is
//! not the one the list's first line names (another vocabulary size or
//! another digest) the head is the full head, and the pick says why.
//! `full` is the full head; a path is that list, any refusal of it by name.

use std::fmt;

use bloomery_levers::MtpHead;
use gguf::Split;
use models::headpick::{
    HeadRowsError, HeadSet, ListDef, parse_head_rows, read_head_rows, vocab_sha256,
};

use super::place::PlaceError;
use crate::fileio::sha256_hex;

/// The shipped list: Qwen3.8's Korean, Chinese and Japanese list at 76,000 rows.
pub const SHIPPED: ListDef = ListDef {
    name: "cjk-76000",
    path: "crates/model/data/mtp-head-rows-qwen38-cjk-76000.txt",
    text: include_str!("../../../data/mtp-head-rows-qwen38-cjk-76000.txt"),
    sha256: "e20c786dd2c1471343273c74794d93ae000f74efb844626d9fd9dc5a566a2cea",
};

/// Why the shipped list is not read.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ShippedError {
    #[error(
        "the shipped head list ({}): its bytes digest to {got}, not the {} the crate pins",
        SHIPPED.path,
        SHIPPED.sha256
    )]
    Pinned { got: String },
    #[error(transparent)]
    Rows(#[from] HeadRowsError),
}

/// Why a load's head was picked: the record's `from` and `why`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeadWhy {
    /// Unset, and the target's tokenizer is the shipped list's.
    Shipped,
    /// Unset, and the target's tokenizer is not the shipped list's: the
    /// full head, with the reader's sentence.
    OtherTokenizer(String),
    /// `BLOOMERY_MTP_HEAD_ROWS=full`.
    SetFull,
    /// `BLOOMERY_MTP_HEAD_ROWS=<path>`.
    SetList(String),
}

impl HeadWhy {
    /// The record's `from`: `shipped`, `other-tokenizer` or `set`.
    #[must_use]
    pub fn from_word(&self) -> &'static str {
        match self {
            HeadWhy::Shipped => "shipped",
            HeadWhy::OtherTokenizer(_) => "other-tokenizer",
            HeadWhy::SetFull | HeadWhy::SetList(_) => "set",
        }
    }
}

impl fmt::Display for HeadWhy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HeadWhy::Shipped => write!(
                f,
                "BLOOMERY_MTP_HEAD_ROWS unset: the shipped list {}, the target's \
                 tokenizer its own",
                SHIPPED.name
            ),
            HeadWhy::OtherTokenizer(why) => write!(
                f,
                "BLOOMERY_MTP_HEAD_ROWS unset: the full head, the shipped list {} \
                 is another tokenizer's ({why})",
                SHIPPED.name
            ),
            HeadWhy::SetFull => write!(f, "BLOOMERY_MTP_HEAD_ROWS=full"),
            HeadWhy::SetList(path) => write!(f, "BLOOMERY_MTP_HEAD_ROWS={path}"),
        }
    }
}

/// The head a drafted load scores with, and why.
#[derive(Clone, Debug, PartialEq)]
pub struct HeadPick {
    pub rows: HeadSet,
    pub why: HeadWhy,
}

impl HeadPick {
    /// The record's `head`: `list` or `full`.
    #[must_use]
    pub fn head_word(&self) -> &'static str {
        match self.rows {
            HeadSet::Full => "full",
            HeadSet::List { .. } => "list",
        }
    }

    /// The rows the head scores: the list's, or `vocab` for the full head.
    #[must_use]
    pub fn rows_of(&self, vocab: u32) -> usize {
        match &self.rows {
            HeadSet::Full => vocab as usize,
            HeadSet::List { ids, .. } => ids.len(),
        }
    }
}

/// The head `set` (`BLOOMERY_MTP_HEAD_ROWS` as `Levers::mtp_head_rows`
/// reads it) picks for the target file `target`, of `vocab` tokens: the
/// module doc's rule.
pub fn head_rows_of(
    set: Option<MtpHead<'_>>,
    target: &Split,
    vocab: u32,
) -> Result<HeadPick, HeadListError> {
    match set {
        Some(MtpHead::Full) => Ok(HeadPick {
            rows: HeadSet::Full,
            why: HeadWhy::SetFull,
        }),
        Some(MtpHead::List(path)) => Ok(HeadPick {
            rows: read_head_rows(path, target, vocab).map_err(PlaceError::from)?,
            why: HeadWhy::SetList(path.display().to_string()),
        }),
        None => {
            let digest = vocab_sha256(target).map_err(PlaceError::from)?;
            Ok(shipped_rows(SHIPPED.text, vocab, &digest)?)
        }
    }
}

/// The shipped list's bytes `text` read for a target of `vocab` tokens whose
/// tokenizer digests to `digest`: refused by name when `text`'s SHA-256 is
/// not [`SHIPPED`]'s pin ([`ListDef::sha256`]), the full head when the
/// list's vocabulary or digest is not the target's, else the list; any
/// other refusal of the reader is the error.
pub fn shipped_rows(text: &str, vocab: u32, digest: &[u8; 32]) -> Result<HeadPick, ShippedError> {
    let got = sha256_hex(text.as_bytes());
    if got != SHIPPED.sha256 {
        return Err(ShippedError::Pinned { got });
    }
    let what = format!("the shipped list {}", SHIPPED.name);
    match parse_head_rows(&what, text, vocab, digest) {
        Ok(rows) => Ok(HeadPick {
            rows,
            why: HeadWhy::Shipped,
        }),
        Err(e @ (HeadRowsError::Vocab { .. } | HeadRowsError::Digest { .. })) => Ok(HeadPick {
            rows: HeadSet::Full,
            why: HeadWhy::OtherTokenizer(e.to_string()),
        }),
        Err(e) => Err(e.into()),
    }
}

/// Why [`head_rows_of`] picked no head.
#[derive(Debug, thiserror::Error)]
pub enum HeadListError {
    #[error(transparent)]
    Shipped(#[from] ShippedError),
    #[error(transparent)]
    Place(#[from] PlaceError),
}

#[cfg(test)]
mod tests {
    use super::{HeadWhy, SHIPPED, ShippedError, shipped_rows};
    use models::headpick::{HEAD_ROWS_FORMAT, HeadRowsError, HeadSet};

    /// The digest the shipped list's first line names, as bytes.
    fn named_digest() -> [u8; 32] {
        let first = SHIPPED.text.lines().next().expect("a first line");
        let hexs = first
            .split_whitespace()
            .find_map(|w| w.strip_prefix("vocab_sha256="))
            .expect("a vocab_sha256 field");
        let mut d = [0u8; 32];
        for (i, b) in d.iter_mut().enumerate() {
            *b = u8::from_str_radix(&hexs[2 * i..2 * i + 2], 16).expect("hex");
        }
        d
    }

    /// The shipped list reads as its first line says: Qwen3.8's vocabulary,
    /// 76,000 ascending ids, on a target of the digest it names.
    #[test]
    fn shipped_list_reads_on_its_own_tokenizer() {
        let first = SHIPPED.text.lines().next().expect("a first line");
        assert!(first.starts_with(HEAD_ROWS_FORMAT), "{first}");
        assert!(first.contains(" vocab=248320 rows=76000 "), "{first}");
        let pick = shipped_rows(SHIPPED.text, 248_320, &named_digest()).expect("the shipped list");
        assert_eq!(pick.why, HeadWhy::Shipped);
        match pick.rows {
            HeadSet::List { ids, digest } => {
                assert_eq!(ids.len(), 76_000);
                assert_eq!(digest, named_digest());
            }
            HeadSet::Full => panic!("the full head"),
        }
    }

    /// Another tokenizer's target, by digest or by vocabulary size, gets the
    /// full head and the reader's sentence; the bytes are still pinned.
    #[test]
    fn shipped_list_on_another_tokenizer_is_the_full_head() {
        let mut other = named_digest();
        other[0] ^= 1;
        let pick = shipped_rows(SHIPPED.text, 248_320, &other).expect("a pick");
        assert_eq!(pick.rows, HeadSet::Full);
        assert!(
            matches!(&pick.why, HeadWhy::OtherTokenizer(w) if w.contains("vocab_sha256")),
            "{:?}",
            pick.why
        );
        let pick = shipped_rows(SHIPPED.text, 151_936, &named_digest()).expect("a pick");
        assert_eq!(pick.rows, HeadSet::Full);
        assert!(
            matches!(&pick.why, HeadWhy::OtherTokenizer(w) if w.contains("vocabulary of 248320")),
            "{:?}",
            pick.why
        );
    }

    /// One byte of the list changed — a hex digit of its digest line, still
    /// a well-formed line, or an id — is refused by name by the pin, before
    /// the reader could take it for another tokenizer's list.
    #[test]
    fn a_changed_byte_of_the_shipped_list_is_refused_by_name() {
        let at = SHIPPED.text.find("vocab_sha256=").expect("the field") + "vocab_sha256=".len();
        let mut bytes = SHIPPED.text.as_bytes().to_vec();
        bytes[at] = if bytes[at] == b'0' { b'1' } else { b'0' };
        let line = String::from_utf8(bytes).expect("ascii");
        let err = shipped_rows(&line, 248_320, &named_digest()).expect_err("refused");
        assert!(matches!(err, ShippedError::Pinned { .. }), "{err}");
        assert!(err.to_string().contains("the crate pins"), "{err}");
        let id = SHIPPED.text.replacen("\n7\n", "\n8\n", 1);
        let err = shipped_rows(&id, 248_320, &named_digest()).expect_err("refused");
        assert!(matches!(err, ShippedError::Pinned { .. }), "{err}");
        assert!(!matches!(
            err,
            ShippedError::Rows(HeadRowsError::Digest { .. })
        ));
    }
}
