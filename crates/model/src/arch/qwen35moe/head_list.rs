//! Qwen3.8's MTP draft head: the one rule that picks the head a drafted load
//! scores with from `BLOOMERY_MTP_HEAD_ROWS`, and the row list the crate
//! ships for it unset.
//!
//! Unset, the head is [`SHIPPED`] — the hangul family at 65,536 rows,
//! `crates/model/data/` built in, so a load finds it with no working
//! directory and no data directory — read through the same reader and the
//! same refusals as a list given by path ([`parse_head_rows`]). Its bytes
//! are pinned ([`SHIPPED_SHA256`]): a list whose bytes are not the pinned
//! ones is refused by name before it is read. On a target whose tokenizer is
//! not the one the list's first line names (another vocabulary size or
//! another digest) the head is the full head, and the pick says why.
//! `full` is the full head; a path is that list, any refusal of it by name.

use std::fmt;

use bloomery_levers::MtpHead;
use gguf::Split;
use models::HeadRows;

use super::place::{HeadRowsError, PlaceError, parse_head_rows, read_head_rows, vocab_sha256};
use crate::fileio::sha256_hex;

/// The shipped list's name, as a pick prints it.
pub const SHIPPED_NAME: &str = "kohangul-65536";

/// Where the shipped list lives in the repository.
pub const SHIPPED_PATH: &str = "crates/model/data/mtp-head-rows-qwen38-kohangul-65536.txt";

/// The shipped list: Qwen3.8's hangul family at 65,536 rows.
pub const SHIPPED: &str = include_str!("../../../data/mtp-head-rows-qwen38-kohangul-65536.txt");

/// The SHA-256 of [`SHIPPED`]'s bytes, as the list was built.
pub const SHIPPED_SHA256: &str = "b9deced66fc00185620bb0a8cdc591240e6801e298bdc76438490eed3e426bdc";

/// Why the shipped list is not read.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ShippedError {
    #[error(
        "the shipped head list ({SHIPPED_PATH}): its bytes digest to {got}, not the \
         {SHIPPED_SHA256} the crate pins"
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
                "BLOOMERY_MTP_HEAD_ROWS unset: the shipped list {SHIPPED_NAME}, the target's \
                 tokenizer its own"
            ),
            HeadWhy::OtherTokenizer(why) => write!(
                f,
                "BLOOMERY_MTP_HEAD_ROWS unset: the full head, the shipped list {SHIPPED_NAME} \
                 is another tokenizer's ({why})"
            ),
            HeadWhy::SetFull => write!(f, "BLOOMERY_MTP_HEAD_ROWS=full"),
            HeadWhy::SetList(path) => write!(f, "BLOOMERY_MTP_HEAD_ROWS={path}"),
        }
    }
}

/// The head a drafted load scores with, and why.
#[derive(Clone, Debug, PartialEq)]
pub struct HeadPick {
    pub rows: HeadRows,
    pub why: HeadWhy,
}

impl HeadPick {
    /// The record's `head`: `list` or `full`.
    #[must_use]
    pub fn head_word(&self) -> &'static str {
        match self.rows {
            HeadRows::Full => "full",
            HeadRows::List { .. } => "list",
        }
    }

    /// The rows the head scores: the list's, or `vocab` for the full head.
    #[must_use]
    pub fn rows_of(&self, vocab: u32) -> usize {
        match &self.rows {
            HeadRows::Full => vocab as usize,
            HeadRows::List { ids, .. } => ids.len(),
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
            rows: HeadRows::Full,
            why: HeadWhy::SetFull,
        }),
        Some(MtpHead::List(path)) => Ok(HeadPick {
            rows: read_head_rows(path, target, vocab)?,
            why: HeadWhy::SetList(path.display().to_string()),
        }),
        None => {
            let digest = vocab_sha256(target).map_err(PlaceError::from)?;
            Ok(shipped_rows(SHIPPED, vocab, &digest)?)
        }
    }
}

/// The shipped list's bytes `text` read for a target of `vocab` tokens whose
/// tokenizer digests to `digest`: refused by name when `text`'s SHA-256 is
/// not [`SHIPPED_SHA256`], the full head when the list's vocabulary or
/// digest is not the target's, else the list; any other refusal of the
/// reader is the error.
pub fn shipped_rows(text: &str, vocab: u32, digest: &[u8; 32]) -> Result<HeadPick, ShippedError> {
    let got = sha256_hex(text.as_bytes());
    if got != SHIPPED_SHA256 {
        return Err(ShippedError::Pinned { got });
    }
    let what = format!("the shipped list {SHIPPED_NAME}");
    match parse_head_rows(&what, text, vocab, digest) {
        Ok(rows) => Ok(HeadPick {
            rows,
            why: HeadWhy::Shipped,
        }),
        Err(e @ (HeadRowsError::Vocab { .. } | HeadRowsError::Digest { .. })) => Ok(HeadPick {
            rows: HeadRows::Full,
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
    use crate::arch::qwen35moe::place::{HEAD_ROWS_FORMAT, HeadRowsError};
    use models::HeadRows;

    /// The digest the shipped list's first line names, as bytes.
    fn named_digest() -> [u8; 32] {
        let first = SHIPPED.lines().next().expect("a first line");
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
    /// 65,536 ascending ids, on a target of the digest it names.
    #[test]
    fn shipped_list_reads_on_its_own_tokenizer() {
        let first = SHIPPED.lines().next().expect("a first line");
        assert!(first.starts_with(HEAD_ROWS_FORMAT), "{first}");
        assert!(first.contains(" vocab=248320 rows=65536 "), "{first}");
        let pick = shipped_rows(SHIPPED, 248_320, &named_digest()).expect("the shipped list");
        assert_eq!(pick.why, HeadWhy::Shipped);
        match pick.rows {
            HeadRows::List { ids, digest } => {
                assert_eq!(ids.len(), 65_536);
                assert_eq!(digest, named_digest());
            }
            HeadRows::Full => panic!("the full head"),
        }
    }

    /// Another tokenizer's target, by digest or by vocabulary size, gets the
    /// full head and the reader's sentence; the bytes are still pinned.
    #[test]
    fn shipped_list_on_another_tokenizer_is_the_full_head() {
        let mut other = named_digest();
        other[0] ^= 1;
        let pick = shipped_rows(SHIPPED, 248_320, &other).expect("a pick");
        assert_eq!(pick.rows, HeadRows::Full);
        assert!(
            matches!(&pick.why, HeadWhy::OtherTokenizer(w) if w.contains("vocab_sha256")),
            "{:?}",
            pick.why
        );
        let pick = shipped_rows(SHIPPED, 151_936, &named_digest()).expect("a pick");
        assert_eq!(pick.rows, HeadRows::Full);
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
        let at = SHIPPED.find("vocab_sha256=").expect("the field") + "vocab_sha256=".len();
        let mut bytes = SHIPPED.as_bytes().to_vec();
        bytes[at] = if bytes[at] == b'0' { b'1' } else { b'0' };
        let line = String::from_utf8(bytes).expect("ascii");
        let err = shipped_rows(&line, 248_320, &named_digest()).expect_err("refused");
        assert!(matches!(err, ShippedError::Pinned { .. }), "{err}");
        assert!(err.to_string().contains("the crate pins"), "{err}");
        let id = SHIPPED.replacen("\n7\n", "\n8\n", 1);
        let err = shipped_rows(&id, 248_320, &named_digest()).expect_err("refused");
        assert!(matches!(err, ShippedError::Pinned { .. }), "{err}");
        assert!(!matches!(
            err,
            ShippedError::Rows(HeadRowsError::Digest { .. })
        ));
    }
}
