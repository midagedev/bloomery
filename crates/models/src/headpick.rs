//! The row-list draft head's host side, one owner for every draft that scores a list of the output's rows: the head a
//! draft scores ([`HeadSet`]), a list the crate ships ([`ListDef`]), the list file's reader ([`parse_head_rows`],
//! [`read_head_rows`]) with the digest that ties a list to its tokenizer ([`vocab_sha256`]), and the break-even miss
//! rate under which a list of a given size beats the full head ([`break_even`]).
//!
//! A family keeps only what its tokenizer and its levers decide: the list's bytes, as a [`ListDef`], and the rule that
//! picks a head from its lever (`model::arch::qwen35moe::head_list`).

use std::path::Path;
use std::sync::Arc;

use gguf::{Split, Value};
use sha2::{Digest, Sha256};

/// The output rows an MTP head scores.
#[derive(Clone, Debug, PartialEq)]
pub enum HeadSet {
    /// Every token of the vocabulary.
    Full,
    /// Only these tokens: ids ascending, distinct, each below the vocabulary.
    List {
        ids: Arc<[u32]>,
        /// The SHA-256 of the tokenizer the ids index.
        digest: [u8; 32],
    },
}

/// A row list the crate ships.
#[derive(Clone, Copy)]
pub struct ListDef {
    /// The name a pick prints.
    pub name: &'static str,
    /// Where the list's bytes live in the repository.
    pub path: &'static str,
    /// The list's bytes, as [`parse_head_rows`] reads them.
    pub text: &'static str,
    /// The SHA-256 of `text`'s bytes as lowercase hex, as the list was built: a list whose bytes differ is not this one.
    pub sha256: &'static str,
}

/// The key of the tokenizer's vocabulary in a GGUF header.
const TOKENS: &str = "tokenizer.ggml.tokens";

/// The row list file's first line, before its fields.
pub const HEAD_ROWS_FORMAT: &str = "# bloomery mtp-head-rows 1";

/// The fields of the row list's first line, in order.
const HEAD_ROWS_FIELDS: [&str; 3] = ["vocab", "rows", "vocab_sha256"];

/// Why a row list is refused.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum HeadRowsError {
    #[error("head rows {what}: cannot read: {detail}")]
    Read { what: String, detail: String },
    #[error("metadata {TOKENS}: {detail}")]
    Tokens { detail: String },
    #[error(
        "head rows {what}: line 1 is not `{HEAD_ROWS_FORMAT} vocab=<n> rows=<n> \
         vocab_sha256=<64 hex digits>`: {line:?}"
    )]
    Header { what: String, line: String },
    #[error("head rows {what}: the first line's {field}: {detail}")]
    Field {
        what: String,
        field: &'static str,
        detail: String,
    },
    #[error("head rows {what}: a list over a vocabulary of {file}; the target's has {target}")]
    Vocab {
        what: String,
        file: u64,
        target: u32,
    },
    #[error("head rows {what}: vocab_sha256 {file}; the target's tokenizer is {target}")]
    Digest {
        what: String,
        file: String,
        target: String,
    },
    #[error("head rows {what}: line {line}: {text:?} is not a token id")]
    NotAnId {
        what: String,
        line: usize,
        text: String,
    },
    #[error("head rows {what}: line {line}: id {id} is past the vocabulary of {vocab}")]
    PastVocab {
        what: String,
        line: usize,
        id: u32,
        vocab: u32,
    },
    #[error("head rows {what}: line {line}: id {id} again")]
    Duplicate { what: String, line: usize, id: u32 },
    #[error("head rows {what}: line {line}: id {id} after {prev}; the ids ascend")]
    Unsorted {
        what: String,
        line: usize,
        id: u32,
        prev: u32,
    },
    #[error("head rows {what}: the first line says {said} rows, the file lists {listed}")]
    Count {
        what: String,
        said: u64,
        listed: usize,
    },
    #[error(
        "head rows {what}: {rows} rows; a list holds 1 to {max} (the whole vocabulary is the \
         full head)"
    )]
    Size { what: String, rows: usize, max: u32 },
}

/// The row list in `text` (`what` names it in a refusal), for a target of
/// `vocab` tokens whose tokenizer digests to `digest` ([`vocab_sha256`]):
/// the first line [`HEAD_ROWS_FORMAT`] and its three fields, then one
/// decimal id a line. Refused by name: a first line of another form, a field
/// absent, repeated, unknown or malformed; a vocabulary or a digest other
/// than the target's; a line that is not an id (a blank one too); an id at
/// or past `vocab`, repeated, or below the one before; a count other than
/// the first line's, or outside `1..vocab`.
pub fn parse_head_rows(
    what: &str,
    text: &str,
    vocab: u32,
    digest: &[u8; 32],
) -> Result<HeadSet, HeadRowsError> {
    let mut lines = text.lines();
    let first = lines.next().unwrap_or("");
    let header = || HeadRowsError::Header {
        what: what.to_string(),
        line: first.to_string(),
    };
    let rest = first
        .strip_prefix(HEAD_ROWS_FORMAT)
        .filter(|r| r.is_empty() || r.starts_with(' '))
        .ok_or_else(header)?;
    let field = |field: &'static str, detail: String| HeadRowsError::Field {
        what: what.to_string(),
        field,
        detail,
    };
    let mut values: [Option<&str>; 3] = [None; 3];
    for pair in rest.split_whitespace() {
        let (key, value) = pair.split_once('=').ok_or_else(header)?;
        let Some(i) = HEAD_ROWS_FIELDS.iter().position(|k| *k == key) else {
            return Err(header());
        };
        if values[i].replace(value).is_some() {
            return Err(field(HEAD_ROWS_FIELDS[i], "is given twice".to_string()));
        }
    }
    let get = |i: usize| values[i].ok_or_else(|| field(HEAD_ROWS_FIELDS[i], "is absent".into()));
    let number = |i: usize| -> Result<u64, HeadRowsError> {
        let v = get(i)?;
        v.parse::<u64>()
            .map_err(|_| field(HEAD_ROWS_FIELDS[i], format!("{v:?} is not a count")))
    };
    let file_vocab = number(0)?;
    let said = number(1)?;
    let file_digest = get(2)?;
    if file_digest.len() != 64 || !file_digest.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(field(
            "vocab_sha256",
            format!("{file_digest:?} is not 64 hex digits"),
        ));
    }
    if file_vocab != u64::from(vocab) {
        return Err(HeadRowsError::Vocab {
            what: what.to_string(),
            file: file_vocab,
            target: vocab,
        });
    }
    let want = hex(digest);
    if !file_digest.eq_ignore_ascii_case(&want) {
        return Err(HeadRowsError::Digest {
            what: what.to_string(),
            file: file_digest.to_string(),
            target: want,
        });
    }
    let mut ids: Vec<u32> = Vec::new();
    for (i, text) in lines.enumerate() {
        let line = i + 2;
        let at = |text: &str| HeadRowsError::NotAnId {
            what: what.to_string(),
            line,
            text: text.to_string(),
        };
        if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
            return Err(at(text));
        }
        let id: u32 = text.parse().map_err(|_| at(text))?;
        if id >= vocab {
            return Err(HeadRowsError::PastVocab {
                what: what.to_string(),
                line,
                id,
                vocab,
            });
        }
        match ids.last() {
            Some(&prev) if prev == id => {
                return Err(HeadRowsError::Duplicate {
                    what: what.to_string(),
                    line,
                    id,
                });
            }
            Some(&prev) if prev > id => {
                return Err(HeadRowsError::Unsorted {
                    what: what.to_string(),
                    line,
                    id,
                    prev,
                });
            }
            _ => {}
        }
        ids.push(id);
    }
    if u64::try_from(ids.len()).ok() != Some(said) {
        return Err(HeadRowsError::Count {
            what: what.to_string(),
            said,
            listed: ids.len(),
        });
    }
    if ids.is_empty() || ids.len() >= vocab as usize {
        return Err(HeadRowsError::Size {
            what: what.to_string(),
            rows: ids.len(),
            max: vocab.saturating_sub(1),
        });
    }
    Ok(HeadSet::List {
        ids: ids.into(),
        digest: *digest,
    })
}

/// The row list at `path` ([`parse_head_rows`]) for the target file
/// `target`, of `vocab` tokens: its tokenizer digested ([`vocab_sha256`]).
pub fn read_head_rows(path: &Path, target: &Split, vocab: u32) -> Result<HeadSet, HeadRowsError> {
    let what = path.display().to_string();
    let text = std::fs::read_to_string(path).map_err(|e| HeadRowsError::Read {
        what: what.clone(),
        detail: e.to_string(),
    })?;
    let digest = vocab_sha256(target)?;
    parse_head_rows(&what, &text, vocab, &digest)
}

/// The SHA-256 of `split`'s vocabulary: each entry of
/// `tokenizer.ggml.tokens`, in id order, as its UTF-8 bytes' count (u64,
/// little-endian) and the bytes — the array's payload as the file stores it
/// (`tools/ref/draft-vocab.py` reads the same bytes). Refused by name: no
/// array, an entry that is not a string, and one holding U+FFFD, which is
/// what the reader makes of bytes that are not UTF-8, so its bytes are not
/// the file's.
pub fn vocab_sha256(split: &Split) -> Result<[u8; 32], HeadRowsError> {
    let refuse = |detail: String| HeadRowsError::Tokens { detail };
    let Some(Value::Array(tokens)) = split.value(TOKENS) else {
        return Err(refuse("is absent or not an array".to_string()));
    };
    let mut h = Sha256::new();
    for (i, t) in tokens.iter().enumerate() {
        let Value::String(s) = t else {
            return Err(refuse(format!("entry {i} is {t:?}, not a string")));
        };
        if s.contains('\u{FFFD}') {
            return Err(refuse(format!(
                "entry {i} holds U+FFFD, the reader's stand-in for bytes that are not UTF-8: \
                 its file bytes cannot be digested"
            )));
        }
        h.update((s.len() as u64).to_le_bytes());
        h.update(s.as_bytes());
    }
    Ok(h.finalize().into())
}

/// `bytes` as lowercase hex, two digits a byte.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A window walks the head three times, each walk reading every row of the list once.
const WALKS: f64 = 3.0;

/// A row of the Q8_0 `output`: 2,560 code bytes and 160 scale bytes.
const ROW_BYTES: f64 = 2_720.0;

/// PIN(2026-10-11): the head's read rate on the A6000, the top of its measured 579-701 GB/s, which saves the fewest
/// ms a window and so gives the lowest break-even.
const HEAD_GB_PER_S: f64 = 701.0;

/// PIN(2026-10-11): the full head's window at P 512 on the A6000, ms.
const WINDOW_MS: f64 = 37.43;

/// PIN(2026-10-11): the windows of docs/cards/q38head2-ab.card at P 512 by draft positions kept, 0 to 3 (26 windows).
const KEPT: [f64; 4] = [0.0, 3.0, 2.0, 21.0];

/// The miss rate q at which a list of `rows` of a `vocab`-token head stops beating the full head: the accepted
/// positions a window, AL(q) = 1 + sum_j A_j (1 - q)^j with A_j the share of windows keeping j or more draft
/// positions (KEPT), fall by the fraction of the window the list saves, AL(q) / AL(0) = 1 - S / T, with S the ms the
/// unread rows save (WALKS x the rows not in the list x ROW_BYTES at HEAD_GB_PER_S) and T the window [derived].
#[must_use]
pub fn break_even(rows: usize, vocab: usize) -> f64 {
    let saved_ms = WALKS * (vocab - rows) as f64 * ROW_BYTES / (HEAD_GB_PER_S * 1e9) * 1e3;
    let windows: f64 = KEPT.iter().sum();
    let al = |q: f64| {
        1.0 + (1..KEPT.len())
            .map(|j| KEPT[j..].iter().sum::<f64>() / windows * (1.0 - q).powi(j as i32))
            .sum::<f64>()
    };
    let target = al(0.0) * (1.0 - saved_ms / WINDOW_MS);
    let (mut lo, mut hi) = (0.0, 1.0);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if al(mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

#[cfg(test)]
mod tests {
    use super::{Arc, HEAD_ROWS_FORMAT, HeadRowsError, HeadSet, hex, parse_head_rows};

    const DIGEST: [u8; 32] = [0xab; 32];

    fn rows_file(vocab: u32, ids: &[u32], said: usize) -> String {
        let mut s = format!(
            "{HEAD_ROWS_FORMAT} vocab={vocab} rows={said} vocab_sha256={}\n",
            hex(&DIGEST)
        );
        for id in ids {
            s.push_str(&format!("{id}\n"));
        }
        s
    }

    #[test]
    fn a_row_list_reads() {
        let got = parse_head_rows("t", &rows_file(10, &[0, 3, 9], 3), 10, &DIGEST);
        assert_eq!(
            got,
            Ok(HeadSet::List {
                ids: Arc::from(vec![0, 3, 9]),
                digest: DIGEST
            })
        );
    }

    /// A refusal case: what it breaks, the file, the refusal it wants.
    type Case = (&'static str, String, fn(&HeadRowsError) -> bool);

    /// Every refusal, each by its name.
    #[test]
    fn a_row_list_is_refused_by_name() {
        let digest_line = format!("vocab_sha256={}", hex(&DIGEST));
        let cases: Vec<Case> = vec![
            ("no header", "0\n1\n".to_string(), |e| {
                matches!(e, HeadRowsError::Header { .. })
            }),
            ("empty", String::new(), |e| {
                matches!(e, HeadRowsError::Header { .. })
            }),
            (
                "digest absent",
                format!("{HEAD_ROWS_FORMAT} vocab=10 rows=1\n0\n"),
                |e| {
                    matches!(
                        e,
                        HeadRowsError::Field {
                            field: "vocab_sha256",
                            ..
                        }
                    )
                },
            ),
            (
                "field twice",
                format!("{HEAD_ROWS_FORMAT} vocab=10 vocab=10 rows=1 {digest_line}\n0\n"),
                |e| matches!(e, HeadRowsError::Field { field: "vocab", .. }),
            ),
            (
                "unknown field",
                format!("{HEAD_ROWS_FORMAT} vocab=10 rows=1 size=3 {digest_line}\n0\n"),
                |e| matches!(e, HeadRowsError::Header { .. }),
            ),
            (
                "short digest",
                format!("{HEAD_ROWS_FORMAT} vocab=10 rows=1 vocab_sha256=abcd\n0\n"),
                |e| {
                    matches!(
                        e,
                        HeadRowsError::Field {
                            field: "vocab_sha256",
                            ..
                        }
                    )
                },
            ),
            (
                "other digest",
                format!(
                    "{HEAD_ROWS_FORMAT} vocab=10 rows=1 vocab_sha256={}\n0\n",
                    hex(&[0xcd; 32])
                ),
                |e| matches!(e, HeadRowsError::Digest { .. }),
            ),
            ("other vocab", rows_file(11, &[0], 1), |e| {
                matches!(
                    e,
                    HeadRowsError::Vocab {
                        file: 11,
                        target: 10,
                        ..
                    }
                )
            }),
            ("not an id", rows_file(10, &[0], 1) + "x\n", |e| {
                matches!(e, HeadRowsError::NotAnId { line: 3, .. })
            }),
            ("blank line", rows_file(10, &[0], 1) + "\n", |e| {
                matches!(e, HeadRowsError::NotAnId { line: 3, .. })
            }),
            ("signed", rows_file(10, &[], 0) + "+1\n", |e| {
                matches!(e, HeadRowsError::NotAnId { line: 2, .. })
            }),
            ("past the vocabulary", rows_file(10, &[0, 10], 2), |e| {
                matches!(
                    e,
                    HeadRowsError::PastVocab {
                        line: 3,
                        id: 10,
                        ..
                    }
                )
            }),
            ("duplicate", rows_file(10, &[0, 4, 4], 3), |e| {
                matches!(e, HeadRowsError::Duplicate { line: 4, id: 4, .. })
            }),
            ("unsorted", rows_file(10, &[0, 5, 4], 3), |e| {
                matches!(
                    e,
                    HeadRowsError::Unsorted {
                        line: 4,
                        id: 4,
                        prev: 5,
                        ..
                    }
                )
            }),
            ("count", rows_file(10, &[0, 1], 3), |e| {
                matches!(
                    e,
                    HeadRowsError::Count {
                        said: 3,
                        listed: 2,
                        ..
                    }
                )
            }),
            ("no rows", rows_file(10, &[], 0), |e| {
                matches!(e, HeadRowsError::Size { rows: 0, .. })
            }),
            (
                "the whole vocabulary",
                rows_file(4, &[0, 1, 2, 3], 4),
                |e| {
                    matches!(
                        e,
                        HeadRowsError::Size {
                            rows: 4,
                            max: 3,
                            ..
                        }
                    )
                },
            ),
        ];
        for (what, text, want) in cases {
            let vocab = if what == "the whole vocabulary" {
                4
            } else {
                10
            };
            let got = parse_head_rows("t", &text, vocab, &DIGEST);
            match got {
                Err(e) if want(&e) => {}
                other => panic!("{what}: {other:?}"),
            }
        }
    }
}
