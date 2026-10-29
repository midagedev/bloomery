//! SentencePiece's highest-score adjacent-pair merges over raw UTF-8 bytes.

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::Error;
use crate::merge::{self, NONE};
use crate::spelling::{Spelling, byte_tokens};

/// Lower heap priority means a higher token score.
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct Score(f32);

impl Eq for Score {}

impl Ord for Score {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.total_cmp(&self.0)
    }
}

impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub(crate) type Scratch = merge::Scratch<Score>;

#[derive(Clone)]
pub(crate) struct Spm {
    scores: Vec<f32>,
    bytes: Box<[u32; 256]>,
}

impl Spm {
    pub(crate) fn new(
        scores: Vec<f32>,
        tokens: &HashMap<String, u32>,
        attrs: &[u32],
    ) -> Result<Self, Error> {
        use crate::vocab::attr;

        Ok(Self {
            scores,
            bytes: byte_tokens(|byte, text| {
                if let Some(&id) = tokens.get(text) {
                    return (attrs[id as usize] & attr::BYTE != 0).then_some(id);
                }
                let literal = [byte];
                let text = std::str::from_utf8(&literal).ok()?;
                let &id = tokens.get(text)?;
                (attrs[id as usize] & (attr::NORMAL | attr::USER_DEFINED) != 0).then_some(id)
            })?,
        })
    }

    /// Append one escaped fragment's ids, preserving invalid UTF-8 as bytes.
    pub(crate) fn word(
        &self,
        word: &[u8],
        tokens: &HashMap<String, u32>,
        scratch: &mut Scratch,
        out: &mut Vec<u32>,
    ) {
        let token = |text: &[u8]| {
            std::str::from_utf8(text)
                .ok()
                .and_then(|text| tokens.get(text))
                .copied()
        };
        scratch.symbols(word, Spelling::Metaspace, |text| {
            token(text).unwrap_or(NONE)
        });
        scratch.run(|left, l, r| {
            let start = left as usize;
            let end = start + l.len as usize + r.len as usize;
            let id = token(&word[start..end])?;
            Some((Score(self.scores[id as usize]), id))
        });
        // Every merged symbol is a token, so only unmerged characters need fallback.
        for (start, sym) in scratch.syms.iter().enumerate() {
            if sym.len == 0 {
                continue;
            }
            if sym.iid != NONE {
                out.push(sym.iid);
            } else {
                out.extend(
                    word[start..start + sym.len as usize]
                        .iter()
                        .map(|&b| self.bytes[usize::from(b)]),
                );
            }
        }
    }
}
