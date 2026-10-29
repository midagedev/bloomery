//! Vocabulary-selected encoding, after the shared special-token partition.

use std::collections::HashMap;

use unicode_normalization_alignments::UnicodeNormalization;

use crate::spelling::{Spelling, escape_whitespace};
use crate::vocab::{Encoding, Vocab};
use crate::{Error, bpe, pretok, spm, unicode};

#[derive(Clone)]
pub(crate) enum Encoder {
    Bpe {
        pre: pretok::Pre,
        bpe: bpe::Bpe,
        nfc: bool,
    },
    Spm {
        add_space_prefix: bool,
        spm: spm::Spm,
    },
}

#[derive(Default)]
pub(crate) struct Scratch {
    cpts: Vec<u32>,
    word: Vec<u8>,
    escaped: Vec<u8>,
    pre: pretok::Scratch,
    bpe: bpe::Scratch,
    spm: spm::Scratch,
}

impl Encoder {
    pub(crate) fn new(encoding: Encoding, vocab: &Vocab) -> Result<Self, Error> {
        match encoding {
            Encoding::Bpe {
                pre,
                merges,
                nfc,
                hf,
            } => Ok(Self::Bpe {
                pre,
                nfc,
                bpe: bpe::Bpe::new(
                    &vocab.texts,
                    &vocab.attrs,
                    &merges,
                    pre.spelling(),
                    pre.ignore_merges(),
                    hf,
                )?,
            }),
            Encoding::Spm {
                scores,
                add_space_prefix,
            } => Ok(Self::Spm {
                add_space_prefix,
                spm: spm::Spm::new(scores, &vocab.token_to_id, &vocab.attrs)?,
            }),
        }
    }

    pub(crate) fn spelling(&self) -> Spelling {
        match self {
            Self::Bpe { pre, .. } => pre.spelling(),
            Self::Spm { .. } => Spelling::Metaspace,
        }
    }

    pub(crate) fn merges_off_vocab(&self) -> usize {
        match self {
            Self::Bpe { bpe, .. } => bpe.merges_off_vocab,
            Self::Spm { .. } => 0,
        }
    }

    pub(crate) fn text(
        &self,
        text: &[u8],
        tokens: &HashMap<String, u32>,
        scratch: &mut Scratch,
        out: &mut Vec<u32>,
    ) {
        let Scratch {
            cpts,
            word,
            escaped,
            pre,
            bpe,
            spm,
        } = scratch;
        match self {
            Self::Bpe {
                pre: kind,
                bpe: encoder,
                nfc,
            } => {
                // The reference respells a fragment's spaces before it decodes the fragment, so
                // an overlong U+0020 stays a space.
                let text = if kind.spelling() == Spelling::Metaspace {
                    escape_whitespace(text, escaped);
                    escaped.as_slice()
                } else {
                    text
                };
                if kind.strict_utf8() {
                    unicode::decode_strict(text, cpts);
                    if *nfc {
                        unicode::normalize_nfc(cpts);
                    }
                } else if *nfc && let Ok(text) = std::str::from_utf8(text) {
                    cpts.clear();
                    cpts.extend(text.nfc().map(|(c, _)| u32::from(c)));
                } else {
                    unicode::decode_lenient(text, cpts);
                }
                let mut at = 0;
                for &n in pretok::split(*kind, cpts, pre) {
                    word.clear();
                    for &cpt in &cpts[at..at + n] {
                        unicode::encode_cpt(cpt, word);
                    }
                    at += n;
                    encoder.word(word, bpe, out, tokens);
                }
            }
            Self::Spm {
                add_space_prefix,
                spm: encoder,
            } => {
                word.clear();
                if *add_space_prefix {
                    word.push(b' ');
                }
                word.extend_from_slice(text);
                escape_whitespace(word, escaped);
                encoder.word(escaped, tokens, spm, out);
            }
        }
    }
}
