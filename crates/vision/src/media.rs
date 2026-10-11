//! What a model tells the server about its image input, and the one thing it does with an image.
//!
//! The server's media layer (`serve::media`) is the same for every model: it reads a request's
//! image parts, checks and decodes the files, keys them, and expands each image's placeholder token
//! into the image's span. A model contributes data — the placeholder its chat template carries for
//! one image, the token that placeholder is, the separator between flattened content parts — and
//! one prepare, which turns a decoded image into the encoder's input on the request thread. A model
//! implements [`MediaModel`] in its own module (`arch::<projector type>::media`); the common
//! code holds no branch per model. The expansion of a tokenized prompt's placeholders into spans
//! ([`expand_spans`]) is common too: the server's chat path and Clef's decision encoder share it.

use std::ops::Range;

use crate::{Patches, Rgb8, VisionError};

/// How a message's content parts are flattened into one text before the chat template sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartJoin {
    /// The same separator between every two parts, an image's placeholder included.
    Uniform(&'static str),
    /// llama-server's rule (`common_chat_msg::to_json_oaicompat` in `common/chat.cpp`): a `\n`
    /// between two text parts, and none before or after an image's placeholder. A text part is
    /// preceded by `\n` unless no text has been written yet or the part before it is a
    /// placeholder; a placeholder is never preceded by one.
    LlamaServer,
}

/// One image made ready for a model: the positions it takes in the prompt and the encoder's input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prepared {
    /// The positions the image's placeholder token is expanded to: the image's rows, and the
    /// delimiter positions too when the model's span carries them as positions of the same id.
    pub span_len: usize,
    /// The encoder's input, with the plan it was cut to.
    pub patches: Patches,
}

/// A model's image input, as data plus one prepare.
pub trait MediaModel {
    /// The placeholder text the chat template carries for one image.
    fn image_placeholder(&self) -> &str;

    /// The id [`expand_spans`] finds in the tokenized prompt and expands to one copy
    /// per position of the image ([`Prepared::span_len`]): the one id V4.1's placeholder
    /// tokenizes to, and the `<|image_pad|>` between the two delimiter ids Clef's tokenizes to.
    fn image_token(&self) -> u32;

    /// The separator between a message's content parts when they are flattened into one text:
    /// the separator of a [`PartJoin::Uniform`] join.
    fn part_separator(&self) -> &str {
        "\n"
    }

    /// How a message's content parts are flattened into one text. A model whose parts do not all
    /// join alike says so here.
    fn part_join(&self) -> PartJoin {
        PartJoin::Uniform("\n")
    }

    /// Resize and patch one decoded image: its span length and the encoder's input.
    fn prepare(&self, image: &Rgb8) -> Result<Prepared, VisionError>;
}

/// A prompt whose image placeholders are expanded to their images' spans.
#[derive(Debug, PartialEq, Eq)]
pub struct Expanded {
    pub ids: Vec<u32>,
    /// Where each image's span sits in `ids`, in the order of the images.
    pub spans: Vec<Range<usize>>,
}

/// Why [`expand_spans`] refused a prompt.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum ExpandError {
    /// The prompt's image placeholders and the images do not pair up.
    #[error("{placeholders} image placeholder(s) in the prompt for {images} image(s)")]
    PlaceholderCount { placeholders: usize, images: usize },
    /// An image whose span is empty (the model's fault, not the request's).
    #[error("image {0} has an empty span")]
    EmptySpan(usize),
}

/// Replace the i-th `image_token` of `ids` by `span_lens[i]` copies of it. The placeholders are
/// counted over the whole of `ids`, so one typed into the text, which pairs with no image, is
/// refused by name.
pub fn expand_spans(
    ids: &[u32],
    image_token: u32,
    span_lens: &[usize],
) -> Result<Expanded, ExpandError> {
    let placeholders = ids.iter().filter(|&&t| t == image_token).count();
    if placeholders != span_lens.len() {
        return Err(ExpandError::PlaceholderCount {
            placeholders,
            images: span_lens.len(),
        });
    }
    if let Some(i) = span_lens.iter().position(|&n| n == 0) {
        return Err(ExpandError::EmptySpan(i));
    }
    let mut out = Vec::with_capacity(ids.len() - placeholders + span_lens.iter().sum::<usize>());
    let mut spans = Vec::with_capacity(placeholders);
    let mut lens = span_lens.iter();
    for &t in ids {
        if t == image_token {
            let n = *lens
                .next()
                .expect("one length per placeholder, counted above");
            spans.push(out.len()..out.len() + n);
            out.extend(std::iter::repeat_n(t, n));
        } else {
            out.push(t);
        }
    }
    Ok(Expanded { ids: out, spans })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_expand_in_order() {
        let got = expand_spans(&[1, 7, 2, 7, 3], 7, &[3, 1]).unwrap();
        assert_eq!(
            got,
            Expanded {
                ids: vec![1, 7, 7, 7, 2, 7, 3],
                spans: vec![1..4, 5..6],
            }
        );
        let none = expand_spans(&[1, 2], 7, &[]).unwrap();
        assert_eq!((none.ids, none.spans.len()), (vec![1, 2], 0));
        let first = expand_spans(&[7], 7, &[2]).unwrap();
        assert_eq!(first.ids, [7, 7]);
        assert_eq!((first.spans.len(), first.spans.first()), (1, Some(&(0..2))));
    }

    #[test]
    fn spans_that_do_not_pair_up_are_named() {
        for (ids, lens, want) in [
            (&[1, 7, 7][..], &[2][..], (2, 1)),
            (&[1, 7][..], &[2, 2][..], (1, 2)),
            (&[1, 2][..], &[2][..], (0, 1)),
        ] {
            assert_eq!(
                expand_spans(ids, 7, lens).unwrap_err(),
                ExpandError::PlaceholderCount {
                    placeholders: want.0,
                    images: want.1
                },
                "{ids:?} {lens:?}"
            );
        }
        assert_eq!(
            expand_spans(&[7, 7], 7, &[1, 0]).unwrap_err(),
            ExpandError::EmptySpan(1)
        );
    }
}
