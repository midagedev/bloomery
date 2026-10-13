//! MiniCPM-V 4.7's image input as the server's media layer sees it, and the preprocessing of an
//! image ([`preprocess`]): the group of tiles.
//!
//! The placeholder of the prompt's image `i` is `<image_id>i</image_id><image>` and the pad token
//! whose copies the group's rows replace.

use std::borrow::Cow;

use super::Hparams;
use super::group::{Group, Marks, preprocess_group};
use crate::media::{EncoderInput, MediaModel, PartJoin, Prepared};
use crate::{Rgb8, VisionError};

/// An image to its group of tiles (see [`super::group`]).
pub fn preprocess(image: &Rgb8, hp: &Hparams) -> Result<Group, VisionError> {
    preprocess_group(image, hp)
}

/// MiniCPM-V 4.7's [`MediaModel`]: the file's hyperparameters, the ids of the marks and the pad.
#[derive(Clone, Debug, PartialEq)]
pub struct Media {
    hp: Hparams,
    marks: Marks,
    pad: u32,
    marker: String,
}

impl Media {
    /// The media description of an encoder file: `hp` from [`Hparams::read`], `marks` from the
    /// text model's tokenizer ([`Marks::lookup`]) and the pad token: `pad_text` is the text that
    /// tokenizes to the single id `pad` ([`MediaModel::image_token`]), which the group's rows are
    /// expanded to.
    #[must_use]
    pub fn new(hp: Hparams, marks: Marks, pad_text: &str, pad: u32) -> Media {
        Media {
            marker: format!("<image>{pad_text}"),
            hp,
            marks,
            pad,
        }
    }

    /// The marks, for the prompt's head and rows ([`Marks::head`], [`Marks::rows`]).
    #[must_use]
    pub fn marks(&self) -> &Marks {
        &self.marks
    }

    /// One image to its group: the tower's input per tile and the layout.
    pub fn group(&self, image: &Rgb8) -> Result<Group, VisionError> {
        preprocess_group(image, &self.hp)
    }
}

impl MediaModel for Media {
    fn image_placeholder(&self, index: usize) -> Cow<'_, str> {
        Cow::Owned(format!("<image_id>{index}</image_id>{}", self.marker))
    }

    fn image_marker(&self) -> &str {
        &self.marker
    }

    fn image_token(&self) -> u32 {
        self.pad
    }

    fn part_join(&self) -> PartJoin {
        PartJoin::LlamaServer
    }

    fn image_head(&self, index: usize) -> Option<Vec<u32>> {
        Some(self.marks.head(index))
    }

    /// The image as its group of tiles; the span is the group's rows, the tiles' tokens and the
    /// separators between them.
    fn prepare(&self, image: &Rgb8) -> Result<Prepared, VisionError> {
        let group = self.group(image)?;
        Ok(Prepared {
            span_len: group.rows(),
            input: EncoderInput::Tiles(group),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Media;
    use crate::Rgb8;
    use crate::arch::minicpmv::group::Marks;
    use crate::arch::minicpmv::hparams::tests::hp;
    use crate::media::{EncoderInput, MediaModel, PartJoin, expand_spans};

    fn media() -> Media {
        let marks = Marks::lookup(|s| Some(1000 + s.len() as u32)).expect("marks");
        Media::new(hp(), marks, "<unk>", 248_077)
    }

    /// The data a server reads: the numbered placeholder, its marker, the token it expands, the
    /// join rule and the ids that precede the token.
    #[test]
    fn the_media_data_a_server_reads() {
        let m = media();
        assert_eq!(m.image_placeholder(0), "<image_id>0</image_id><image><unk>");
        assert_eq!(
            m.image_placeholder(12),
            "<image_id>12</image_id><image><unk>"
        );
        assert_eq!(m.image_marker(), "<image><unk>");
        assert_eq!(m.image_token(), 248_077);
        assert_eq!(m.part_join(), PartJoin::LlamaServer);
        // `<image_id>`, the digits, `</image_id>`, `<image>`.
        let head = m.image_head(12).expect("a numbered model has a head");
        assert_eq!(head, m.marks().head(12));
        assert_eq!(head.len(), 5);
    }

    /// `prepare` gives the group's rows, the tiles' tokens and the separators, as its span, and
    /// the group itself as the encoder's input. 448×448 and 200×200 are one overview of 8×8
    /// tokens and its `</image>` (64 + 1); 1000×700 is an overview and four slices of 10×7
    /// tokens, 5 × 70 = 350 tokens and 2 + 2 + 3 + 2 + 1 = 10 separators. A prompt's placeholder
    /// expands to that many positions.
    #[test]
    fn prepare_gives_the_groups_rows_as_the_span() {
        let m = media();
        for ((w, h), span, tiles) in [
            ((448, 448), 65, 1),
            ((1000, 700), 360, 5),
            ((200, 200), 65, 1),
        ] {
            let p = m.prepare(&Rgb8::filled(w, h, [1, 2, 3])).expect("prepare");
            assert_eq!(p.span_len, span, "{w}x{h}");
            let EncoderInput::Tiles(group) = &p.input else {
                panic!("{w}x{h}: a MiniCPM-V image is a group of tiles");
            };
            assert_eq!((group.tiles.len(), group.rows()), (tiles, span), "{w}x{h}");
            let ids =
                expand_spans(&[5, 248_077, 6], m.image_token(), &[p.span_len]).expect("expand");
            // One span, the range itself. `[1..1 + span]` is a one-element array of a
            // range, which is the lint's single-range init, not the positions inside it.
            assert_eq!(ids.spans.len(), 1, "{w}x{h}");
            assert_eq!(ids.spans[0], 1..1 + span, "{w}x{h}");
        }
    }
}
