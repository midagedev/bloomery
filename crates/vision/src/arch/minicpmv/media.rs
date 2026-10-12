//! MiniCPM-V 4.7's image input as the server's media layer sees it, and the preprocessing of an
//! image ([`preprocess`]): the group of tiles.
//!
//! The placeholder is `<image_id>0</image_id><image>` and the pad token whose copies the group's
//! rows replace; the image index is 0 in the placeholder text, as one image prompt has it.

use super::Hparams;
use super::group::{Group, Marks, preprocess_group};
use crate::media::{MediaModel, PartJoin, Prepared};
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
    placeholder: String,
}

impl Media {
    /// The media description of an encoder file: `hp` from [`Hparams::read`], `marks` from the
    /// text model's tokenizer ([`Marks::lookup`]) and the pad token: `pad_text` is the text that
    /// tokenizes to the single id `pad` ([`MediaModel::image_token`]), which the group's rows are
    /// expanded to.
    #[must_use]
    pub fn new(hp: Hparams, marks: Marks, pad_text: &str, pad: u32) -> Media {
        Media {
            placeholder: format!("<image_id>0</image_id><image>{pad_text}"),
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
    fn image_placeholder(&self) -> &str {
        &self.placeholder
    }

    fn image_token(&self) -> u32 {
        self.pad
    }

    fn part_join(&self) -> PartJoin {
        PartJoin::LlamaServer
    }

    /// A group is several tiles of several sizes, and [`Prepared`] holds one patch tensor: this
    /// model's image is prepared by [`Media::group`], whose `rows()` is the span length.
    fn prepare(&self, image: &Rgb8) -> Result<Prepared, VisionError> {
        Err(VisionError::Size {
            width: image.width,
            height: image.height,
            detail: "a MiniCPM-V 4.7 image is a group of tiles, not one patch tensor; \
                     prepare it with minicpmv::Media::group"
                .into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Media;
    use crate::Rgb8;
    use crate::arch::minicpmv::group::Marks;
    use crate::arch::minicpmv::hparams::tests::hp;
    use crate::media::{MediaModel, PartJoin, expand_spans};

    fn media() -> Media {
        let marks = Marks::lookup(|s| Some(1000 + s.len() as u32)).expect("marks");
        Media::new(hp(), marks, "<unk>", 248_077)
    }

    /// The data a server reads, and the one thing `prepare` refuses by name; the span a prompt's
    /// placeholder expands to is the group's rows.
    #[test]
    fn the_media_data_and_the_group_span() {
        let m = media();
        assert_eq!(m.image_placeholder(), "<image_id>0</image_id><image><unk>");
        assert_eq!(m.image_token(), 248_077);
        assert_eq!(m.part_join(), PartJoin::LlamaServer);
        let image = Rgb8::filled(1000, 700, [1, 2, 3]);
        assert!(
            m.prepare(&image)
                .unwrap_err()
                .to_string()
                .contains("Media::group")
        );
        let rows = m.group(&image).expect("group").rows();
        assert_eq!(rows, 360);
        let ids = expand_spans(&[5, 248_077, 6], m.image_token(), &[rows]).expect("expand");
        assert_eq!(ids.spans.len(), 1);
        assert_eq!(ids.spans[0], 1..361);
    }
}
