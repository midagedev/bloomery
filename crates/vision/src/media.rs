//! What a model tells the server about its image input, and the one thing it does with an image.
//!
//! The server's media layer (`serve::media`) is the same for every model: it reads a request's
//! image parts, checks and decodes the files, keys them, and expands each image's placeholder token
//! into the image's span. A model contributes data — the placeholder its chat template carries for
//! one image, the token that placeholder is, the separator between flattened content parts — and
//! one prepare, which turns a decoded image into the encoder's input on the request thread. A model
//! implements [`MediaModel`] in its own module (`arch::<projector type>::media`); the common
//! code holds no branch per model.

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

    /// The id `serve::media::expand_spans` finds in the tokenized prompt and expands to one copy
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
