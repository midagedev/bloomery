//! What a model tells the server about its image input, and the one thing it does with an image.
//!
//! The server's media layer (`serve::media`) is the same for every model: it reads a request's
//! image parts, checks and decodes the files, keys them, and expands each image's placeholder token
//! into the image's span. A model contributes data — the placeholder its chat template carries for
//! one image, the token that placeholder is, the separator between flattened content parts — and
//! one prepare, which turns a decoded image into the encoder's input on the request thread. A model
//! implements [`MediaModel`] in its own module (`arch::deepseek41v::media` for V4.1); the common
//! code holds no branch per model.

use crate::{Patches, Rgb8, VisionError};

/// One image made ready for a model: the positions it takes in the prompt and the encoder's input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prepared {
    /// The image's span length in the prompt, delimiters included: its placeholder token is
    /// expanded to this many positions.
    pub span_len: usize,
    /// The encoder's input, with the plan it was cut to.
    pub patches: Patches,
}

/// A model's image input, as data plus one prepare.
pub trait MediaModel {
    /// The placeholder text the chat template carries for one image.
    fn image_placeholder(&self) -> &str;

    /// The token id that placeholder tokenizes to; every position of an image's span carries it.
    fn image_token(&self) -> u32;

    /// The separator between a message's content parts when they are flattened into one text.
    fn part_separator(&self) -> &str {
        "\n"
    }

    /// Resize and patch one decoded image: its span length and the encoder's input.
    fn prepare(&self, image: &Rgb8) -> Result<Prepared, VisionError>;
}
