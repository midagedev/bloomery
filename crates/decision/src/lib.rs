//! The decision models' requests, heads and answers: a SystemOne request validated, the backbone's
//! hidden states turned into per-option scores, and the scores into the SystemOne response body. Two
//! kinds of model run, and the crate is pure host code. A joint head ([`head`], [`encode`]) is Clef's
//! (Cloudflare's): one prompt holds every question, and a head of its own reads the backbone's
//! hidden states at each option's tokens. A label readout ([`label`]) is lev's, and the family's
//! (nimble, pplx-decider, OpenJev): one prompt a question variant from the model's own `systemone`
//! template, the answer the language-model head's logits at label tokens.
//!
//! [`json`] reads a request as Python does, [`render`] writes JSON as Python does, [`request`]
//! validates (under the model's [`request::Rules`]), [`encode`] builds Clef's ids, [`head`] loads (through
//! `hf::safetensors`) and runs Clef's head, [`gguf_head`] reads it from a model file that carries it
//! (llama.cpp's Clef layout), [`rows`] gives either model its output embedding rows from a GGUF file,
//! [`answer`] builds the shared parts of a body, [`label`] the label models'; [`release`] holds what a
//! server needs to seat each model.

pub mod answer;
pub mod encode;
pub mod gguf_head;
pub mod head;
pub mod json;
pub mod label;
mod ops;
mod pool;
pub mod release;
pub mod render;
pub mod request;
pub mod rows;

/// Why a request, a file or a forward pass was refused.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("JSON at byte {at}: {what}")]
    Json { at: usize, what: String },
    #[error("JSON nests deeper than {0} levels")]
    JsonTooDeep(usize),
    #[error("JSON literal {0} (Python reads it; a request number must be finite)")]
    JsonNonFinite(&'static str),
    #[error("JSON float {0} is past the f64 range (Python reads it as infinity)")]
    JsonFloatRange(String),
    #[error("JSON string holds the lone surrogate \\u{0:04x}")]
    JsonLoneSurrogate(u32),
    #[error("{0}")]
    Request(String),
    /// A request this engine cannot answer (images, videos): llama.cpp's 501 `not_supported_error`.
    #[error("{0}")]
    NotSupported(String),
    #[error("{id}: {what}")]
    Question { id: String, what: String },
    #[error("{question}: {what} span is empty (its mean would be NaN)")]
    EmptySpan {
        question: String,
        what: &'static str,
    },
    #[error("{question}: span {span:?} is outside the {ids} ids")]
    SpanOutside {
        question: String,
        span: (usize, usize),
        ids: usize,
    },
    #[error("{question}: {spans} option spans for {ids} option ids")]
    Options {
        question: String,
        spans: usize,
        ids: usize,
    },
    #[error("schema requires {fixed} tokens before state; maximum is {max}")]
    SchemaTooLong { fixed: usize, max: usize },
    /// The request's own text carries an image placeholder: the prompt holds more of them than the
    /// request has images.
    #[error(
        "the prompt holds {placeholders} image placeholder(s) for {images} image(s): one typed into the text pairs with no image; send an image as `images` or an `image_url` part"
    )]
    PlaceholderInText { placeholders: usize, images: usize },
    /// The images alone, without the schema or the state, pass the encoder's length.
    #[error("{images} image(s) take {media} tokens; maximum is {max}")]
    ImagesTooLong {
        images: usize,
        media: usize,
        max: usize,
    },
    /// The image part of an encode, not the request's fault: token counts that do not pair with the
    /// images, an image with none, or a tokenizer that does not read the placeholder as Clef's.
    #[error("images: {0}")]
    Images(String),
    #[error("{0}: {1}")]
    Io(String, std::io::Error),
    #[error("safetensors: {0}")]
    Safetensors(String),
    #[error("safetensors: {name} is {dtype}, a type this reader does not hold")]
    Dtype { name: String, dtype: String },
    #[error("weight {0} is missing")]
    MissingWeight(String),
    #[error(
        "the head's weights differ from its config: missing {missing:?}, extra {extra:?}, shapes {shapes:?}"
    )]
    Weights {
        missing: Vec<String>,
        extra: Vec<String>,
        shapes: Vec<String>,
    },
    #[error("joint_head_config.json: {0}")]
    HeadConfig(String),
    #[error("the head in the model file: {0}")]
    InFileHead(String),
    /// A label model's file or setup (its template, labels, temperatures).
    #[error("the label model: {0}")]
    Label(String),
    #[error("hidden states hold {values} values: not {ids} rows of {width}")]
    HiddenShape {
        values: usize,
        ids: usize,
        width: usize,
    },
    #[error("output rows: {0}")]
    Rows(String),
    #[error("{question}/{option}: the logit is not finite")]
    NonFinite { question: String, option: String },
    #[error("logits: {0}")]
    Logits(String),
    #[error("the model file has neither output.weight nor token_embd.weight")]
    NoOutputHead,
    #[error("{name}: {what}")]
    OutputHead { name: &'static str, what: String },
}

impl From<hf::safetensors::Error> for Error {
    fn from(e: hf::safetensors::Error) -> Error {
        use hf::safetensors::Error as E;
        match e {
            E::Io(path, e) => Error::Io(path, e),
            E::Format(what) => Error::Safetensors(what),
            E::Dtype { name, dtype } => Error::Dtype { name, dtype },
            E::Kind { name, have, want } => {
                Error::Safetensors(format!("{name} is {have}; this read wants {want}"))
            }
            E::Missing(name) => Error::MissingWeight(name),
        }
    }
}
