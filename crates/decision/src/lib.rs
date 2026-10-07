//! A decision model's request and head: a SystemOne request to token ids and spans, the backbone's
//! hidden states through the joint schema head to per-option logits, and the logits to the SystemOne
//! response body. One decision model runs today, Clef (Cloudflare's), so the head and the prompt are
//! Clef's, named so; the crate is pure host code.
//!
//! [`json`] reads a request as Python does, [`render`] writes JSON as Python does, [`request`]
//! validates, [`encode`] builds the ids, [`safetensors`] and [`head`] load and run the head,
//! [`gguf_head`] reads the head from a model file that carries it (llama.cpp's Clef layout),
//! [`rows`] gives the head its output embedding rows from a GGUF file, [`answer`] builds the body;
//! [`release`] holds what a server needs to seat Clef.

pub mod answer;
pub mod encode;
pub mod gguf_head;
pub mod head;
pub mod json;
mod ops;
mod pool;
pub mod release;
pub mod render;
pub mod request;
pub mod rows;
pub mod safetensors;

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
    #[error("{0}: {1}")]
    Io(String, std::io::Error),
    #[error("safetensors: {0}")]
    Safetensors(String),
    #[error("safetensors: {name} is {dtype}; this reader converts BF16, F16 and F32")]
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
