//! bloomery-serve: a llama-server-compatible HTTP API over an [`Engine`].
//!
//! Endpoints: `POST /v1/chat/completions`, `POST /completion`, `POST /tokenize`,
//! `POST /detokenize`, `POST /apply-template`, `GET /v1/models`, `GET /health`,
//! `GET /props`, `GET /slots`, `GET /metrics`. JSON field names, defaults and
//! stream framing are llama-server's; see `api` for the one-slot model.

mod api;
pub mod dsml;
pub mod engine;
mod genloop;
mod http;
pub mod mock;
pub mod reasoning;
pub mod sampling;
mod stop;
pub mod template;

pub use api::{EngineFailure, FATAL_LINGER, ServeError, Server, ServerConfig};
pub use engine::{
    Decoder, DeviceProps, DraftProps, Engine, EngineError, EngineProps, ModelProps, PlacementProps,
    Sampler, SamplerFactory, SamplingParams, Tokenizer,
};
pub use mock::{MockEngine, MockTokenizer, ScriptedEngine};
pub use template::{ChatTemplate, TemplateError};
