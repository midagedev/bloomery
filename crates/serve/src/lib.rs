//! bloomery-serve: a llama-server-compatible HTTP API over an [`Engine`].
//!
//! Endpoints: `POST /v1/chat/completions`, `POST /completion`, `POST /tokenize`,
//! `POST /detokenize`, `POST /apply-template`, `GET /v1/models`, `GET /health`,
//! `GET /props`, `GET /slots`, `POST /slots/{id}?action=save|restore|erase`,
//! `GET /metrics`, and `POST /residency/reset` (bloomery's own). JSON field names, defaults and stream framing are
//! llama-server's; see `api` for the slots, `sched` for who gets one and
//! `worker` for the engine thread that steps them. [`decide`] is a decision model's server
//! (`POST /v1/systemone`) on the same HTTP layer.

mod api;
pub mod decide;
pub mod dsml;
pub mod engine;
mod genloop;
pub mod glmxml;
mod http;
pub mod mock;
mod promptcache;
pub mod reasoning;
pub mod sampling;
mod sched;
mod slotfile;
mod stop;
pub mod template;
mod worker;

pub use api::{
    EngineFailure, FATAL_LINGER, KEEP_ALIVE_IDLE, MAX_CONNECTIONS, ServeError, Server,
    ServerConfig, VERSION,
};
pub use engine::{
    CacheNote, Decoder, DeviceProps, DraftProps, Drafted, Engine, EngineError, EngineProps,
    ModelProps, PlacementProps, ResidencyReset, Sampler, SamplerFactory, SamplingParams, Saved,
    SavedState, SlotRow, StateError, Tokenizer,
};
pub use mock::{DraftMock, MockEngine, MockTokenizer, ScriptedEngine};
pub use sched::{FifoPicker, SlotConfig, SlotPicker, SlotSummary, WaitingRequest};
pub use template::{ChatTemplate, TemplateError};
