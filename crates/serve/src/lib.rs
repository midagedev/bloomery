//! bloomery-serve: a llama-server-compatible HTTP API over an [`Engine`].
//!
//! Endpoints: `POST /v1/chat/completions`, `POST /v1/messages` and
//! `POST /v1/messages/count_tokens` (Anthropic's Messages API on the chat
//! path), `POST /completion`, `POST /v1/completions` (OpenAI's text
//! completion shape on the completion path), `POST /chat/completions/input_tokens`
//! and `POST /v1/chat/completions/input_tokens` (the chat prompt's id count),
//! `POST /tokenize`,
//! `POST /detokenize`, `POST /apply-template`, `POST /v1/embeddings`, `POST /embeddings` and
//! `POST /embedding` (an [`embed::Embed`] beside the engine), `GET /v1/models`,
//! `GET /v1/models/{model}`, Ollama's `GET /api/tags` and `POST /api/show`, `GET /health`,
//! `GET /props`, `GET /slots`, `POST /slots/{id}?action=save|restore|erase`,
//! `GET /metrics`, `POST /residency/reset` and `POST /shutdown` (bloomery's
//! own; the shutdown is the orderly stop, accepted from a loopback peer
//! only). JSON field names, defaults and stream framing are
//! llama-server's; see `api` for the slots, `sched` for who gets one and
//! `worker` for the engine thread that steps them, `swap` for slots that take one engine in turns. [`decide`] is a decision model's server
//! (`POST /v1/systemone`) on the same HTTP layer. [`preflight`] is what a
//! server checks before it fetches or loads (the CPU, the driver), [`flag`]
//! the seats' shared flag spellings, [`screen`] the start screen a person
//! reads at a terminal. For the agent that starts the server: [`Listener`]
//! binds the socket before the model loads and answers 503 `loading` until
//! the server runs, [`event`] prints the start's events as JSON lines on
//! stdout, and [`exit`] is the one table of exit statuses.

mod admit;
mod api;
pub use api::embeddings;
pub use api::listener::{self, Listener};
pub mod callstream;
mod constrain;
pub mod decide;
pub mod dsml;
pub mod embed;
pub mod embed_cache;
pub mod engine;
pub mod event;
pub mod exit;
pub mod flag;
mod genloop;
pub mod glmxml;
mod guard;
pub mod hermes;
mod http;
pub mod live;
pub mod media;
pub mod mock;
mod models;
pub mod museatem;
pub mod preflight;
mod promptcache;
pub mod qwenxml;
pub mod reasoning;
pub mod sampling;
mod sched;
pub mod screen;
mod slotfile;
mod stop;
mod swap;
pub mod v41;
pub mod video;
pub mod vocab;
mod worker;

pub use api::{
    COMMIT, ChatPreparation, ChatRequest, EngineFailure, FATAL_LINGER, KEEP_ALIVE_IDLE, Keys,
    MAX_CONNECTIONS, ServeError, Server, ServerConfig, VERSION, panic_text,
};
pub use engine::{
    Ahead, CacheNote, Decoder, DeviceProps, DraftProps, Drafted, Engine, EngineError, EngineProps,
    Flight, Landed, ModelProps, Pending, PlacementProps, ResidencyReset, Sampler, SamplerFactory,
    SamplerRefused, SamplingParams, SaveInFlight, Saved, SavedState, SharedPrefix, SlotDraw,
    SlotPass, SlotRow, StateError, Tokenizer, Waits,
};
pub use jinja::{ChatTemplate, TemplateError};
pub use mock::{DraftMock, MockEngine, MockTokenizer, SaveStats, ScriptedEngine};
pub use sched::{FifoPicker, SlotConfig, SlotPicker, SlotSummary, WaitingRequest};
pub use swap::{Park, QUANTUM, SwapEngine};
