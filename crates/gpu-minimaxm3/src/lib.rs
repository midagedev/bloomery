//! MiniMax-M3 on the card. Every layer is grouped-query attention (64 query heads over 4 key heads of
//! 128 values, a per-head q/k norm and a partial NEOX rope); a sparse layer attends the blocks its indexer picks.
//! The leading layers are dense blocks, the rest a routed block with one shared expert, every block SwiGLU-OAI.
//!
//! The program has no device code of its own beyond [`attn`]'s two selection kernels: the head kernel, the flash
//! over the picked blocks, the typed sites, the gated block and the routed block with its shared expert, the router
//! `sigmoid_128_4`, the tier shell and the host leg are the common owners'. What only MiniMax's architecture forces
//! stays here: the sparse attention ([`attn`]), the layer's walk over its two attention kinds and two feed-forward
//! kinds ([`step`]), and the stores (K/V planes, and an index key a position on a sparse layer, [`body`]).
//! Several sequences park those planes ([`bloomery_gpu::Slots`]); the step stays one token of the live sequence.
//!
//! - [`attn`]: the indexer's head kernel and key cache, the block-max selector and the flash over the picks;
//! - [`body`]: the load, the stores, the buffers, [`body::Body`];
//! - [`step`]: the one-token step's walk;
//! - [`ffn`]: the routed block with its shared expert around the host leg, and the dense block;
//! - [`seq`]: the sequence's rows on the host, for a server's prompt cache;
//! - [`stream`]: the family's facts for the common walk's host streaming (`BLOOMERY_XSTREAM`);
//! - [`tier`]: the expert tier card's node family over the common card leg.

pub mod attn;
pub mod body;
pub mod ffn;
pub mod forced;
pub mod prefill;
pub mod seq;
pub mod step;
pub mod stream;
pub mod tier;

pub use body::{Body, MiniMaxM3Model, StepInput, set_taps};
pub use forced::{ForcedRoute, ForcedRow};
pub use seq::MiniMaxSeq;
