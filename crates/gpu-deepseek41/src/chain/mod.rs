//! The V4.1 decode step as three pieces: [`attn`], the attention sub-layer
//! with the compressor, the index keys and the indexer on the layers that
//! have them; [`ffn`], the MoE sub-layer and its join with the host tier;
//! [`glue`], what sits outside the sub-layers — the embedding broadcast, the
//! engram, the hyper-connection collapse and the head.
//!
//! A piece is built once at load: every name, view, launch configuration and
//! scratch buffer its launches need is resolved then. Its enqueue only
//! launches — no allocation, no synchronization, no host copy — so the step's
//! pieces record into one captured graph. A piece takes the buffers it
//! shares with the rest of the step as arguments — the hyper-connection
//! streams it reads and the ones it writes, its folded input and the fold it
//! leaves for the next sub-layer, the layer's cache, the step image's device
//! copy — and the resident weights as a [`bloomery_gpu::weights::Weights`];
//! never the [`crate::body::Body`] that holds them. A piece's gate runs a
//! layer against buffers it allocated itself; the step's assembly passes the
//! body's.

pub mod attn;
pub mod ffn;
pub mod glue;
