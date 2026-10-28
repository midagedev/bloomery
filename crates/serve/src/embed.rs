//! An embedding model the server holds beside its engine ([`crate::Server::with_side`]): the one
//! seam between the embeddings endpoints and the model.
//!
//! The server reads a request as llama-server's embeddings handler does and tokenizes its text
//! with the model's own vocabulary ([`Embed::tokenizer`]); the model runs one pass over one input's
//! ids, including its pooling and learned projections ([`Embed::embed`]), or one call over several
//! ([`Embed::embed_batch`]). The cut to a
//! `dimensions` width, the normalization
//! ([`normalize`], llama.cpp's `common_embd_normalize`) and the answer's shape belong to `bloomery-embed`.

use std::sync::Arc;

use crate::engine::Tokenizer;

/// The model failed on a valid input (a 500), which ends the server.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct EmbedError(pub String);

/// Prepared audio chunks and their first position in the embedding input.
#[derive(Clone)]
pub struct AudioFeed {
    pub at: usize,
    pub chunks: Arc<Vec<vision::media::AudioChunk>>,
}

/// An embedding model: one input's ids in, its pooled vector out. The server calls it one
/// request at a time, behind a mutex, and reads the facts below once, when it takes the model.
pub trait Embed: Send {
    /// Optional pooled-vector and tower-row cache, shared with the server.
    fn cache(&self) -> Option<crate::embed_cache::Shared> {
        None
    }
    /// The model's vocabulary, which the server tokenizes the inputs' text with.
    fn tokenizer(&self) -> Arc<dyn Tokenizer>;
    /// The model's name: its `/v1/models` id and the `model` every OpenAI answer carries.
    fn name(&self) -> String;
    /// The model file, reported as `/props`' `embedding_model.path`.
    fn path(&self) -> String;
    /// The most positions one input takes.
    fn ctx(&self) -> usize;
    /// The widths a request's `dimensions` may cut the pooled vector to, ascending; the last is
    /// the pooled vector's own width.
    fn dims(&self) -> Vec<usize>;
    /// Named input prefixes from the model's sentence-transformers configuration.
    fn prompts(&self) -> std::collections::BTreeMap<String, String> {
        std::collections::BTreeMap::default()
    }
    /// The side model's image preparer, present when a projector is loaded.
    fn media(&self) -> Option<crate::media::SharedMediaModel> {
        None
    }
    /// The seated projector's audio input rule, if it accepts decoded mono PCM.
    fn audio(&self) -> Option<crate::media::SharedMediaModel> {
        None
    }
    /// One pass over one input: at least one id and at most [`Embed::ctx`], each below the
    /// vocabulary's size. The pooled vector, the full width (the last of [`Embed::dims`]).
    fn embed(&mut self, ids: &[u32]) -> Result<Vec<f32>, EmbedError>;
    /// Several inputs, one pooled vector each, in the inputs' order. The default is one
    /// [`Embed::embed`] per input. A model may pack them into fewer passes.
    fn embed_batch(&mut self, inputs: &[&[u32]]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let mut out = Vec::with_capacity(inputs.len());
        for ids in inputs {
            out.push(self.embed(ids)?);
        }
        Ok(out)
    }
    /// One pass with prepared images and audio chunks replacing the positions they occupy.
    fn embed_media(
        &mut self,
        _ids: &[u32],
        _images: &[crate::media::MediaFeed],
        _audio: &[AudioFeed],
    ) -> Result<Vec<f32>, EmbedError> {
        Err(EmbedError(
            "the embedding model has no media encoder (--embd-mmproj)".into(),
        ))
    }
}

pub use ::embed::{Norm, normalize};
