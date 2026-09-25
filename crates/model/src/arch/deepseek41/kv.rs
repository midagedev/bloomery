//! The KV bytes a DeepSeek-V4.1 or V4 layer holds at `ctx_max` tokens —
//! placement's accounting (`docs/v41-placement.md` §4), not the cache itself.
//!
//! Every layer keeps a window of `min(ctx_max, W)` latent rows in f16. A layer
//! that carries `attn_compressor_kv` sources a compressed stream (the fork's
//! loader reads the roles off the tensors the same way,
//! `llama.cpp-fork/src/models/deepseek41.cpp:209-247`): at ratio r it adds
//! `⌈ctx_max / r⌉` rows of latent plus index key in f16, and, when r is above
//! 1, a pooling state in f32, twice (values and scores): r rows of the
//! compressor's projection width, or 2r rows when its groups overlap (ik's
//! CSA state, `llama-dsv4.cpp:985-990`); a ratio-1 group is its one row, so
//! nothing pools and no state is kept. The index key is `indexer.attn_k`'s
//! output on a V4.1 source; on a V4 layer with an index-key compressor it is
//! `attention.indexer.key_length` values, and that compressor keeps its own
//! pooling state by the same rule; a V4 layer that attends its stream whole
//! keeps no index key. The layers that read a source's stream hold none of
//! it. The window and every layer's ratio come from [`Hparams`].
//!
//! Beside the cache ([`KvBytes::shadow_bytes`]), every layer keeps a shadow of
//! its window ring in page-locked host memory: `ctx_max` latent rows in f16,
//! one per position, from which a cut of the cache restores the ring rows
//! later positions overwrote. The plan counts it on the host, not on the card.

use gguf::Split;

use super::hparams::{Compressor, Hparams};
use super::names;
use crate::placement::{KvBytes, PlacementError};

const F16_BYTES: u64 = 2;
const F32_BYTES: u64 = 4;

/// A layer's compressed stream.
#[derive(Clone, Copy, Debug)]
struct Source {
    ratio: u64,
    index_key: u64,
    /// Its compressors' pooling state, both of them, in bytes.
    state: u64,
}

/// The pooling state of a compressor over rows of `row` values at ratio
/// `ratio`: values and scores in f32, of `ratio` groups of the projection's
/// width, or two groups each when they overlap; none at ratio 1.
fn state_bytes(c: Compressor, ratio: u64, row: u64) -> u64 {
    if ratio <= 1 {
        return 0;
    }
    let span = if c.overlap { 2 } else { 1 };
    span * ratio * span * row * F32_BYTES * 2
}

/// One file's KV byte function.
#[derive(Clone, Debug)]
pub struct KvLayout {
    window: u64,
    latent: u64,
    /// Per layer: the stream it sources, if any.
    sources: Vec<Option<Source>>,
}

impl KvLayout {
    /// The window from `hp`'s [`Hparams::window`], the latent width from
    /// `attn_kv`'s output dim (one width on every layer), the sources from the
    /// layers that own a compressor, each with its stream's ratio, its index
    /// key width (module doc) and its compressors' state; a missing tensor is
    /// an error naming it.
    pub fn from_file(split: &Split, hp: &Hparams) -> Result<KvLayout, PlacementError> {
        let mut latent = None;
        let mut sources = Vec::with_capacity(hp.n_layer);
        for (l, kind) in hp.layers.iter().enumerate() {
            let kv_name = names::attn_kv(l);
            let width = out_dim(split, &kv_name)?;
            if let Some(first) = latent.filter(|&w| w != width) {
                return Err(PlacementError::Tensor {
                    name: kv_name,
                    detail: format!("latent width {width}, layer 0's is {first}"),
                });
            }
            latent = Some(width);
            let (Some(c), true) = (kind.compressor, kind.compressed()) else {
                sources.push(None);
                continue;
            };
            let ratio = u64::from(kind.ratio());
            let (index_key, index_state) = match (kind.index_compressor, kind.dense) {
                (Some(ic), _) => {
                    let key = hp.indexer.head_dim as u64;
                    (key, state_bytes(ic, ratio, key))
                }
                (None, Some(_)) => (0, 0),
                (None, None) => (out_dim(split, &names::indexer_attn_k(l))?, 0),
            };
            sources.push(Some(Source {
                ratio,
                index_key,
                state: state_bytes(c, ratio, width) + index_state,
            }));
        }
        Ok(KvLayout {
            window: hp.window as u64,
            latent: latent.unwrap_or(0),
            sources,
        })
    }

    /// The layers that source a compressed stream, with their ratios.
    pub fn sources(&self) -> impl Iterator<Item = (usize, u64)> + '_ {
        self.sources
            .iter()
            .enumerate()
            .filter_map(|(l, s)| s.map(|s| (l, s.ratio)))
    }
}

impl KvBytes for KvLayout {
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        let window = ctx_max.min(self.window) * self.latent * F16_BYTES;
        match self.sources.get(layer).copied().flatten() {
            Some(s) => {
                window
                    + ctx_max.div_ceil(s.ratio) * (self.latent + s.index_key) * F16_BYTES
                    + s.state
            }
            None => window,
        }
    }

    fn shadow_bytes(&self, _layer: usize, ctx_max: u64) -> u64 {
        ctx_max * self.latent * F16_BYTES
    }
}

/// A 2-D tensor's output width, `dims[1]`.
fn out_dim(split: &Split, name: &str) -> Result<u64, PlacementError> {
    let refuse = |detail: String| PlacementError::Tensor {
        name: name.to_string(),
        detail,
    };
    let Some((_, t)) = split.find(name) else {
        return Err(refuse("is not in the file".to_string()));
    };
    t.dims
        .get(1)
        .copied()
        .ok_or_else(|| refuse(format!("has dims {:?}, no output width", t.dims)))
}
