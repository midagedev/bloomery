//! The KV bytes a DeepSeek-V4.1 layer holds at `ctx_max` tokens — placement's
//! accounting (`docs/v41-placement.md` §4), not the cache itself.
//!
//! Every layer keeps a window of `min(ctx_max, W)` latent rows in f16. A layer
//! that carries `attn_compressor_kv` sources a compressed stream (the fork's
//! loader reads the roles off the tensors the same way,
//! `llama.cpp-fork/src/models/deepseek41.cpp:209-247`): at ratio r it adds
//! `⌈ctx_max / r⌉` rows of latent plus index key in f16, and a pooling state of
//! r latent rows in f32, twice (values and scores). The layers that read a
//! source's stream hold none of it. The window and every layer's ratio come
//! from [`Hparams`].

use gguf::Split;

use super::hparams::Hparams;
use super::names;
use crate::placement::{KvBytes, PlacementError};

const F16_BYTES: u64 = 2;
const F32_BYTES: u64 = 4;

/// A layer's compressed stream.
#[derive(Clone, Copy, Debug)]
struct Source {
    ratio: u64,
    index_key: u64,
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
    /// The window from [`Hparams::window`], the latent width from `attn_kv`'s
    /// output dim (one width on every layer), the sources from the layers that
    /// own a compressor, each with its stream's ratio and its index key width from
    /// `indexer.attn_k`'s output dim; a missing key or tensor is an error
    /// naming it.
    pub fn from_file(split: &Split) -> Result<KvLayout, PlacementError> {
        let hp = Hparams::read(split)?;
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
            let Some(stream) = kind.compressor.and(kind.stream) else {
                sources.push(None);
                continue;
            };
            let index_key = out_dim(split, &names::indexer_attn_k(l))?;
            sources.push(Some(Source {
                ratio: u64::from(stream.ratio),
                index_key,
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
                    + s.ratio * self.latent * F32_BYTES * 2
            }
            None => window,
        }
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
