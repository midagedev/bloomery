//! The KV bytes an ERNIE 5.5 layer holds at `ctx_max` positions: placement's accounting, not the
//! cache itself. The two kinds of layer keep different stores:
//! - an HCA layer keeps a ring of the last `window` rows of its one key-value head (f16), one
//!   pooled row of that head every `ratio` positions (f16, every one attended each step) and the
//!   pooling state of the compressor: `ratio` rows of the compressor's projection, values and
//!   scores, in f32;
//! - a latent layer keeps one row a position of the absorbed latent and its rotated key values
//!   (f16, a step selecting `top_k` of them) and one indexer key a position (f16, every one scored
//!   each step).
//!
//! The by-position buffers a room could back are the pooled rows, the latent rows and the indexer
//! keys; the ring and the pooling state are fixed.

use runtime::stores::{Access, F16_BYTES, PosBuf, PosLayout};

use super::hparams::Hparams;
use crate::placement::KvBytes;

/// The layout's name for an HCA layer's pooled rows.
pub const POOLED: &str = "pooled rows";
/// The layout's name for a latent layer's cached rows.
pub const LATENT_ROWS: &str = "latent rows";
/// The layout's name for a latent layer's indexer keys.
pub const KEYS: &str = "index keys";

const F32_BYTES: u64 = 4;

/// One layer's stores.
#[derive(Clone, Debug)]
struct Layer {
    /// The ring's rows and its row's bytes; no ring on a latent layer.
    ring_rows: u64,
    ring_row: u64,
    /// The compressors' pooling state, values and scores.
    state: u64,
    /// The by-position buffers.
    bufs: Vec<PosBuf>,
}

/// One file's KV byte function.
#[derive(Clone, Debug)]
pub struct KvLayout {
    layers: Vec<Layer>,
    rooms: PosLayout,
}

/// The stores of every layer of `hp`.
#[must_use]
pub fn kv_layout(hp: &Hparams) -> KvLayout {
    let layers: Vec<Layer> = (0..hp.n_layer)
        .map(|l| {
            if hp.latent_layer(l) {
                let m = &hp.mla;
                Layer {
                    ring_rows: 0,
                    ring_row: 0,
                    state: 0,
                    bufs: vec![
                        PosBuf {
                            what: LATENT_ROWS,
                            row: m.row() as u64 * F16_BYTES,
                            per: 1,
                            runs: 1,
                            access: Access::Select {
                                rows: hp.indexer.top_k as u64,
                            },
                        },
                        PosBuf {
                            what: KEYS,
                            row: hp.indexer.d as u64 * F16_BYTES,
                            per: 1,
                            runs: 1,
                            access: Access::Scan,
                        },
                    ],
                }
            } else {
                let c = &hp.hca;
                let row = c.head as u64 * F16_BYTES;
                Layer {
                    ring_rows: c.window as u64,
                    ring_row: row,
                    state: c.ratio as u64 * c.head as u64 * F32_BYTES * 2,
                    bufs: vec![PosBuf {
                        what: POOLED,
                        row,
                        per: c.ratio as u64,
                        runs: 1,
                        access: Access::Scan,
                    }],
                }
            }
        })
        .collect();
    let rooms = PosLayout::new(layers.iter().map(|l| l.bufs.clone()).collect());
    KvLayout { layers, rooms }
}

impl KvLayout {
    /// The bytes of layer `layer`'s by-position buffers over `ctx_max` positions.
    #[must_use]
    pub fn by_position(&self, layer: usize, ctx_max: u64) -> u64 {
        self.layers
            .get(layer)
            .map_or(0, |l| l.bufs.iter().map(|b| b.bytes(ctx_max)).sum())
    }

    /// The bytes of layer `layer`'s fixed stores: the ring (to `ctx_max` positions) and the
    /// pooling state.
    #[must_use]
    pub fn fixed(&self, layer: usize, ctx_max: u64) -> u64 {
        self.layers
            .get(layer)
            .map_or(0, |l| ctx_max.min(l.ring_rows) * l.ring_row + l.state)
    }
}

impl KvBytes for KvLayout {
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        self.by_position(layer, ctx_max) + self.fixed(layer, ctx_max)
    }

    fn rooms(&self) -> Option<&PosLayout> {
        Some(&self.rooms)
    }
}
