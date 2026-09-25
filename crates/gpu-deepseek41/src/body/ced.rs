//! The CED triangle ([`Ced`]): which positions of a prompt call each layer
//! runs, walked back from every reader of what the layer leaves.
//!
//! A layer's block at position `p` reads the layer's own window ring — its
//! latent rows of `p + 1 − W ..= p`, `W` the ring's rows — and, through its
//! stream, the rows, index keys and lists of layers at or below it, the
//! lists written at `p` itself. What a call of positions `first .. end`
//! leaves for the next step is each layer's ring, the compressed rows, keys
//! and compressor states, and the head's logits at `end − 1`. So, walking
//! from the last layer down:
//!
//! - the last layer runs its block at `end − 1`, for the head; a layer the
//!   feature tap reads after runs its block at the last `window` positions
//!   of the call, the rows a draft keeps;
//! - every layer writes its latent rows at the last `W` positions, the ring
//!   the next step reads, and at the `W − 1` before its first block
//!   position, which that block's window reads;
//! - a layer runs its block wherever the layer above reads its input, for
//!   the rows or for the block;
//! - a layer that owns a compressor or index keys writes its latent part —
//!   the latent rows, the compressor, the index key — at every position:
//!   every later position of every layer of its stream reads them. Below it
//!   every layer then runs every position.
//!
//! Each start rounds down to a chunk start of the call's batches: a chunk's
//! image and words are laid out from its first position, so a layer runs
//! whole chunks, and running a position no reader needs changes nothing. The
//! lists need no term: a layer reads the list its top-k source (at or below
//! it) wrote at the same position, where that layer's block ran too.
//!
//! The walk needs two facts of the file, checked at load ([`exact`]): every
//! source a layer reads is at or below it, and index keys sit only on a
//! compressor's layer, whose latent part writes them. A file that breaks one
//! runs every layer at every position.

use std::ops::Range;

use bloomery_gpu::GpuError;
use model::arch::deepseek41::hparams::LayerKind;

use super::prefill::CHUNK;

/// What layer `i` of the card runs over a prompt call: its latent part —
/// the latent rows, with the compressor and index key on a layer that owns
/// them — from `part`, its whole block from `full`. `first <= part <= full
/// <= end`, both chunk starts of the call or `end`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerNeed {
    pub part: usize,
    pub full: usize,
}

/// The layers' needs over one prompt call of positions `first .. end`, and
/// the first position whose features the call hands over (`end` for none).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Need {
    pub first: usize,
    pub end: usize,
    pub features: usize,
    pub layers: Vec<LayerNeed>,
}

/// What a chunk of one layer runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mode {
    None,
    Part,
    Full,
}

impl Need {
    /// Every layer at every position of `first .. end`.
    #[must_use]
    pub fn all(first: usize, end: usize, layers: usize, features: usize) -> Need {
        Need {
            first,
            end,
            features,
            layers: vec![
                LayerNeed {
                    part: first,
                    full: first
                };
                layers
            ],
        }
    }

    /// What layer `i` runs over the chunk that starts at `start`.
    pub(super) fn mode(&self, i: usize, start: usize) -> Mode {
        match self.layers.get(i) {
            Some(n) if start >= n.full => Mode::Full,
            Some(n) if start >= n.part => Mode::Part,
            _ => Mode::None,
        }
    }

    /// The positions whose ring shadow rows some layer leaves unwritten:
    /// `first` to the highest latent start. A cut whose restore reads one of
    /// them cannot be granted.
    #[must_use]
    pub fn hole(&self) -> Range<usize> {
        let top = self
            .layers
            .iter()
            .map(|n| n.part)
            .max()
            .unwrap_or(self.first);
        self.first..top.max(self.first)
    }

    /// The shadow rows layer `i` writes: its latent part's positions.
    #[must_use]
    pub fn written(&self, i: usize) -> Range<usize> {
        self.layers
            .get(i)
            .map_or(self.end..self.end, |n| n.part..self.end)
    }

    /// Positions of blocks and of latent parts over every layer: what the
    /// call runs, against `layers × (end − first)` for every layer at every
    /// position.
    #[must_use]
    pub fn counts(&self) -> (usize, usize) {
        self.layers.iter().fold((0, 0), |(f, p), n| {
            (f + (self.end - n.full), p + (n.full - n.part))
        })
    }

    /// Each layer's block start, then each layer's latent start, as the
    /// `stat prefill ced=` line prints them.
    #[must_use]
    pub fn describe(&self) -> String {
        let join = |f: fn(&LayerNeed) -> usize| {
            self.layers
                .iter()
                .map(|n| f(n).to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        let (full, part) = self.counts();
        format!(
            "first={} end={} features_from={} block_positions={full} part_positions={part} \
             full_from=[{}] part_from=[{}]",
            self.first,
            self.end,
            self.features,
            join(|n| n.full),
            join(|n| n.part)
        )
    }
}

/// One layer as the walk sees it: whether it owns a compressor and index
/// keys, and the sources of its stream (rows, index keys, list).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CedLayer {
    pub compressor: bool,
    pub index_keys: bool,
    pub sources: Option<[usize; 3]>,
}

impl CedLayer {
    /// Layer `kind`'s facts.
    #[must_use]
    pub fn of(kind: &LayerKind) -> CedLayer {
        CedLayer {
            compressor: kind.compressor.is_some(),
            index_keys: kind.index_keys,
            sources: kind
                .stream
                .map(|s| [s.kv_source, s.index_key_source, s.topk_source]),
        }
    }
}

/// Whether the walk is exact for `layers` (the model's, from layer 0), or
/// the fact it breaks: a source above its reader, or index keys on a layer
/// without a compressor.
pub fn exact(layers: &[CedLayer]) -> Result<(), &'static str> {
    for (l, c) in layers.iter().enumerate() {
        if c.index_keys && !c.compressor {
            return Err("index keys on a layer without a compressor");
        }
        if c.sources.is_some_and(|s| s.iter().any(|&src| src > l)) {
            return Err("a layer reads the stream, keys or list of a layer above it");
        }
    }
    Ok(())
}

/// Whether a prefill runs the triangle, and why not when it does not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CedState {
    On,
    Off(&'static str),
}

impl std::fmt::Display for CedState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CedState::On => write!(f, "on"),
            CedState::Off(why) => write!(f, "off ({why})"),
        }
    }
}

/// `BLOOMERY_CED`: unset or `on` runs the triangle where the file allows it,
/// `off` runs every layer at every position — the same-binary arm the
/// triangle is timed against; any other value is refused by name.
pub(super) fn from_env() -> Result<bool, GpuError> {
    match std::env::var("BLOOMERY_CED").as_deref() {
        Err(_) | Ok("on") => Ok(true),
        Ok("off") => Ok(false),
        Ok(_) => Err(GpuError::State {
            what: "BLOOMERY_CED",
            missing: "on or off",
        }),
    }
}

/// The walk's load-time facts: per layer of the card, whether it runs its
/// latent part at every position; the ring's rows; the state.
pub(super) struct Ced {
    every: Vec<bool>,
    slots: usize,
    state: CedState,
}

impl Ced {
    /// The walk for `kinds` (every layer of the model, the card's from 0)
    /// over rings of `slots` rows, `state` as the load decided it.
    pub(super) fn new(kinds: &[LayerKind], slots: usize, on: bool) -> Ced {
        let facts: Vec<CedLayer> = kinds.iter().map(CedLayer::of).collect();
        let state = match (on, exact(&facts)) {
            (false, _) => CedState::Off("BLOOMERY_CED=off"),
            (true, Err(why)) => CedState::Off(why),
            (true, Ok(())) if slots == 0 => CedState::Off("a window ring of no rows"),
            (true, Ok(())) => CedState::On,
        };
        Ced {
            every: facts.iter().map(|c| c.compressor || c.index_keys).collect(),
            slots,
            state,
        }
    }

    pub(super) fn state(&self) -> CedState {
        self.state
    }

    /// The needs of a call of positions `first .. end` fed as batches
    /// starting at `starts` (the first of them `first`), with the feature
    /// tap's `after` (per layer, whether its output is tapped) and the
    /// trailing positions a reader keeps, when features are handed over.
    pub(super) fn need(
        &self,
        first: usize,
        end: usize,
        starts: &[usize],
        taps: Option<(&[bool], usize)>,
    ) -> Need {
        let features = taps.map_or(end, |(_, w)| end.saturating_sub(w).max(first));
        let n = self.every.len();
        if self.state != CedState::On || end <= first {
            return Need::all(first, end, n, features);
        }
        let floor = |x: usize| -> usize {
            let x = x.max(first);
            let batch = starts.iter().copied().filter(|&s| s <= x).max();
            (x - x % CHUNK).max(batch.unwrap_or(first)).max(first)
        };
        let tapped = |i: usize| taps.is_some_and(|(after, _)| after.get(i) == Some(&true));
        let mut layers = vec![
            LayerNeed {
                part: end,
                full: end
            };
            n
        ];
        let mut read_from = end - 1;
        for i in (0..n).rev() {
            if tapped(i) {
                read_from = read_from.min(features);
            }
            let full = floor(read_from);
            let part = if self.every[i] {
                first
            } else {
                floor(
                    (full + 1)
                        .saturating_sub(self.slots)
                        .min(end.saturating_sub(self.slots)),
                )
            };
            layers[i] = LayerNeed { part, full };
            read_from = part;
        }
        Need {
            first,
            end,
            features,
            layers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// V4.1's shape: 40 layers, compressors (and index keys) on 2, 8, 14
    /// and 20, each layer reading the last one at or below it.
    fn v41() -> Vec<CedLayer> {
        let sources = [2, 8, 14, 20];
        (0..40)
            .map(|l| {
                let src = sources.iter().copied().filter(|&s| s <= l).max();
                CedLayer {
                    compressor: sources.contains(&l),
                    index_keys: sources.contains(&l),
                    sources: src.map(|s| [s, s, s]),
                }
            })
            .collect()
    }

    fn walk(layers: &[CedLayer], slots: usize) -> Ced {
        Ced {
            every: layers
                .iter()
                .map(|c| c.compressor || c.index_keys)
                .collect(),
            slots,
            state: exact(layers).map_or_else(CedState::Off, |()| CedState::On),
        }
    }

    /// The rule the walk implements, checked on its output: nested ranges,
    /// chunk starts, every reader covered.
    fn check(need: &Need, slots: usize, starts: &[usize]) {
        let aligned = |x: usize| x == need.end || x.is_multiple_of(CHUNK) || starts.contains(&x);
        let n = need.layers.len();
        for (i, l) in need.layers.iter().enumerate() {
            assert!(need.first <= l.part && l.part <= l.full && l.full <= need.end);
            assert!(aligned(l.part) && aligned(l.full), "layer {i}: {l:?}");
            let want_rows = (l.full + 1)
                .saturating_sub(slots)
                .min(need.end.saturating_sub(slots));
            assert!(l.part <= want_rows.max(need.first), "layer {i}: {l:?}");
            if i + 1 < n {
                assert!(
                    l.full <= need.layers[i + 1].part,
                    "layer {i} feeds {}",
                    i + 1
                );
            }
        }
        assert!(need.layers[n - 1].full < need.end);
    }

    #[test]
    fn v41_triangle_from_zero() {
        let c = walk(&v41(), 128);
        assert_eq!(c.state, CedState::On);
        let need = c.need(0, 4096, &[0, 512, 1024, 1536, 2048, 2560, 3072, 3584], None);
        check(&need, 128, &[0]);
        // Layer 39 − j runs its block at the last 8 + 128 j positions, its
        // latent rows at the last 136 + 128 j.
        for j in 0..19 {
            let l = need.layers[39 - j];
            assert_eq!(l.full, 4096 - (8 + 128 * j), "layer {}", 39 - j);
            assert_eq!(l.part, 4096 - (136 + 128 * j), "layer {}", 39 - j);
        }
        assert_eq!(
            need.layers[20],
            LayerNeed {
                part: 0,
                full: 4096 - 2440
            }
        );
        for l in &need.layers[..20] {
            assert_eq!(*l, LayerNeed { part: 0, full: 0 });
        }
        assert_eq!(need.hole(), 0..3960);
        assert_eq!(need.counts().0, 20 * 4096 + 20 * 8 + 128 * 190);
    }

    #[test]
    fn short_calls_run_everything_where_the_window_reaches() {
        let c = walk(&v41(), 128);
        for end in [1, 2, 5, 127, 128, 129] {
            let need = c.need(0, end, &[0], None);
            check(&need, 128, &[0]);
            for l in &need.layers[..39] {
                assert_eq!(l.part, 0, "end {end}");
            }
        }
        // A second call from an unaligned start: its first chunk is partial.
        let need = c.need(700, 1100, &[700], None);
        check(&need, 128, &[700]);
        assert_eq!(
            need.layers[39],
            LayerNeed {
                part: 968,
                full: 1096
            }
        );
        assert_eq!(
            need.layers[37],
            LayerNeed {
                part: 712,
                full: 840
            }
        );
        assert_eq!(
            need.layers[36],
            LayerNeed {
                part: 700,
                full: 712
            }
        );
        assert_eq!(
            need.layers[35],
            LayerNeed {
                part: 700,
                full: 700
            }
        );
    }

    #[test]
    fn the_tap_widens_its_layers() {
        let c = walk(&v41(), 128);
        let mut after = vec![false; 40];
        after[36] = true;
        after[37] = true;
        after[38] = true;
        let plain = c.need(0, 1100, &[0, 550], Some((&after, 128)));
        assert_eq!(
            plain,
            Need {
                features: 972,
                ..c.need(0, 1100, &[0, 550], None)
            }
        );
        let wide = c.need(0, 1100, &[0, 550], Some((&after, 300)));
        check(&wide, 128, &[0, 550]);
        assert_eq!(wide.features, 800);
        assert_eq!(wide.layers[38].full, 800);
        assert!(wide.layers[38].full < plain.layers[38].full);
    }

    #[test]
    fn a_file_that_breaks_the_walk_runs_everything() {
        let mut keys_alone = v41();
        keys_alone[30].index_keys = true;
        assert_eq!(
            exact(&keys_alone),
            Err("index keys on a layer without a compressor")
        );
        let mut above = v41();
        above[25].sources = Some([20, 20, 26]);
        assert!(exact(&above).is_err());
        let c = walk(&above, 128);
        assert!(matches!(c.state, CedState::Off(_)));
        let need = c.need(0, 4096, &[0], None);
        assert_eq!(need, Need::all(0, 4096, 40, 4096));
        assert_eq!(need.hole(), 0..0);
        // A compressor above the decoder's first layer is exact: the layers
        // below it run every position.
        let mut high = v41();
        high[30].compressor = true;
        high[30].index_keys = true;
        let c = walk(&high, 128);
        assert_eq!(c.state, CedState::On);
        let need = c.need(0, 4096, &[0], None);
        check(&need, 128, &[0]);
        assert!(need.layers[..30].iter().all(|l| l.full == 0));
        assert_eq!(need.layers[30].part, 0);
    }
}
