//! The dispatches and the weight bytes one decoded Inkling 2 token walks,
//! visit by visit ([`token_count`], [`token_bytes`]): the paper arithmetic
//! the family's session is checked against, every dispatch named by its
//! kernel.
//!
//! A visit is the walk's unit — a stored layer's one run, its own cache slot
//! — and its dispatches divide by its kind. Every visit runs: one mix of the
//! banked blocks (a windowed visit one query row, a windowless one three,
//! one launch either way), its attention projections (a windowed visit's
//! sites in two segmented dispatches, a windowless one's in three — a
//! segment norms one input — and its two dynamic-taps projections beside
//! them), four short convolutions (k, v, the attention output's, the
//! feed-forward block's), one qk clip epilogue, one rel-bias flash, one
//! output gate, one output projection, one feed-forward mix, one router,
//! one expert pair and one down. Every sixth visit — the block residual's
//! boundary, the stored layer a multiple of the block — pushes its input
//! onto the bank. A core visit of a pass past the first adds two message
//! mixes and four controllers, one a module. The ends are five: the
//! embedding row, the final mix, the final norm, the logits and the pick.
//!
//! The weights a token reads ([`token_bytes`]): the attention's Q8_0
//! projection sites and the output projection once a visit, the kept eight
//! experts' stacks once a visit (the first twenty stored layers' in NVFP4
//! blocks and their per-expert scale words, the coda's four in Q8_0), and
//! the embedding row and the head's rows once. The small f32 matrices — the
//! norms' gains, the conv taps, the bias bank, the controllers' rows, the
//! router — are the residue beside these terms.

use models::{Mixer, ModelSpec};

use crate::arch::inkling2::layout;
use crate::inkling2_geom;

/// The block residual's block: every sixth stored layer closes its stream
/// onto the bank.
pub const BLOCK: usize = 6;

/// The stored layers whose expert stacks the Mini's file holds in NVFP4
/// blocks; the coda's four hold theirs in Q8_0.
pub const NVFP4_LAYERS: usize = 20;

/// Values one Q8_0 block holds (ggml's own `block_q8_0`).
pub const Q8_0_VALUES: u64 = 32;

/// Bytes one Q8_0 block holds.
pub const Q8_0_BYTES: u64 = 34;

/// The bytes of `values` Q8_0 values.
#[must_use]
pub const fn q8_0_bytes(values: u64) -> u64 {
    values / Q8_0_VALUES * Q8_0_BYTES
}

/// One visit's attention projection sites, in values: the query, key and
/// value projections, the channel-wise output gate's, the relative
/// projection's (`d_rel` values a query head) and the sink projection's (a
/// logit a query head), each over the hidden — and the output projection
/// over the heads' values.
pub const ATTN_SITE_VALUES: u64 = (3072 + 1024 + 1024 + 2304 + 384 + 24) * 2304 + 2304 * 3072;

/// The dispatches of the four short convolutions a visit runs.
pub const CONV_DISPATCHES: u32 = 4;

/// The dispatches of one visit's feed-forward half: the mix, the router,
/// the pair and the down.
pub const FFN_DISPATCHES: u32 = 4;

/// The dispatches a block boundary adds: the bank push.
pub const JOIN_DISPATCHES: u32 = 1;

/// The dispatches a core visit of a pass past the first adds: two message
/// mixes (the attention's and the feed-forward's tables) and four
/// controllers (the attention's, the k's, the v's and the feed-forward's,
/// one launch a module).
pub const MESSAGE_DISPATCHES: u32 = 6;

/// The dispatches of the walk's ends: the embedding row, the final mix, the
/// final norm, the logits and the pick.
pub const ENDS_DISPATCHES: u32 = 5;

/// The projections a visit's attention reads: a windowed visit's sites in
/// two segmented dispatches, a windowless one's in three (a segment norms
/// one input) and its two dynamic-taps projections beside them.
#[must_use]
pub const fn projection_dispatches(windowed: bool) -> u32 {
    if windowed { 2 } else { 3 + 2 }
}

/// The dispatches of one visit of `windowed` kind, a core visit of a pass
/// past the first when `message`, at a block boundary when `boundary`.
#[must_use]
pub const fn visit_dispatches(windowed: bool, message: bool, boundary: bool) -> u32 {
    let attn = 5 + projection_dispatches(windowed) + CONV_DISPATCHES;
    let join = if boundary { JOIN_DISPATCHES } else { 0 };
    let message = if message { MESSAGE_DISPATCHES } else { 0 };
    attn + FFN_DISPATCHES + join + message
}

/// One visit's kind in the walk: whether it attends a window, whether it is
/// a core visit of a pass past the first, and whether its stored layer
/// closes a block onto the bank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Visit {
    pub windowed: bool,
    pub message: bool,
    pub boundary: bool,
    /// The stored layer the visit runs.
    pub layer: usize,
}

/// The visits of one decoded token, in walk order ([`ModelSpec::visits`],
/// a visit a cache slot).
///
/// # Panics
///
/// On a layer of another mixer, by layer: no file of this architecture
/// holds one.
#[must_use]
pub fn token_visits(spec: &ModelSpec) -> Vec<Visit> {
    spec.visits()
        .iter()
        .map(|v| {
            let layer = v.layer as usize;
            let Mixer::RelBias(a) = &spec.layers[layer].mixer else {
                panic!("layer {layer}: a mixer other than the relative-position bias one");
            };
            Visit {
                windowed: a.window.is_some(),
                message: v.pass > 0,
                boundary: layer.is_multiple_of(BLOCK),
                layer,
            }
        })
        .collect()
}

/// The dispatches one decoded token encodes over `spec`'s visits, the ends
/// included.
///
/// # Panics
///
/// On a layer of another mixer (as [`token_visits`] does) or a count that
/// does not fit `u32`.
#[must_use]
pub fn token_count(spec: &ModelSpec) -> u32 {
    let mut n = ENDS_DISPATCHES;
    for v in &token_visits(spec) {
        n = n
            .checked_add(visit_dispatches(v.windowed, v.message, v.boundary))
            .unwrap_or_else(|| {
                panic!("token count: the dispatch count of a token does not fit u32")
            });
    }
    n
}

/// The kept experts' stacks of one visit, in bytes: the gate-up and down
/// stacks of each kept expert at the file's own blocks — the NVFP4 layers'
/// 36 bytes a 64 values and each expert's two per-expert scale words, the
/// Q8_0 coda's at the type's own blocks.
#[must_use]
pub fn kept_expert_bytes(spec: &ModelSpec, layer: usize) -> u64 {
    let models::Ffn::Moe(m) = &spec.layers[layer].ffn else {
        panic!("layer {layer}: a dense FFN is not this family's");
    };
    let (kept, ff, hidden) = (
        u64::from(m.top_k),
        u64::from(m.expert_ff),
        u64::from(spec.hidden),
    );
    if layer < NVFP4_LAYERS {
        let blocks = hidden / inkling2_geom::NVFP4_VALUES as u64;
        kept * (3 * ff * blocks * inkling2_geom::NVFP4_BYTES as u64 + 2 * 4)
    } else {
        kept * q8_0_bytes(3 * ff * hidden)
    }
}

/// The weight bytes one decoded token reads over `spec`'s visits: the
/// attention sites once a visit, the kept experts' stacks once a visit
/// ([`kept_expert_bytes`]), and the embedding row and the head's rows once.
#[must_use]
pub fn token_bytes(spec: &ModelSpec) -> u64 {
    let mut bytes = 0u64;
    for v in &token_visits(spec) {
        bytes += q8_0_bytes(ATTN_SITE_VALUES);
        bytes += kept_expert_bytes(spec, v.layer);
    }
    bytes += 2 * q8_0_bytes(u64::from(spec.vocab) * u64::from(spec.hidden));
    bytes
}

/// The K and V bytes one decoded token's flashes read at `depth` live
/// positions: the windowed visits' windows and the windowless visits'
/// planes, f16 a value.
#[must_use]
pub fn token_kv_bytes(spec: &ModelSpec, depth: u64) -> u64 {
    let layout = layout::Layout::of(spec);
    let row = layout.kv_row_bytes();
    layout.windowed_visits() * inkling2_geom::WINDOW as u64 * row
        + layout.windowless_visits() * depth * row
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::inkling2::layout::mini;

    /// The visits' kinds over the Mini's walk: the stem's six (the
    /// windowless layer 5 among them), the core's three passes of twelve
    /// (the windowless layers 11 and 17 in each, the message visits of the
    /// later passes), the coda's six; a block boundary every sixth stored
    /// layer, all of them windowed. Mutant: a boundary read off the slot
    /// instead of the stored layer.
    #[test]
    fn the_mini_visits_kinds() {
        let visits = token_visits(&mini());
        assert_eq!(visits.len(), 48);
        // The stem: layer 0 opens a block, layer 5 is windowless.
        assert_eq!(
            (visits[0].windowed, visits[0].boundary, visits[0].layer),
            (true, true, 0)
        );
        assert_eq!((visits[5].windowed, visits[5].layer), (false, 5));
        // The core's first pass: layers 6 and 12 open blocks, 11 and 17
        // are windowless; the pass runs no message visit.
        assert_eq!((visits[6].boundary, visits[6].message), (true, false));
        assert!(!visits[11].windowed && !visits[11].message);
        assert!(visits[12].boundary);
        assert!(!visits[17].windowed);
        // The later passes: every visit a message one.
        assert!(visits[18].message && visits[18].boundary);
        assert!(!visits[23].windowed && visits[23].message);
        assert!(visits[30].boundary);
        assert!(!visits[35].windowed);
        // The coda: layer 18 opens the last block, layer 23 is windowless.
        assert_eq!(visits[42].layer, 18);
        assert!(visits[42].boundary);
        assert!(!visits[47].windowed);
        assert_eq!(visits[47].layer, 23);
        let boundaries = visits.iter().filter(|v| v.boundary).count();
        assert_eq!(boundaries, 8, "the block's layers, their visits");
        assert!(
            visits.iter().filter(|v| v.boundary).all(|v| v.windowed),
            "the block's layers are the windowed ones"
        );
    }

    /// One visit's dispatches by its kind: a windowed visit's attention is
    /// the mix, two projections, four convolutions, the epilogue, the
    /// flash, the gate and the output projection; a windowless one's trades
    /// the second projection for the third and the two dynamic-taps
    /// projections; the feed-forward is four more; a block boundary pushes
    /// the bank; a message visit adds two mixes and four controllers.
    /// Mutant: the output projection counted in the gate.
    #[test]
    fn one_visits_dispatches_by_kind() {
        assert_eq!(visit_dispatches(true, false, false), 15);
        assert_eq!(visit_dispatches(true, false, true), 16);
        assert_eq!(visit_dispatches(false, false, false), 18);
        assert_eq!(visit_dispatches(true, true, false), 21);
        assert_eq!(visit_dispatches(true, true, true), 22);
        assert_eq!(visit_dispatches(false, true, false), 24);
    }

    /// One decoded token's dispatches: the ends' five beside the visits' —
    /// twenty windowed visits of the stem, the coda and the core's first
    /// pass, twenty message visits of the core's later passes, four plain
    /// windowless ones (the stem's 5, the coda's 23 and the core's first
    /// pass's 11 and 17) and four message ones, eight visits boundaries —
    /// the same order of magnitude as the CUDA side's walk of the same
    /// model, and the dispatch cost the resource timeline charges it for.
    /// Mutant: a core pass counted once.
    #[test]
    fn one_tokens_dispatches() {
        let spec = mini();
        let visits = token_visits(&spec);
        let by_kind = |windowed: bool, message: bool| {
            visits
                .iter()
                .filter(|v| v.windowed == windowed && v.message == message)
                .count()
        };
        assert_eq!(
            (
                by_kind(true, false),
                by_kind(false, false),
                by_kind(true, true),
                by_kind(false, true)
            ),
            (20, 4, 20, 4)
        );
        assert_eq!(token_count(&spec), 901);
    }

    /// The kept experts' stacks: an NVFP4 layer's eight experts read their
    /// blocks and their two scale words each, a Q8_0 coda layer's eight the
    /// type's own blocks; the coda's are the wider bytes of the two.
    /// Mutant: the scales counted a stack a visit.
    #[test]
    fn the_kept_experts_stacks() {
        let spec = mini();
        let blocks = 2304 / inkling2_geom::NVFP4_VALUES as u64;
        assert_eq!(
            kept_expert_bytes(&spec, 0),
            8 * (3 * 1152 * blocks * 36 + 2 * 4)
        );
        assert_eq!(
            kept_expert_bytes(&spec, 23),
            8 * q8_0_bytes(3 * 1152 * 2304)
        );
        assert!(
            kept_expert_bytes(&spec, 23) > kept_expert_bytes(&spec, 0),
            "the coda's Q8_0 stacks are the wider"
        );
    }

    /// The weight bytes one decoded token reads: the attention sites and
    /// the kept experts' stacks once a visit — the forty-four NVFP4
    /// visits' at their blocks, the four Q8_0 coda visits' at the type's
    /// own — and the embedding and head rows once, in all the byte wall
    /// the reference engines of this backend's machines sit under.
    /// Mutant: the head's rows counted per visit.
    #[test]
    fn one_tokens_weight_bytes() {
        let spec = mini();
        let bytes = token_bytes(&spec);
        let sites = 48 * q8_0_bytes(ATTN_SITE_VALUES);
        let nvfp4 = 44 * kept_expert_bytes(&spec, 0);
        let q8_0 = 4 * kept_expert_bytes(&spec, 23);
        let ends = 2 * q8_0_bytes(201_024 * 2304);
        assert_eq!(bytes, sites + nvfp4 + q8_0 + ends, "the named terms");
        assert!(
            bytes > 4_000_000_000 && bytes < 4_200_000_000,
            "{bytes} is the wall the walks sit under"
        );
    }

    /// The K and V bytes one token's flashes read: the windowed visits'
    /// windows are flat in the depth, the windowless visits' planes grow
    /// with it. Mutant: a window counted per position.
    #[test]
    fn one_tokens_kv_bytes() {
        let spec = mini();
        let layout = layout::Layout::of(&spec);
        let row = layout.kv_row_bytes();
        assert_eq!(token_kv_bytes(&spec, 6), (40 * 1024 + 8 * 6) * row);
        assert_eq!(token_kv_bytes(&spec, 4096), (40 * 1024 + 8 * 4096) * row);
        assert_eq!(
            token_kv_bytes(&spec, 262_144),
            (40 * 1024 + 8 * 262_144) * row
        );
    }
}
