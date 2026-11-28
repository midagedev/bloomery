//! The groups, dispatches and weight bytes one decoded Inkling 2 token walks,
//! visit by visit ([`token_groups`], [`token_count`], [`token_bytes`]): the
//! paper arithmetic the family's session is checked against (the e2e gate's
//! `k_count` holds the encoder's own tally to it), every group named by what
//! it runs ([`Kind`]).
//!
//! A visit is the walk's unit — a stored layer's one run, its own cache slot
//! — and the walk (`ops.rs`) encodes it as groups, a group being the
//! dispatches between two barriers. A visit's groups, in order:
//!
//! - [`Kind::Mix`]: the attention's mix of the blocks and the stream (one
//!   query on a windowed visit, three on a windowless one; the stored layer 0
//!   reads the stream as it is, so it runs none) and, on a core visit of a
//!   pass past the first, the loop's message mix beside it. One dispatch each.
//! - [`Kind::LoopDown`] and [`Kind::LoopMerge`], a message visit's: the three
//!   attention modules' controllers, each a launch in two groups.
//! - [`Kind::Proj`]: the q, relative and gate rows in one launch, the sink
//!   rows, the k and the v, and on a windowless visit the two dynamic-taps
//!   projections: four dispatches, six on a windowless visit.
//! - [`Kind::QkClip`]: the k and v short convolutions, the clip epilogue and
//!   the cache append, one launch.
//! - [`Kind::Flash`]: the rel-bias attention, two groups (the segment walk,
//!   then the fold of its partials, a barrier the launch itself holds).
//! - [`Kind::Gate`], [`Kind::Wo`], [`Kind::ConvAttn`]: the output gate, the
//!   output projection and the attention's post-convolution, which adds its
//!   output into the stream in the same launch (a block boundary's writes the
//!   next row itself).
//! - [`Kind::FfnMix`], the feed-forward's mix, then on a message visit
//!   [`Kind::LoopFfn`] (its controller: two groups), then [`Kind::Router`]
//!   (the norm in its front).
//! - [`Kind::GateUp`], [`Kind::Pair`], [`Kind::Down`], [`Kind::Combine`]: the
//!   kept experts' rows, their activation, their down rows, and the sum, the
//!   post-convolution of it and its add into the stream in one launch.
//! - [`Kind::MsgEnd`], on the last core visit of a pass the loop runs again:
//!   the pass's blocks added into its message bank.
//!
//! The head is three groups ([`head_groups`]): the mix, the logits and the
//! pick (the logits and the pick are one launch's two groups). The embedding
//! row is the host's, so the token runs no embed dispatch.
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

/// What a group of the walk runs: the name a profile's line carries and the
/// tag the walk puts on its encoder (`ComputeEncoder::tag`). The variants
/// are in walk order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    Mix = 1,
    LoopDown,
    LoopMerge,
    Proj,
    QkClip,
    Flash,
    Gate,
    Wo,
    ConvAttn,
    FfnMix,
    LoopFfn,
    Router,
    GateUp,
    Pair,
    Down,
    Combine,
    MsgEnd,
    HeadMix,
    Head,
}

/// Every kind, in the token's walk order: the order a profile's lines print.
pub const KINDS: [Kind; 19] = [
    Kind::Mix,
    Kind::LoopDown,
    Kind::LoopMerge,
    Kind::Proj,
    Kind::QkClip,
    Kind::Flash,
    Kind::Gate,
    Kind::Wo,
    Kind::ConvAttn,
    Kind::FfnMix,
    Kind::LoopFfn,
    Kind::Router,
    Kind::GateUp,
    Kind::Pair,
    Kind::Down,
    Kind::Combine,
    Kind::MsgEnd,
    Kind::HeadMix,
    Kind::Head,
];

impl Kind {
    /// The kind's name in a profile's line.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Kind::Mix => "mix",
            Kind::LoopDown => "loop_down",
            Kind::LoopMerge => "loop_merge",
            Kind::Proj => "proj",
            Kind::QkClip => "qk_clip",
            Kind::Flash => "flash",
            Kind::Gate => "gate",
            Kind::Wo => "wo",
            Kind::ConvAttn => "conv_attn",
            Kind::FfnMix => "ffn_mix",
            Kind::LoopFfn => "loop_ffn",
            Kind::Router => "router",
            Kind::GateUp => "gate_up",
            Kind::Pair => "pair",
            Kind::Down => "down",
            Kind::Combine => "combine",
            Kind::MsgEnd => "msg_end",
            Kind::HeadMix => "head_mix",
            Kind::Head => "head",
        }
    }

    /// The kind a tag names, `None` for a tag no kind has (0 is no tag).
    #[must_use]
    pub fn from_tag(tag: u8) -> Option<Kind> {
        KINDS.iter().copied().find(|k| u8::from(*k) == tag)
    }

    /// Whether the group streams a weight matrix the token reads once: the
    /// projections, the output projection, the kept experts' rows and the
    /// head. The other kinds read activations, the cache and small f32
    /// matrices.
    #[must_use]
    pub const fn streams_weights(self) -> bool {
        matches!(
            self,
            Kind::Proj | Kind::Wo | Kind::GateUp | Kind::Down | Kind::Head
        )
    }
}

impl From<Kind> for u8 {
    fn from(k: Kind) -> u8 {
        k as u8
    }
}

/// One group of the walk: what it runs and how many dispatches it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Group {
    pub kind: Kind,
    pub dispatches: u32,
}

/// What a token encodes, summed over its command buffers: the shape of
/// `EncodeCount` on every OS.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TokenCount {
    pub groups: usize,
    pub dispatches: usize,
    pub barriers: usize,
}

/// One visit's kind in the walk: whether it attends a window, whether it is
/// a core visit of a pass past the first, whether its stored layer closes a
/// block onto the bank, whether it ends a pass the loop runs again, and
/// whether it is the stored layer 0 (which has no attention mix).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Visit {
    pub windowed: bool,
    pub message: bool,
    pub boundary: bool,
    pub pass_end: bool,
    pub first: bool,
    /// The stored layer the visit runs.
    pub layer: usize,
}

/// The groups of one visit, in the order the walk encodes them.
#[must_use]
pub fn visit_groups(v: &Visit) -> Vec<Group> {
    let g = |kind, dispatches| Group { kind, dispatches };
    let mut out = Vec::new();
    let mixes = u32::from(!v.first) + u32::from(v.message);
    if mixes > 0 {
        out.push(g(Kind::Mix, mixes));
    }
    if v.message {
        out.push(g(Kind::LoopDown, 3));
        out.push(g(Kind::LoopMerge, 3));
    }
    out.push(g(Kind::Proj, if v.windowed { 4 } else { 6 }));
    out.push(g(Kind::QkClip, 1));
    out.push(g(Kind::Flash, 1));
    out.push(g(Kind::Flash, 1));
    out.push(g(Kind::Gate, 1));
    out.push(g(Kind::Wo, 1));
    out.push(g(Kind::ConvAttn, 1));
    out.push(g(Kind::FfnMix, 1));
    if v.message {
        out.push(g(Kind::LoopFfn, 1));
        out.push(g(Kind::LoopFfn, 1));
    }
    out.push(g(Kind::Router, 1));
    for kind in [Kind::GateUp, Kind::Pair, Kind::Down, Kind::Combine] {
        out.push(g(kind, 1));
    }
    if v.pass_end {
        out.push(g(Kind::MsgEnd, 1));
    }
    out
}

/// The groups of the head: the mix, then the logits and the pick.
#[must_use]
pub const fn head_groups() -> [Group; 3] {
    [
        Group {
            kind: Kind::HeadMix,
            dispatches: 1,
        },
        Group {
            kind: Kind::Head,
            dispatches: 1,
        },
        Group {
            kind: Kind::Head,
            dispatches: 1,
        },
    ]
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
    let core = spec
        .loop_
        .map(|l| (l.core_start as usize, l.core_end as usize, l.passes));
    spec.visits()
        .iter()
        .map(|v| {
            let layer = v.layer as usize;
            let Mixer::RelBias(a) = &spec.layers[layer].mixer else {
                panic!("layer {layer}: a mixer other than the relative-position bias one");
            };
            let in_core = core.is_some_and(|(start, end, _)| (start..end).contains(&layer));
            Visit {
                windowed: a.window.is_some(),
                message: in_core && v.pass > 0,
                boundary: layer.is_multiple_of(BLOCK),
                pass_end: core.is_some_and(|(_, end, passes)| {
                    in_core && layer + 1 == end && v.pass + 1 < passes
                }),
                first: layer == 0,
                layer,
            }
        })
        .collect()
}

/// The groups one decoded token encodes over `spec`'s visits, the head's
/// included, in walk order.
///
/// # Panics
///
/// On a layer of another mixer (as [`token_visits`] does).
#[must_use]
pub fn token_groups(spec: &ModelSpec) -> Vec<Group> {
    let mut out: Vec<Group> = token_visits(spec).iter().flat_map(visit_groups).collect();
    out.extend(head_groups());
    out
}

/// What one decoded token encodes over `spec`'s visits, the head's included:
/// every group, every dispatch, and a barrier after every group but the
/// head's last (the walk closes each group with one, and ends the token on
/// the head's pick).
///
/// # Panics
///
/// On a layer of another mixer (as [`token_visits`] does).
#[must_use]
pub fn token_count(spec: &ModelSpec) -> TokenCount {
    let groups = token_groups(spec);
    TokenCount {
        groups: groups.len(),
        dispatches: groups.iter().map(|g| g.dispatches as usize).sum(),
        barriers: groups.len() - 1,
    }
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

    /// The groups and dispatches of one visit by its kind: a plain windowed
    /// visit runs fourteen groups of seventeen dispatches (the flash and
    /// the projections' launch the odd ones: the flash is two groups, the
    /// projections four dispatches); a block boundary's post-conv is the
    /// next row's stream and adds into nothing, which the plain visit's
    /// carries in the same launch, so the two count alike; a windowless
    /// visit's projections are six dispatches; a message visit adds the
    /// controllers' four groups and the message mix; the stored layer 0 has
    /// no attention mix; a pass end adds its bank's add. Mutant: the flash
    /// counted one group.
    #[test]
    fn one_visits_groups_by_kind() {
        let plain = Visit {
            windowed: true,
            message: false,
            boundary: false,
            pass_end: false,
            first: false,
            layer: 1,
        };
        let shape = |v: &Visit| {
            let g = visit_groups(v);
            (g.len(), g.iter().map(|g| g.dispatches).sum::<u32>())
        };
        assert_eq!(shape(&plain), (14, 17));
        assert_eq!(
            shape(&Visit {
                boundary: true,
                ..plain
            }),
            (14, 17)
        );
        assert_eq!(
            shape(&Visit {
                windowed: false,
                ..plain
            }),
            (14, 19)
        );
        assert_eq!(
            shape(&Visit {
                message: true,
                ..plain
            }),
            (18, 26)
        );
        assert_eq!(
            shape(&Visit {
                first: true,
                ..plain
            }),
            (13, 16)
        );
        assert_eq!(
            shape(&Visit {
                pass_end: true,
                ..plain
            }),
            (15, 18)
        );
        let kinds: Vec<Kind> = visit_groups(&plain).iter().map(|g| g.kind).collect();
        assert_eq!(
            kinds.iter().filter(|k| **k == Kind::Flash).count(),
            2,
            "the walk and the fold"
        );
        assert!(
            kinds.windows(2).all(|w| u8::from(w[0]) <= u8::from(w[1])),
            "a visit's groups run in the kinds' own order"
        );
    }

    /// One decoded token's groups: the Mini's forty-eight visits and the
    /// head's three. Derived on paper from the walk's call sites (the e2e
    /// gate's `k_count` holds the encoder to it): 772 groups, 1,052
    /// dispatches, a barrier after every group but the last. Mutant: a core
    /// pass counted once.
    #[test]
    fn one_tokens_groups() {
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
        assert_eq!(visits.iter().filter(|v| v.pass_end).count(), 2);
        assert_eq!(
            token_count(&spec),
            TokenCount {
                groups: 772,
                dispatches: 1052,
                barriers: 771
            }
        );
        let groups = token_groups(&spec);
        assert_eq!(groups.last().map(|g| g.kind), Some(Kind::Head));
        assert_eq!(
            groups.iter().filter(|g| g.kind == Kind::Head).count(),
            2,
            "the logits and the pick"
        );
    }

    /// Every kind has one name, one tag and a place in [`KINDS`]; a tag no
    /// kind has names none. Mutant: two kinds sharing a name.
    #[test]
    fn the_kinds_name_themselves_once() {
        for (i, k) in KINDS.iter().enumerate() {
            assert_eq!(Kind::from_tag(u8::from(*k)), Some(*k));
            assert_eq!(
                usize::from(u8::from(*k)),
                i + 1,
                "the tags run in walk order"
            );
            assert!(
                KINDS[..i].iter().all(|o| o.name() != k.name()),
                "{} named twice",
                k.name()
            );
        }
        assert_eq!(Kind::from_tag(0), None);
        assert_eq!(Kind::from_tag(KINDS.len() as u8 + 1), None);
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
