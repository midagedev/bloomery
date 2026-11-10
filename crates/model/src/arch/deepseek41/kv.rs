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
//! its window ring in host memory: `ctx_max` latent rows in f16, one per position, from which
//! a cut of the cache restores the ring rows later positions overwrote. The plan counts it on
//! the host, not on the card, at the cap whatever the floor (the shadow is a host room backed by
//! position, which still grows to the cap).
//!
//! The by-position buffers a body holds in rooms ([`KvBytes::rooms`]) are the compressed rows and
//! the index keys of each source layer: a step selects at most `indexer.top_k` compressed rows
//! (host-able), and the indexer scores every index key (card-only). The window ring and the
//! compressor states are fixed; the ring shadow is a host room of its own, not in the layout
//! (the layout is the part of [`KvBytes::layer_bytes`] a room backs).

use gguf::Split;

use super::hparams::{Compressor, Hparams};
use super::names;
use crate::placement::{KvBytes, PlacementError};
use runtime::stores::{Access, PosBuf, PosLayout};

/// The layout's names for a source layer's two buffers.
pub const ROWS: &str = "compressed rows";
pub const KEYS: &str = "index keys";

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
    /// The by-position buffers a body holds in rooms ([`KvBytes::rooms`]).
    rooms: PosLayout,
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
        Ok(KvLayout::assemble(
            hp.window as u64,
            latent.unwrap_or(0),
            hp.indexer.top_k as u64,
            sources,
        ))
    }

    /// The layout of `window` ring rows, `latent`-wide rows, a selection of `top_k` compressed
    /// rows and `sources`, with the rooms they declare: per source layer its compressed rows
    /// (`Select`: a step reads at most `top_k` of them) and, when it owns index keys, those
    /// (`Scan`: the indexer scores every row), one row every `ratio` positions.
    fn assemble(window: u64, latent: u64, top_k: u64, sources: Vec<Option<Source>>) -> KvLayout {
        let layers = sources
            .iter()
            .map(|s| {
                let Some(s) = s else { return Vec::new() };
                let mut bufs = vec![PosBuf {
                    what: ROWS,
                    row: latent * F16_BYTES,
                    per: s.ratio,
                    runs: 1,
                    access: Access::Select { rows: top_k },
                }];
                if s.index_key > 0 {
                    bufs.push(PosBuf {
                        what: KEYS,
                        row: s.index_key * F16_BYTES,
                        per: s.ratio,
                        runs: 1,
                        access: Access::Scan,
                    });
                }
                bufs
            })
            .collect();
        KvLayout {
            window,
            latent,
            sources,
            rooms: PosLayout::new(layers),
        }
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

    /// The sources' compressed rows and index keys: the rest of a layer (ring, compressor state)
    /// is fixed.
    fn rooms(&self) -> Option<&PosLayout> {
        Some(&self.rooms)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::deepseek41::place::PlanInputs;
    use crate::placement::{
        Card, Host, KvBytes, KvDefault, Machine, ModelTensor, ModelTensors, Node, PlanLevers, Role,
        UnifiedPool, workstation::HostRead,
    };
    use bloomery_levers::{KvFloor, KvHost};
    use gguf::GgmlType;
    use models::{Arch, ChatSpec, ModelSpec};
    use std::num::NonZeroU64;

    /// The tiny four-layer file's inputs over `kv`, planned on a host room of `pool` bytes.
    fn tiny_inputs(kv: KvLayout, pool: u64) -> PlanInputs {
        PlanInputs {
            hp: Hparams::tiny(4),
            model: ModelTensors {
                tensors: (0..4)
                    .flat_map(|l| {
                        ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"].map(|stem| ModelTensor {
                            name: format!("blk.{l}.{stem}.weight"),
                            shard: 0,
                            layer: Some(l),
                            role: Role::RoutedExperts,
                            ty: GgmlType::Q4_K,
                            dims: vec![256, 256, 64],
                            file_bytes: 256 * 256 * 64 / 256 * 144,
                            gathered_rows: None,
                        })
                    })
                    .collect(),
                layers: 4,
                experts: 64,
                experts_used: 2,
                bf16_words: |_| false,
            },
            spec: ModelSpec {
                arch: Arch::Deepseek4,
                hidden: 256,
                vocab: 1,
                ctx_train: 128,
                rms_eps: 1e-6,
                layers: Vec::new(),
                mtp: Vec::new(),
                hc: None,
                engram: None,
                encoder: None,
                logit_epilogue: None,
                chat: ChatSpec {
                    pre: String::new(),
                    template: None,
                    tools: None,
                    reasoning: None,
                },
            },
            kv,
            room: (pool, HostRead::Given),
            arena: None,
            kv_host_page: None,
        }
    }

    /// One card and one node of `pool` bytes, the unified pool the plan runs in.
    fn pool_machine(pool: u64) -> Machine {
        Machine {
            cards: vec![Card {
                name: "pool".into(),
                tier_leg: None,
                device: None,
                usable_bytes: pool,
                context_bytes: 0,
                scratch_bytes: 0,
                margin_bytes: 0,
                granule_bytes: NonZeroU64::MIN,
                free_bytes: Some(pool),
                held_by: None,
                layers: 0..4,
                head: true,
                token_embedding: false,
                reserves: Vec::new(),
                reserve_only: false,
            }],
            tiers: Vec::new(),
            nodes: vec![Node {
                host: Host {
                    usable_bytes: pool,
                    reserves: Vec::new(),
                },
                unified: Some(UnifiedPool { bytes: pool }),
            }],
        }
    }

    #[test]
    fn nvme_room_is_capped_by_the_unified_pool() {
        // The card's budget and the pool each hold the prompt batch beside
        // the 2 MiB card and 26 MiB pool the room cap is read at.
        let kv = || KvLayout::assemble(128, 256, 1, vec![None; 4]);
        let batch = tiny_inputs(kv(), 0)
            .prompt_reserve_bytes(&pool_machine(0).cards[0], 64, false)
            .expect("the tiny file's batch");
        let pool = (26 << 20) + batch;
        let mut inputs = tiny_inputs(kv(), pool);
        let machine = pool_machine(pool);
        let levers = PlanLevers {
            card_budget_bytes: bloomery_levers::CardBudget::Every((2 << 20) + batch),
            ..PlanLevers::default()
        };
        let expected = inputs
            .plan(&machine, 64, &levers)
            .expect("the pool fits with NVMe");
        let expected_rows = format!("{:?}", expected.rows);
        assert!(expected.host.nvme_expert_bytes > 0);
        inputs.room.0 += 16 << 20;
        let plan = inputs
            .plan(&machine, 64, &levers)
            .expect("a larger host reading keeps the same pool budget");
        assert_eq!(format!("{:?}", plan.rows), expected_rows);
        assert!(plan.violations().is_empty());
    }

    /// The design's DeepSeek-V4.1-Flash stack: 40 layers, a latent of 512, 128-wide index keys,
    /// three sources at ratio 2 and one at ratio 1.
    fn flash() -> KvLayout {
        let source = |ratio| {
            Some(Source {
                ratio,
                index_key: 128,
                state: 0,
            })
        };
        let mut sources = vec![None; 40];
        for (l, ratio) in [(1, 2), (2, 2), (3, 2), (4, 1)] {
            sources[l] = source(ratio);
        }
        KvLayout::assemble(128, 512, 512, sources)
    }

    /// The declaration's bytes a position: 3 × (1,024 + 256) / 2 + 1,280 = 3,200 on the card
    /// (rows and index keys), of which 3 × 512 + 1,024 = 2,560 a selected row (host-able), and
    /// the ring shadow's 40 × 1,024 = 40,960 on the host. FAIL-first: a `per` of 1 on the
    /// ratio-2 buffers, or the keys marked `Select`, moves a figure and goes red.
    #[test]
    fn the_declared_rooms_are_3200_card_and_40960_host_bytes_a_position() {
        let kv = flash();
        let rooms = kv.rooms().expect("a V4.1 layout declares rooms");
        let ctx = 8_192;
        let card: u64 = (0..40).map(|l| rooms.bytes(l, ctx)).sum();
        let selected: u64 = (0..40).map(|l| rooms.host_able(l, ctx)).sum();
        let shadow: u64 = (0..40).map(|l| kv.shadow_bytes(l, ctx)).sum();
        assert_eq!(card, 3_200 * ctx);
        assert_eq!(selected, 2_560 * ctx);
        assert_eq!(shadow, 40_960 * ctx);
    }

    /// A layer's declared rooms are the rows and keys of its layer bytes and nothing else: what
    /// the plan counts at the cap beside them (the ring, the compressor state) does not grow
    /// with the context past the window. FAIL-first: a rooms buffer that is not a part of
    /// `layer_bytes` (a wrong row width) breaks the equality at once.
    #[test]
    fn the_rooms_are_the_layer_bytes_less_the_fixed_ring_and_state() {
        let kv = flash();
        let rooms = kv.rooms().expect("rooms");
        for l in 0..40 {
            let fixed = |ctx| kv.layer_bytes(l, ctx) - rooms.bytes(l, ctx);
            assert_eq!(fixed(1_000_000), fixed(262_144), "layer {l}");
            assert_eq!(fixed(262_144), 128 * 512 * 2, "layer {l}");
        }
    }

    /// The card's cache term of a plan of `kv` at `ctx_max` under `levers`.
    fn card_kv(kv: &KvLayout, ctx_max: u64, levers: &PlanLevers) -> u64 {
        let pool = 1 << 30;
        let inputs = tiny_inputs(kv.clone(), pool);
        let machine = pool_machine(pool);
        inputs
            .plan(&machine, ctx_max, levers)
            .expect("the pool holds the cache")
            .cards[0]
            .kv_bytes
    }

    /// With every lever unset the cache is counted whole at the cap; a floor lever counts the
    /// declared rooms at the floor and the fixed part at the cap; a seat default that the
    /// context passes counts as that floor, and one the context does not pass counts none.
    /// FAIL-first: reading the rooms at the cap under a floor (the declaration dropped)
    /// leaves the floor row equal to the unset one.
    #[test]
    fn the_floor_counts_the_rooms_at_the_floor_and_the_rest_at_the_cap() {
        let kv = KvLayout::assemble(
            128,
            256,
            1,
            (0..4)
                .map(|l| {
                    Some(Source {
                        ratio: if l == 3 { 1 } else { 2 },
                        index_key: 128,
                        state: 0,
                    })
                })
                .collect(),
        );
        let (cap, floor) = (16_384, 8_192);
        let rooms = kv.rooms().expect("rooms");
        let whole: u64 = (0..4).map(|l| kv.layer_bytes(l, cap)).sum();
        let at_floor: u64 = (0..4)
            .map(|l| kv.layer_bytes(l, cap) - rooms.bytes(l, cap) + rooms.bytes(l, floor))
            .sum();
        assert!(at_floor < whole);
        let unset = PlanLevers::default();
        assert_eq!(card_kv(&kv, cap, &unset), whole);
        let lever = PlanLevers {
            kv_floor: Some(floor),
            ..PlanLevers::default()
        };
        assert_eq!(card_kv(&kv, cap, &lever), at_floor);
        let seat = PlanLevers {
            kv_floor_default: Some(floor),
            ..PlanLevers::default()
        };
        assert_eq!(card_kv(&kv, cap, &seat), at_floor);
        assert_eq!(
            card_kv(&kv, floor, &seat),
            (0..4).map(|l| kv.layer_bytes(l, floor)).sum::<u64>()
        );
    }

    /// The selected rows' host bytes a spill plan counts: the pages past the floor, 2,560 B a
    /// position a slot on the V4.1 stack (3 × 512 + 1,024 selected, the index keys card-only),
    /// every row at the cap under `all`, none under `off` or with the cache planned whole. A
    /// spill or `all` plan with no page read is refused by name. FAIL-first: counting the keys
    /// (every buffer, not the host class) or the floor's own pages moves the spill figure.
    #[test]
    fn spill_counts_the_selected_rows_past_the_floor_on_the_host() {
        let mut inputs = tiny_inputs(flash(), 1 << 30);
        let (cap, floor) = (262_144, 8_192);
        assert!(
            inputs
                .kv_host_bytes(cap, 1, KvHost::Spill, Some(floor))
                .is_err()
        );
        inputs.kv_host_page = Some(4_096);
        let per_slot = 2_560 * (cap - floor);
        assert_eq!(
            inputs
                .kv_host_bytes(cap, 2, KvHost::Spill, Some(floor))
                .unwrap(),
            2 * per_slot
        );
        assert_eq!(per_slot, 650_117_120);
        assert_eq!(
            inputs.kv_host_bytes(cap, 2, KvHost::All, None).unwrap(),
            2 * 2_560 * cap
        );
        assert_eq!(
            inputs.kv_host_bytes(cap, 2, KvHost::Spill, None).unwrap(),
            0
        );
        assert_eq!(
            inputs
                .kv_host_bytes(cap, 2, KvHost::Off, Some(floor))
                .unwrap(),
            0
        );
    }

    /// A plan under the seat's default (floor 8,192, spill) counts the spill's host bytes
    /// against the host room, and the plan with every lever unset counts none: the host's
    /// headroom falls by exactly those bytes. FAIL-first: dropping the `kv_host_planes` call
    /// leaves both plans with the same headroom.
    #[test]
    fn the_unset_plan_counts_no_host_rows_and_the_seat_default_counts_the_spill() {
        let kv = KvLayout::assemble(
            128,
            256,
            1,
            (0..4)
                .map(|l| {
                    Some(Source {
                        ratio: if l == 3 { 1 } else { 2 },
                        index_key: 128,
                        state: 0,
                    })
                })
                .collect(),
        );
        let pool = 1 << 30;
        let mut inputs = tiny_inputs(kv, pool);
        inputs.kv_host_page = Some(4_096);
        let machine = pool_machine(pool);
        let seat = PlanLevers::default().resolve_kv(
            KvFloor::Unset,
            None,
            KvDefault {
                floor: 8_192,
                host: KvHost::Spill,
            },
        );
        let held = |levers: &PlanLevers| {
            let plan = inputs.plan(&machine, 16_384, levers).expect("a plan");
            (plan.host.kv_host_bytes, plan.host.headroom_bytes)
        };
        let (none, _) = held(&PlanLevers::default());
        // The same floor with the rows kept on the card frees the same card bytes and counts no
        // host rows: the headroom differs by the spill's bytes alone.
        let floor_off = PlanLevers {
            kv_host: KvHost::Off,
            ..seat.clone()
        };
        let (off, room) = held(&floor_off);
        let (spill, after) = held(&seat);
        assert_eq!((none, off), (0, 0));
        // 3 ratio-2 layers of 256-wide rows (256 B a position) and one ratio-1 layer (512 B).
        assert_eq!(spill, (3 * 256 + 512) * 8_192);
        assert_eq!(room - after, i128::from(spill));
    }

    /// The ring shadows are host rooms that grow to the cap as positions arrive, so the plan
    /// counts them on the host at the cap whatever the floor (an uncounted growth is a
    /// mid-session host OOM): a row of the latent's width (256 values of f16) a position a layer.
    /// FAIL-first: the planner reading the shadows at the floor leaves the floor row at half.
    #[test]
    fn the_shadow_term_is_counted_at_the_cap_under_a_floor() {
        let kv = KvLayout::assemble(
            128,
            256,
            1,
            (0..4)
                .map(|l| {
                    Some(Source {
                        ratio: if l == 3 { 1 } else { 2 },
                        index_key: 128,
                        state: 0,
                    })
                })
                .collect(),
        );
        let pool = 1 << 30;
        let inputs = tiny_inputs(kv, pool);
        let machine = pool_machine(pool);
        let (cap, floor) = (16_384, 8_192);
        let shadow = |levers: &PlanLevers| {
            inputs
                .plan(&machine, cap, levers)
                .expect("a plan")
                .host
                .shadow_bytes
        };
        let per_position = 4 * 256 * 2;
        assert_eq!(shadow(&PlanLevers::default()), per_position * cap);
        let lever = PlanLevers {
            kv_floor: Some(floor),
            ..PlanLevers::default()
        };
        assert_eq!(shadow(&lever), per_position * cap);
        assert!(per_position * floor < per_position * cap);
    }
}
