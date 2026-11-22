//! The Inkling 2 family's visit buffers: what a visit holds beside the
//! weights — its K and V rows (a plane of every position on a windowless
//! visit, a ring of the window's rows on a windowed one) and its short-conv
//! state — over the trunk's layer visits, and the room the whole model
//! leaves on this backend's machines.
//!
//! The counts are the common placement's ([`KvLayout::of_visits`]'s on a
//! Mac, held to these formulas by the test beside them): a visit's K and V
//! are `kv_heads · 2 · head_dim` f16 values a position, its conv state the
//! last `taps − 1` inputs over the trunk's widest channel width
//! (`2 · kv_heads · head_dim + 2 · hidden` channels, the reference's one
//! recurrent state buffer a visit), and a looped layer's visits each their
//! own, the weights shared. The conv a launch runs keeps a wider ring — the
//! runtime's `conv_ring_rows` adds a pass's rows so a cut anywhere inside a
//! pass still finds every input — which the session's arena picks when it
//! lands; the placement's count is the floor the plan is fitted to.
//!
//! The resident side reads the MLX folder this backend opens: its
//! safetensors headers name every tensor and its byte span, and the
//! resident test sums them. `MINI_RESIDENT` is that sum, a number the
//! every-OS room table reads and the Mac test holds the folder to.

use models::{
    Act, Arch, ChatSpec, Ffn, LayerSpec, LogitEpilogue, LoopSpec, Mixer, ModelSpec, Moe, OutGate,
    RelBias, Residual, Router, RouterNorm, Score,
};

use crate::inkling2_geom;

/// Bytes of an f16 value, the K and V rows' own.
const F16_BYTES: u64 = 2;

/// Bytes of an f32 value, the conv state's own.
const F32_BYTES: u64 = 4;

/// One stored layer's visits and their kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerVisits {
    /// The visits the layer runs: a looped core layer's passes, one
    /// elsewhere.
    pub visits: u64,
    /// The layer attends a window: its visits hold a ring of the window's
    /// rows. A windowless layer's hold a plane of every position.
    pub windowed: bool,
}

/// A trunk's visit buffers, read off its description. The weights are not
/// here: a looped layer's visits share its stored set, and the resident
/// side of a file is the folder's own bytes ([`MINI_RESIDENT`]).
#[derive(Clone, Debug)]
pub struct Layout {
    /// The residual width.
    pub hidden: u64,
    /// Key and value heads a layer.
    pub kv_heads: u64,
    /// Values a head.
    pub head_dim: u64,
    /// The windowed layers' window, in positions.
    pub window: u64,
    /// The short convolutions' taps.
    pub taps: u64,
    /// Per stored layer, its visits and their kind.
    pub layers: Vec<LayerVisits>,
}

impl Layout {
    /// The layout of `spec`'s trunk: every layer's visits
    /// ([`ModelSpec::visits`], a visit a cache slot) and whether it attends
    /// a window, its axes read off its [`Mixer::RelBias`] rows. A layer of
    /// another mixer names the family no file of this architecture holds;
    /// it is refused by name.
    ///
    /// # Panics
    ///
    /// On a layer of another mixer, by layer.
    pub fn of(spec: &ModelSpec) -> Layout {
        let mut layers = Vec::with_capacity(spec.layers.len());
        for (layer, l) in spec.layers.iter().enumerate() {
            let Mixer::RelBias(a) = &l.mixer else {
                panic!("layer {layer}: a mixer other than the relative-position bias one");
            };
            layers.push(LayerVisits {
                visits: u64::try_from(
                    spec.visits()
                        .iter()
                        .filter(|v| v.layer as usize == layer)
                        .count(),
                )
                .expect("a layer's visits fit u64"),
                windowed: a.window.is_some(),
            });
        }
        let first = spec
            .layers
            .iter()
            .find_map(|l| match &l.mixer {
                Mixer::RelBias(a) => Some(a),
                _ => None,
            })
            .expect("a trunk of this family holds a relative-position bias layer");
        let window = first
            .window
            .expect("a trunk of this family holds a windowed layer");
        Layout {
            hidden: u64::from(spec.hidden),
            kv_heads: u64::from(first.kv_heads),
            head_dim: u64::from(first.head_dim),
            window: u64::from(window),
            taps: u64::from(first.taps),
            layers,
        }
    }

    /// The trunk's visits in all: the session's cache-slot count.
    #[must_use]
    pub fn visits(&self) -> u64 {
        self.layers.iter().map(|l| l.visits).sum()
    }

    /// The visits that hold a ring: the windowed layers'.
    #[must_use]
    pub fn windowed_visits(&self) -> u64 {
        self.layers
            .iter()
            .filter(|l| l.windowed)
            .map(|l| l.visits)
            .sum()
    }

    /// The visits that hold a plane: the windowless layers'.
    #[must_use]
    pub fn windowless_visits(&self) -> u64 {
        self.visits() - self.windowed_visits()
    }

    /// One position's K and V bytes over every key head, f16 a value.
    #[must_use]
    pub fn kv_row_bytes(&self) -> u64 {
        self.kv_heads * 2 * self.head_dim * F16_BYTES
    }

    /// The conv state's channel width: the k and v convolutions' channels
    /// and the two hidden-width outputs', the trunk's widest layer's.
    #[must_use]
    pub fn conv_channels(&self) -> u64 {
        2 * self.kv_heads * self.head_dim + 2 * self.hidden
    }

    /// One visit's conv state: the last `taps − 1` inputs over the channel
    /// width, f32.
    #[must_use]
    pub fn conv_ring_bytes(&self) -> u64 {
        (self.taps - 1) * self.conv_channels() * F32_BYTES
    }

    /// The rows a windowed visit's ring holds at `ctx` positions: the
    /// runtime's own rule — the window and a prompt batch, padded to the
    /// ring's alignment, never past the context.
    #[must_use]
    pub fn ring_rows(&self, ctx: u64) -> u64 {
        runtime::stores::ring_rows(self.window, ctx)
    }

    /// One stored layer's visits' buffers at `ctx` positions: each visit's
    /// K and V rows and its conv state.
    #[must_use]
    pub fn layer_bytes(&self, layer: usize, ctx: u64) -> u64 {
        let Some(l) = self.layers.get(layer) else {
            return 0;
        };
        let rows = if l.windowed { self.ring_rows(ctx) } else { ctx };
        l.visits * (rows * self.kv_row_bytes() + self.conv_ring_bytes())
    }

    /// The trunk's visit buffers at `ctx` positions: every stored layer's
    /// visits' buffers in all.
    #[must_use]
    pub fn stores_bytes(&self, ctx: u64) -> u64 {
        (0..self.layers.len())
            .map(|l| self.layer_bytes(l, ctx))
            .sum()
    }
}

/// The Mini's resident bytes: the MLX folder this backend opens, its
/// safetensors headers' tensor spans summed (the resident test reads them
/// back). The GGUF-side dense and routed weights of the same model are the
/// placement's own pins, which the description's round holds.
pub const MINI_RESIDENT: u64 = 17_308_123_136;

/// The share of RAM the working set holds itself to on this backend's
/// machines, measured on them (the machine table of this backend's docs):
/// the census's free bytes are the working set less this process and the
/// host's available, the smaller.
pub const WORKING_SET_SHARE: f64 = 0.78;

/// The bytes a session of the Mini may hold at `ctx` positions: the
/// resident weights and the visit buffers. The session's own overhead — its
/// arena, its planes and its activations — sits beside these; the room
/// verdicts below leave it the band the other family of this backend
/// measured.
#[must_use]
pub fn mini_bytes(ctx: u64) -> u64 {
    MINI_RESIDENT + Layout::of(&mini()).stores_bytes(ctx)
}

/// The working set of a machine of `ram` bytes.
#[must_use]
pub fn working_set(ram: u64) -> u64 {
    (WORKING_SET_SHARE * ram as f64) as u64
}

/// The Inkling 2 Mini's description, the reader's own shape: 24 stored
/// layers of the relative-position bias attention (the windowless at 5, 11,
/// 17 and 23), the routed situ experts behind the biased sigmoid router,
/// the block mix of the residual, the head's softcap and column cut, and
/// the looped core of `[6, 18)` in three passes at rank 64.
pub fn mini() -> ModelSpec {
    let act = Act::Situ {
        gate_beta: 4.0,
        value_beta: 25.0,
    };
    let layer = |l: usize| LayerSpec {
        mixer: Mixer::RelBias(RelBias {
            heads: 24,
            kv_heads: 8,
            head_dim: 128,
            window: (l % 6 != 5).then_some(1024),
            out_gate: OutGate::Proj,
            d_rel: 16,
            extent: 1024,
            qk_clip: 22.627_417,
            rel_clip: 8.0,
            log_floor: 8192,
            log_alpha: 0.1,
            taps: 4,
            dynamic: l % 6 == 5,
        }),
        ffn: Ffn::Moe(Moe {
            experts: 128,
            top_k: 8,
            expert_ff: 1152,
            latent: None,
            pre_exp_norm: false,
            post_exp_norm: None,
            sum_norm: false,
            act,
            router: Router {
                score: Score::Sigmoid,
                bias: true,
                norm: RouterNorm::SelectedSum,
                scale: 8.0,
                hash: false,
            },
            shared: None,
        }),
        residual: Residual::AttnResMix {
            block: 6,
            queries: u32::from(l % 6 == 5) * 2 + 1,
        },
        extras: Vec::new(),
    };
    ModelSpec {
        arch: Arch::Inkling2,
        hidden: 2304,
        vocab: 201_024,
        ctx_train: 1_048_576,
        rms_eps: 1e-6,
        layers: (0..24).map(layer).collect(),
        mtp: Vec::new(),
        hc: None,
        engram: None,
        encoder: None,
        logit_epilogue: Some(LogitEpilogue {
            scale: 1.0,
            softcap: 30.0,
            cut: Some(200_058),
        }),
        loop_: Some(LoopSpec {
            core_start: 6,
            core_end: 18,
            passes: 3,
            gate_rank: 64,
        }),
        chat: ChatSpec {
            pre: "inkling2".to_string(),
            template: None,
            tools: None,
            reasoning: None,
        },
    }
}

const _: () = {
    assert!(
        inkling2_geom::HIDDEN as u64 == 2304,
        "the kernel geometry and the description agree on the hidden width"
    );
};

#[cfg(test)]
mod tests {
    use super::*;

    /// The Mini's trunk over its visits: 48 visits of 24 stored layers, the
    /// windowless slots the windowless layers' (5 and 23 once, 11 and 17
    /// three times each), every windowed visit a ring of the window and a
    /// prompt batch's rows once filled. Mutant: a windowless layer's visits
    /// counted once, or the ring without its pad.
    #[test]
    fn the_mini_trunk_over_its_visits() {
        let layout = Layout::of(&mini());
        assert_eq!(layout.visits(), 48, "a visit a cache slot");
        assert_eq!(layout.windowed_visits(), 40);
        assert_eq!(layout.windowless_visits(), 8);
        // The windowless layers: the stem's 5, the core's 11 and 17 (three
        // visits each) and the coda's 23.
        let windowless: Vec<usize> = layout
            .layers
            .iter()
            .enumerate()
            .filter(|(_, l)| !l.windowed)
            .map(|(at, _)| at)
            .collect();
        assert_eq!(windowless, [5, 11, 17, 23]);
        assert_eq!(layout.layers[11].visits, 3, "a core layer's passes");
        assert_eq!(layout.layers[5].visits, 1, "a stem layer's one");
        assert_eq!(layout.ring_rows(4096), 1536);
        assert_eq!(
            layout.ring_rows(262_144),
            1536,
            "the ring never passes the context"
        );
        assert_eq!(layout.ring_rows(6), 6, "a short context holds its own rows");
    }

    /// One visit's buffers: its K and V row (the key heads' key and value
    /// rows, f16 a value) and its conv state (the last `taps − 1` inputs
    /// over the trunk's channel width, f32), the placement's own terms.
    /// Mutant: the conv state's rows are the taps, or its channels the key
    /// heads' alone.
    #[test]
    fn one_visits_buffers() {
        let layout = Layout::of(&mini());
        assert_eq!(layout.kv_row_bytes(), 8 * 2 * 128 * F16_BYTES);
        assert_eq!(layout.kv_row_bytes(), 4096);
        assert_eq!(layout.conv_channels(), 2 * 8 * 128 + 2 * 2304);
        assert_eq!(layout.conv_channels(), 6656);
        assert_eq!(layout.conv_ring_bytes(), 3 * 6656 * F32_BYTES);
        assert_eq!(layout.conv_ring_bytes(), 79_872);
        // A windowed stem layer at a short context: its ring holds the
        // context's own rows.
        assert_eq!(layout.layer_bytes(0, 6), 6 * 4096 + 79_872);
        // A windowless stem layer: a plane of every position.
        assert_eq!(layout.layer_bytes(5, 6), 6 * 4096 + 79_872);
        // A core windowed layer's three visits, a core windowless layer's.
        assert_eq!(layout.layer_bytes(6, 4096), 3 * (1536 * 4096 + 79_872));
        assert_eq!(layout.layer_bytes(11, 4096), 3 * (4096 * 4096 + 79_872));
        // Past the trunk: nothing.
        assert_eq!(layout.layer_bytes(24, 4096), 0);
    }

    /// The trunk's stores at the pinned contexts: the placement's own
    /// numbers (the description's meta gate), held here by the formulas —
    /// the 40 windowed visits' rings capped at their filled rows, the 8
    /// windowless visits' planes, and every visit's conv state. Mutant: the
    /// windowed visits' rings counted per position.
    #[test]
    fn the_stores_at_the_pinned_contexts() {
        let layout = Layout::of(&mini());
        let conv = layout.conv_ring_bytes();
        let row = layout.kv_row_bytes();
        for (ctx, want) in [
            (6, 5_013_504u64),
            (4096, 389_709_824),
            (262_144, 8_845_426_688),
        ] {
            let windowed = 40 * (layout.ring_rows(ctx) * row + conv);
            let windowless = 8 * (ctx * row + conv);
            assert_eq!(layout.stores_bytes(ctx), windowed + windowless, "at {ctx}");
            assert_eq!(layout.stores_bytes(ctx), want, "the pin at {ctx}");
        }
        // At the looped context between the pins: the windowed rings stay
        // capped, the windowless planes grow with the context.
        let at32768 = 40 * (1536 * 4096 + conv) + 8 * (32_768 * 4096 + conv);
        assert_eq!(layout.stores_bytes(32_768), at32768);
        assert_eq!(layout.stores_bytes(32_768), 1_329_233_920);
    }

    /// The common placement's own layout over the visits, on a Mac (its
    /// crate is this backend's macOS side): every stored layer's bytes at
    /// several contexts equal these formulas', and the trunk's whole equals
    /// the stores pin. Mutant: a term of `layer_bytes` the placement counts
    /// and this file does not.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_formulas_are_the_common_placements() {
        use model::placement::KvBytes;
        let spec = mini();
        let mine = Layout::of(&spec);
        let placement =
            model::arch::place_hybrid::KvLayout::of_visits(&spec).expect("the trunk is sized");
        for ctx in [6u64, 64, 4096, 32_768, 262_144] {
            for layer in 0..spec.layers.len() {
                assert_eq!(
                    mine.layer_bytes(layer, ctx),
                    placement.layer_bytes(layer, ctx),
                    "layer {layer} at {ctx}"
                );
            }
            assert_eq!(
                (0..spec.layers.len())
                    .map(|l| placement.layer_bytes(l, ctx))
                    .sum::<u64>(),
                mine.stores_bytes(ctx),
                "the trunk at {ctx}"
            );
        }
    }

    /// The room the Mini leaves on this backend's machines: the resident
    /// weights and the visit buffers against the working set, the session's
    /// own overhead in the band the other family of this backend measured
    /// (0.8 to 1.5 GB, both ends of it). A 48 GB machine opens every
    /// context of the file's training reach; a 32 GB one opens the prompt
    /// contexts and refuses the deepest, its last fitting context near two
    /// hundred thousand positions. Mutant: the overhead counted in the
    /// working set's share.
    #[test]
    fn the_room_on_a_48gb_and_a_32gb_machine() {
        let (m4, m1) = (48_000_000_000u64, 32_000_000_000u64);
        assert_eq!(working_set(m4), 37_440_000_000);
        assert_eq!(working_set(m1), 24_960_000_000);
        let overhead = [800_000_000u64, 1_500_000_000];
        for ctx in [4096u64, 32_768] {
            let bytes = mini_bytes(ctx);
            for room in [working_set(m4), working_set(m1)] {
                assert!(
                    bytes + overhead[1] < room,
                    "ctx {ctx}: {bytes} bytes plus the overhead band fits {room}"
                );
            }
        }
        let deep = mini_bytes(262_144);
        assert!(
            deep + overhead[0] < working_set(m4),
            "the 48 GB machine opens the training reach"
        );
        assert!(
            deep + overhead[0] > working_set(m1),
            "the 32 GB machine refuses it"
        );
        // The 32 GB machine's last fitting context: the weights and the
        // overhead band leave this many positions of windowless plane.
        let spare = working_set(m1).saturating_sub(MINI_RESIDENT + overhead[0]);
        let per_position = 8 * Layout::of(&mini()).kv_row_bytes();
        let contexts = spare / per_position;
        assert!(
            (200_000..=230_000).contains(&contexts),
            "{contexts} positions"
        );
    }

    /// The MLX folder this backend opens, its headers read back: every
    /// tensor's span summed per shard and in all, the number
    /// [`MINI_RESIDENT`] names. The folder is a machine's own copy; the
    /// test runs where it is planted.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs the Inkling-2-Mini-mlx4 folder under ~/.cache/bloomery/mac-port/models"]
    fn hw_inkling2_mini_mlx4_resident_bytes() {
        let dir = std::path::Path::new(
            "/Users/hckim/.cache/bloomery/mac-port/models/Inkling-2-Mini-mlx4",
        );
        let index = dir.join(safetensors::Folder::INDEX);
        let map = safetensors::Folder::weight_map(&index)
            .unwrap_or_else(|e| panic!("{}: {e}", index.display()));
        let folder =
            safetensors::Folder::open(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        let mut per_file: std::collections::BTreeMap<String, u64> =
            std::collections::BTreeMap::new();
        let mut total = 0u64;
        for name in map.keys() {
            let (path, range) = folder
                .file_range(name)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let file = path.file_name().expect("a shard's file name");
            *per_file
                .entry(file.to_string_lossy().into_owned())
                .or_default() += range.end - range.start;
            total += range.end - range.start;
        }
        let want: std::collections::BTreeMap<String, u64> = [
            ("model-00001-of-00004.safetensors", 5_333_626_880u64),
            ("model-00002-of-00004.safetensors", 5_031_729_792),
            ("model-00003-of-00004.safetensors", 4_978_883_328),
            ("model-00004-of-00004.safetensors", 1_963_883_136),
        ]
        .into_iter()
        .map(|(f, b)| (f.to_string(), b))
        .collect();
        assert_eq!(per_file, want);
        assert_eq!(total, MINI_RESIDENT);
        assert_eq!(map.len(), 1559, "a tensor a header entry");
    }
}
