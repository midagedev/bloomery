//! The V4-Flash file's pins (`just gate-deepseek4-meta`): unsloth's
//! `UD-Q3_K_M` split set, headers only. Every value was read from that file's
//! headers or derived from them; a changed value is a red gate.

/// One compressed layer family: which layers, at which ratio.
pub struct LayerSet {
    pub what: &'static str,
    pub layers: &'static [usize],
}

// PIN(2026-09-25): the file's tensor count and tensor bytes, all four shards.
pub const TENSORS: usize = 1_328;
pub const TENSOR_BYTES: u64 = 128_073_138_524;

// PIN(2026-09-25): bytes per role family, by name (the gate's `family`).
pub const FAMILY_BYTES: &[(&str, usize, u64)] = &[
    ("routed_gate", 43, 35_668_361_216),
    ("routed_up", 43, 35_668_361_216),
    ("routed_down", 43, 49_056_579_584),
    ("shared_expert", 129, 1_149_763_584),
    ("attention", 387, 4_887_474_944),
    ("compressor", 164, 281_970_688),
    ("indexer", 126, 256_080_384),
    ("hyper_connection", 261, 135_537_756),
    ("router", 83, 90_218_496),
    ("hash_table", 3, 9_308_160),
    ("ffn_norm", 43, 704_512),
    ("token_embd", 1, 434_380_800),
    ("head", 2, 434_397_184),
];

// PIN(2026-09-25): tensors and bytes per GGML type.
pub const TYPE_BYTES: &[(&str, usize, u64)] = &[
    ("f32", 662, 165_067_100),
    ("q8_0", 489, 6_546_522_112),
    ("q6_K", 2, 868_761_600),
    ("bf16", 43, 90_177_536),
    ("i32", 3, 9_308_160),
    ("iq3_xxs", 84, 69_055_021_056),
    ("mxfp4", 45, 51_338_280_960),
];

// PIN(2026-09-25): one expert's gate, up and down bytes: IQ3_XXS gate and up
// (4096 x 2048 at 98 B per 256) and MXFP4 down (17 B per 32) on every routed
// layer but 26, whose gate and up are MXFP4 too.
pub const EXPERT_BYTES: u64 = 10_878_976;
pub const EXPERT_BYTES_MXFP4_LAYER: (usize, u64) = (26, 13_369_344);

// PIN(2026-09-25): the layer table, from compress_ratios and the tensors.
pub const WINDOW_ONLY: LayerSet = LayerSet {
    what: "ratio 0",
    layers: &[0, 1],
};
pub const SELECTED: LayerSet = LayerSet {
    what: "ratio 4, top-k over its own index keys",
    layers: &[
        2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42,
    ],
};
pub const DENSE: LayerSet = LayerSet {
    what: "ratio 128, every row",
    layers: &[
        3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41,
    ],
};
pub const HASH_ROUTED: LayerSet = LayerSet {
    what: "ffn_gate_tid2eid",
    layers: &[0, 1, 2],
};

/// One plan's card line.
pub struct CardPin {
    pub card: &'static str,
    pub dense: u64,
    pub rounding: u64,
    pub kv: u64,
    /// The planner's card experts: 0 while no card format loads IQ3_XXS or MXFP4.
    pub experts: u64,
    /// [derived] The experts the card's budget would hold at the file's bytes
    /// per expert: (usable − KV − context − scratch − margin − dense − rounding)
    /// / `EXPERT_BYTES`, rounded down, no allocator rounding on the experts.
    pub capacity: u64,
}

/// One plan's host line.
pub struct HostPin {
    pub expert_bytes: u64,
    pub table_bytes: u64,
    pub shadow: u64,
    pub headroom: i128,
}

pub struct PlanPin {
    pub name: &'static str,
    pub cards: &'static [CardPin],
    pub host: HostPin,
}

// PIN(2026-09-25): design §5 (a) on this file — the A6000 runs all 43 layers and the head; every
// routed stack stays on the host.
pub const PLAN_A: PlanPin = PlanPin {
    name: "(a) A6000",
    cards: &[CardPin {
        card: "A6000",
        dense: 7_326_325_084,
        rounding: 3_221_156,
        kv: 243_286_016,
        experts: 0,
        capacity: 3_833,
    }],
    host: HostPin {
        expert_bytes: 120_393_302_016,
        table_bytes: 443_688_960,
        shadow: 1_442_840_576,
        headroom: 136_801_572_864,
    },
};

// PIN(2026-09-25): design §5 (b) on this file, the V4.1 cut at layer 20 kept (V4 shares no stream,
// so any cut is a group boundary).
pub const PLAN_B: PlanPin = PlanPin {
    name: "(b) A6000 0-19, 3090 20-42",
    cards: &[
        CardPin {
            card: "A6000",
            dense: 3_187_784_416,
            rounding: 1_983_776,
            kv: 104_808_448,
            experts: 0,
            capacity: 4_226,
        },
        CardPin {
            card: "3090",
            dense: 4_138_540_668,
            rounding: 3_334_532,
            kv: 138_477_568,
            experts: 0,
            capacity: 1_782,
        },
    ],
    host: HostPin {
        expert_bytes: 120_393_302_016,
        table_bytes: 443_688_960,
        shadow: 1_442_840_576,
        headroom: 136_801_572_864,
    },
};

// PIN(2026-09-25): the gate placement on this file — the 3090 alone.
pub const PLAN_GATE: PlanPin = PlanPin {
    name: "gate 3090",
    cards: &[CardPin {
        card: "3090",
        dense: 7_326_325_084,
        rounding: 3_221_156,
        kv: 243_286_016,
        experts: 0,
        capacity: 1_479,
    }],
    host: HostPin {
        expert_bytes: 120_393_302_016,
        table_bytes: 443_688_960,
        shadow: 1_442_840_576,
        headroom: 136_801_572_864,
    },
};

/// Every layer but 26, whose gate and up stacks are MXFP4.
const IQ3_XXS_LAYERS: &[usize] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
];
const ALL_LAYERS: &[usize] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
];
const COMPRESSED_LAYERS: &[usize] = &[
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
];

// PIN(2026-09-25): what the engine refuses the file for (`PlanInputs::read`), feature by feature
// with its layers in order; `None` for a model-wide one.
pub const UNIMPLEMENTED: &[(&str, Option<&[usize]>)] = &[
    ("hash routing by ffn_gate_tid2eid", Some(HASH_ROUTED.layers)),
    (
        "the compressor's position table attn_compressor_ape",
        Some(COMPRESSED_LAYERS),
    ),
    ("overlapping compressor groups", Some(SELECTED.layers)),
    (
        "index keys from the indexer's own compressor",
        Some(SELECTED.layers),
    ),
    (
        "a compressed stream attended whole, without a top-k",
        Some(DENSE.layers),
    ),
    ("iq3_xxs routed experts on a card", Some(IQ3_XXS_LAYERS)),
    ("mxfp4 routed experts on a card", Some(ALL_LAYERS)),
    ("the per-head query RMS norm", None),
    ("the hyper-connection head output_hc_*", None),
    ("a model without engram sites", None),
];
