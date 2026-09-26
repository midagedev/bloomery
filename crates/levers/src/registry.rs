//! The rows: every `BLOOMERY_*` lever a crate reads, and the runner variables
//! the repository's lever documentation names — each name once. A row with a
//! file to read it in place (`Site::Direct`, or a non-empty `left`) is also a
//! line of `tools/levers-direct.txt`, the list `tools/check-levers.sh` holds
//! every in-place environment read of the crates' sources to; the test
//! `registry_and_allow_list_agree` holds the two to each other.

use crate::{Class, InPlace, Kind, LeverSpec, Site, Unset};

pub const THREADS: &str = "BLOOMERY_THREADS";
pub const SPIN: &str = "BLOOMERY_SPIN";
pub const CED: &str = "BLOOMERY_CED";
pub const PREFILL: &str = "BLOOMERY_PREFILL";
pub const PREFILL_GROUP: &str = "BLOOMERY_PREFILL_GROUP";
pub const ENGRAM_HELPER: &str = "BLOOMERY_ENGRAM_HELPER";
pub const STEP_STATS: &str = "BLOOMERY_STEP_STATS";
pub const HOT_LIST: &str = "BLOOMERY_HOT_LIST";
pub const CARD_BUDGET: &str = "BLOOMERY_CARD_BUDGET";
pub const PIN_MAIN: &str = "BLOOMERY_PIN_MAIN";
pub const DRAFT: &str = "BLOOMERY_DRAFT";
pub const CHECK_FINITE: &str = "BLOOMERY_CHECK_FINITE";
pub const CARD_EXPERTS: &str = "BLOOMERY_CARD_EXPERTS";
pub const STEP_PAIR: &str = "BLOOMERY_STEP_PAIR";
pub const FLASH_MMA: &str = "BLOOMERY_FLASH_MMA";

/// The rounds that convert the levers still read in place.
const R03: &str = "[03]";
const V2FENCE: &str = "v2fence";
const DECISION_4: &str = "decision 4";

const OPS: &str = "crates/model/src/ops.rs";
const ATTN: &str = "crates/model/src/arch/deepseek2/attn.rs";
const HYBRID: &str = "crates/gpu/src/hybrid.rs";
const DECODE: &str = "crates/model/src/bin/bloomery-decode.rs";

/// Every lever, in the order the tables print them: the parsed ones, the
/// ones still read in place, the retired names, the runner variables.
pub static REGISTRY: &[LeverSpec] = &[
    LeverSpec {
        name: THREADS,
        class: Class::C,
        kind: Kind::Count {
            min: 1,
            max: u64::MAX,
            trim: true,
        },
        default: Unset::Means("the physical core count"),
        doc: "Threads of the process-wide worker pool, the calling thread among them; 1 \
              runs every range inline and spawns nothing. Read when the pool is built.",
        site: Site::Parsed { left: &[] },
    },
    LeverSpec {
        name: SPIN,
        class: Class::C,
        kind: Kind::Count {
            min: 0,
            max: u64::MAX,
            trim: true,
        },
        default: Unset::Is("20000"),
        doc: "Spin iterations a waiting pool thread makes before it parks. Read when the \
              pool is built.",
        site: Site::Parsed { left: &[] },
    },
    LeverSpec {
        name: CED,
        class: Class::A,
        kind: Kind::OnOff,
        default: Unset::Is("on"),
        doc: "V4.1: a prompt call runs each layer only at the positions a later reader \
              needs (the CED triangle, `body/ced.rs`), where the file allows it; `off` runs \
              every layer at every position, the same-binary arm. The `load` line prints \
              `ced=on` or `ced=off (<reason>)`.",
        site: Site::Parsed { left: &[] },
    },
    LeverSpec {
        name: PREFILL,
        class: Class::M,
        kind: Kind::Words(&["batch", "steps"]),
        default: Unset::Is("batch"),
        doc: "V4.1: how a binary feeds a prompt, in batches (`body::prefill`) or one \
              decode step per id, the feed the batches are bit for bit equal to. The \
              `load` line prints `prefill=`.",
        site: Site::Parsed { left: &[] },
    },
    LeverSpec {
        name: PREFILL_GROUP,
        class: Class::A,
        kind: Kind::Count {
            min: 1,
            max: 8,
            trim: false,
        },
        default: Unset::Is("2"),
        doc: "V4.1: the batches a prompt group runs layer by layer, each layer-batch's \
              route enqueued ahead of the previous one's host serve; 1 runs each batch \
              alone, the same-binary arm, and both write the same bits. The `load` line \
              prints `group=`.",
        site: Site::Parsed { left: &[] },
    },
    LeverSpec {
        name: ENGRAM_HELPER,
        class: Class::A,
        kind: Kind::Flag,
        default: Unset::Is("1"),
        doc: "V4.1 `StepRows`: a helper thread reads a step's engram rows while the step \
              thread reads its embedding row; `0` has the step thread read them itself.",
        site: Site::Parsed { left: &[] },
    },
    LeverSpec {
        name: STEP_STATS,
        class: Class::D,
        kind: Kind::Flag,
        default: Unset::Is("0"),
        doc: "`generate_ds41`: a `stat step` line per generated step and a `stat summary` \
              (host-tier counters, page faults, free device bytes, engram row timings), \
              the card's time per layer-batch in `stat prefill split`, and the prompt \
              call's queue entries as each site enqueued them, per batch (`stat prefill \
              front`) and per layer-batch (`stat prefill lb`); `gate_deepseek41_prefill`: \
              that split line per case. Off, nothing is read.",
        site: Site::Parsed { left: &[] },
    },
    LeverSpec {
        name: HOT_LIST,
        class: Class::C,
        kind: Kind::Path,
        default: Unset::Means("the id prefix [0, n_l)"),
        doc: "Placement: a hot list file (`tools/ref/router-hotlist.py`); each routed \
              layer's card keeps the file's first `n_l` ranked ids instead of the id \
              prefix, the same counts and bytes. A layer listing fewer than the plan's \
              `n_l` is refused. The `plan` line prints `hot_list=`.",
        site: Site::Parsed { left: &[] },
    },
    LeverSpec {
        name: CARD_BUDGET,
        class: Class::C,
        kind: Kind::Bytes,
        default: Unset::Means("each card's own usable bytes"),
        doc: "Placement: every card of a plan plans with `min(usable, budget)` usable \
              bytes, so a large card stands in for a smaller one; a budget below a card's \
              floor is refused with each term. `M` and `G` are binary units. The `plan` \
              line prints `card_budget=`.",
        site: Site::Parsed { left: &[] },
    },
    LeverSpec {
        name: PIN_MAIN,
        class: Class::A,
        kind: Kind::Flag,
        default: Unset::Is("1"),
        doc: "A binary that owns its main thread pins it to the dispatcher's cpu slot; `0` \
              leaves it floating. The `load` line prints the ask and the outcome.",
        site: Site::Parsed {
            left: &[
                InPlace {
                    file: DECODE,
                    round: V2FENCE,
                },
                InPlace {
                    file: "crates/model/src/bin/bench_v41_host.rs",
                    round: R03,
                },
            ],
        },
    },
    LeverSpec {
        name: DRAFT,
        class: Class::M,
        kind: Kind::Words(&["lookup", "dspark"]),
        default: Unset::Means("the plain path, one token a step"),
        doc: "`generate_ds41`: `lookup` serves an n-gram lookup draft, `dspark` the DSpark \
              draft (`$BLOOMERY_DSPARK_MODEL`), through the skewed two-row pass; the \
              `tokens` line is the plain run's.",
        site: Site::Parsed { left: &[] },
    },
    LeverSpec {
        name: CHECK_FINITE,
        class: Class::D,
        kind: Kind::Flag,
        default: Unset::Is("0"),
        doc: "`generate_ds41`: every position is first stepped eagerly outside the graph \
              with each sub-layer's streams read back, then taken back and stepped through \
              the engine; a `stat finite` line per step names the first non-finite \
              `(layer, site)`. Refused beside `--time`, `BLOOMERY_DRAFT` and \
              `BLOOMERY_STEP_STATS=1`.",
        site: Site::Parsed { left: &[] },
    },
    LeverSpec {
        name: "BLOOMERY_POISON",
        class: Class::D,
        kind: Kind::Flag,
        default: Unset::Is("0"),
        doc: "Host tier: every `scratch` block starts as NaN, so a kernel that leaves a \
              cell unwritten fails the gates loudly.",
        site: Site::Direct {
            at: &[InPlace {
                file: OPS,
                round: R03,
            }],
        },
    },
    LeverSpec {
        name: "BLOOMERY_DEFER_QUANT",
        class: Class::T,
        kind: Kind::Flag,
        default: Unset::Is("1"),
        doc: "Host tier: the row dispatch quantizes its activations itself; `0` is the \
              caller-side pre-pass, the twin the union tests compare against.",
        site: Site::Direct {
            at: &[InPlace {
                file: OPS,
                round: R03,
            }],
        },
    },
    LeverSpec {
        name: "BLOOMERY_STEAL",
        class: Class::A,
        kind: Kind::Flag,
        default: Unset::Is("1"),
        doc: "Host tier: a lane steals blocks of other lanes once its own are done; `0` \
              runs whole-lane blocks on home lanes only.",
        site: Site::Direct {
            at: &[InPlace {
                file: OPS,
                round: R03,
            }],
        },
    },
    LeverSpec {
        name: "BLOOMERY_STEAL_BLOCKS",
        class: Class::A,
        kind: Kind::Count {
            min: 1,
            max: u64::MAX,
            trim: true,
        },
        default: Unset::Is("4"),
        doc: "Host tier: blocks a lane is cut into for stealing; a host-union block is at \
              most 144 rows.",
        site: Site::Direct {
            at: &[InPlace {
                file: OPS,
                round: R03,
            }],
        },
    },
    LeverSpec {
        name: "BLOOMERY_EXPERT_LOG",
        class: Class::D,
        kind: Kind::Path,
        default: Unset::Means("no log"),
        doc: "CPU MoE: each block's routed expert ids are appended to this file \
              (`tools/ref/expert-union-dump.sh`).",
        site: Site::Direct {
            at: &[InPlace {
                file: "crates/model/src/moe.rs",
                round: R03,
            }],
        },
    },
    LeverSpec {
        name: "BLOOMERY_PROFILE",
        class: Class::D,
        kind: Kind::Count {
            min: 0,
            max: 2,
            trim: true,
        },
        default: Unset::Is("0"),
        doc: "CPU engine profiler: 1 times each call, 2 adds the stage split; \
              `bloomery-decode --profile` sets 1.",
        site: Site::Direct {
            at: &[
                InPlace {
                    file: "crates/model/src/profile.rs",
                    round: V2FENCE,
                },
                InPlace {
                    file: DECODE,
                    round: V2FENCE,
                },
            ],
        },
    },
    LeverSpec {
        name: "BLOOMERY_FLASH_SIMD",
        class: Class::T,
        kind: Kind::Flag,
        default: Unset::Is("1"),
        doc: "CPU flash attention: `0` forces the scalar kernel, the twin the AVX2 kernel \
              is gated against.",
        site: Site::Direct {
            at: &[InPlace {
                file: ATTN,
                round: V2FENCE,
            }],
        },
    },
    LeverSpec {
        name: "BLOOMERY_KV_PREFETCH",
        class: Class::A,
        kind: Kind::Flag,
        default: Unset::Is("1"),
        doc: "CPU flash attention: `0` turns a decode row's key prefetch hint off.",
        site: Site::Direct {
            at: &[InPlace {
                file: ATTN,
                round: V2FENCE,
            }],
        },
    },
    LeverSpec {
        name: "BLOOMERY_KV_PREFETCH_ROWS",
        class: Class::A,
        kind: Kind::Count {
            min: 0,
            max: u64::MAX,
            trim: false,
        },
        default: Unset::Is("16"),
        doc: "CPU flash attention: key rows a decode row prefetches ahead; 0 is off.",
        site: Site::Direct {
            at: &[InPlace {
                file: ATTN,
                round: V2FENCE,
            }],
        },
    },
    LeverSpec {
        name: "BLOOMERY_FLASH_SEGMENTS",
        class: Class::T,
        kind: Kind::Count {
            min: 1,
            max: 32,
            trim: true,
        },
        default: Unset::Is("32"),
        doc: "CPU flash attention: segments a decode row's keys are cut into for split-K, \
              a function of the visible key count alone; 1 is the single-pass online \
              softmax the split is banded against.",
        site: Site::Direct {
            at: &[InPlace {
                file: ATTN,
                round: V2FENCE,
            }],
        },
    },
    LeverSpec {
        name: "BLOOMERY_ATTN_BUNDLE",
        class: Class::T,
        kind: Kind::Words(&["1", "8"]),
        default: Unset::Is("8"),
        doc: "CPU flash attention: `1` runs the per-head online segment kernel instead of \
              the 8-head bundle tile, the twin the tile is banded against.",
        site: Site::Direct {
            at: &[InPlace {
                file: ATTN,
                round: V2FENCE,
            }],
        },
    },
    LeverSpec {
        name: "BLOOMERY_ATTN_HALVES",
        class: Class::A,
        kind: Kind::Words(&["1", "2"]),
        default: Unset::Is("1"),
        doc: "CPU flash attention, multi-query rows: `2` runs each (token, head) row on two \
              threads, each accumulating half the latent; bit-identical and slower.",
        site: Site::Direct {
            at: &[InPlace {
                file: ATTN,
                round: V2FENCE,
            }],
        },
    },
    LeverSpec {
        name: "BLOOMERY_FLASH_SEG",
        class: Class::A,
        kind: Kind::Multiple { of: 32 },
        default: Unset::Is("64"),
        doc: "V2-Lite GPU flash: keys per segment of the tensor-core split pass. Read at \
              first use; the value fixes a captured graph's grid.",
        site: Site::Direct {
            at: &[InPlace {
                file: "crates/gpu/src/flash.rs",
                round: V2FENCE,
            }],
        },
    },
    LeverSpec {
        name: "BLOOMERY_GQA_MMA",
        class: Class::T,
        kind: Kind::Flag,
        default: Unset::Is("1"),
        doc: "Qwen3 GQA flash: `0` runs the scalar segment pass instead of the tensor-core \
              one, its banded twin. The `load` line prints which ran.",
        site: Site::Direct {
            at: &[InPlace {
                file: "crates/gpu/src/flash_gqa.rs",
                round: R03,
            }],
        },
    },
    LeverSpec {
        name: "BLOOMERY_Q3K_SPLIT",
        class: Class::A,
        kind: Kind::Words(&["1", "2", "4", "8"]),
        default: Unset::Is("1"),
        doc: "GPU Q3_K gemv: the split-K width of a row whose walk has at least \
              `BLOOMERY_Q3K_SPLIT_ITERS` two-super-block iterations the width divides (on \
              V4.1, `wo_b` alone); 1 is the plain kernel.",
        site: Site::Direct {
            at: &[InPlace {
                file: "crates/gpu/src/lib.rs",
                round: DECISION_4,
            }],
        },
    },
    LeverSpec {
        name: "BLOOMERY_Q3K_SPLIT_ITERS",
        class: Class::A,
        kind: Kind::Count {
            min: 1,
            max: u64::MAX,
            trim: false,
        },
        default: Unset::Is("16"),
        doc: "GPU Q3_K gemv: the walk length from which `BLOOMERY_Q3K_SPLIT` splits a row.",
        site: Site::Direct {
            at: &[InPlace {
                file: "crates/gpu/src/lib.rs",
                round: DECISION_4,
            }],
        },
    },
    LeverSpec {
        name: "BLOOMERY_HYBRID_NL",
        class: Class::C,
        kind: Kind::Count {
            min: 0,
            max: u64::MAX,
            trim: true,
        },
        default: Unset::Means("every expert on the card"),
        doc: "V2-Lite hybrid MoE: experts `[0, n_l)` of every routed stack stay on the \
              card and the rest run on the host tier inside the captured step.",
        site: Site::Direct {
            at: &[InPlace {
                file: HYBRID,
                round: R03,
            }],
        },
    },
    LeverSpec {
        name: "BLOOMERY_HYBRID_OVERLAP",
        class: Class::T,
        kind: Kind::Flag,
        default: Unset::Is("1"),
        doc: "V2-Lite hybrid MoE: `0` puts each layer's wait right after its go instead of \
              after the card's experts and the shared expert.",
        site: Site::Direct {
            at: &[InPlace {
                file: HYBRID,
                round: R03,
            }],
        },
    },
    LeverSpec {
        name: "BLOOMERY_HOST_POPULATE",
        class: Class::C,
        kind: Kind::Flag,
        default: Unset::Is("1"),
        doc: "Placed load: the plan's host set is read in with `MADV_POPULATE_READ`; `0` \
              leaves it to fault in, the fresh-fault arm.",
        site: Site::Direct {
            at: &[InPlace {
                file: HYBRID,
                round: R03,
            }],
        },
    },
    LeverSpec {
        name: "BLOOMERY_HOST_LOCK",
        class: Class::C,
        kind: Kind::Flag,
        default: Unset::Is("0"),
        doc: "Placed load: after populating, `mlock` the host set for the model's life; an \
              `RLIMIT_MEMLOCK` refusal is an error that names the limit.",
        site: Site::Direct {
            at: &[InPlace {
                file: HYBRID,
                round: R03,
            }],
        },
    },
    LeverSpec {
        name: "BLOOMERY_CARD_DONTNEED",
        class: Class::C,
        kind: Kind::Flag,
        default: Unset::Is("1"),
        doc: "Placed load: each uploaded card segment's file pages are dropped right after \
              its upload (`token_embd` and the engram table excepted); `0` keeps them in \
              the page cache.",
        site: Site::Direct {
            at: &[InPlace {
                file: HYBRID,
                round: R03,
            }],
        },
    },
    LeverSpec {
        name: "BLOOMERY_LAUNCH_THREAD",
        class: Class::A,
        kind: Kind::Flag,
        default: Unset::Is("0"),
        doc: "`GpuModel`: `1` runs each replay's `cuGraphLaunch` on a launcher thread, the \
              SMT sibling of a pinned opener's cpu, while the decode thread goes straight \
              into the first host service's wait.",
        site: Site::Direct {
            at: &[InPlace {
                file: "crates/gpu/src/model/launcher.rs",
                round: DECISION_4,
            }],
        },
    },
    LeverSpec {
        name: "BLOOMERY_QWEN3_UBATCH",
        class: Class::C,
        kind: Kind::Count {
            min: 1,
            max: 4096,
            trim: false,
        },
        default: Unset::Is("4096"),
        doc: "Qwen3 GEMM prefill: tokens per ubatch, read at load; a token's bits do not \
              depend on it. The `load` line prints `ubatch=`.",
        site: Site::Direct {
            at: &[InPlace {
                file: "crates/gpu/src/arch/qwen3moe/ubatch.rs",
                round: R03,
            }],
        },
    },
    LeverSpec {
        name: "BLOOMERY_POPULATE",
        class: Class::A,
        kind: Kind::Flag,
        default: Unset::Is("1"),
        doc: "`bloomery-decode`: the weights' mapping is populated at open; `0` keeps it \
              lazy.",
        site: Site::Direct {
            at: &[InPlace {
                file: DECODE,
                round: V2FENCE,
            }],
        },
    },
    LeverSpec {
        name: "BLOOMERY_WEIGHTS",
        class: Class::A,
        kind: Kind::Words(&["anon", "huge"]),
        default: Unset::Means("the file's own mapping"),
        doc: "`bloomery-decode`: the weights are copied into anonymous memory, 4 KiB pages \
              (`anon`) or transparent 2 MiB pages (`huge`), instead of running off the \
              page cache.",
        site: Site::Direct {
            at: &[InPlace {
                file: DECODE,
                round: V2FENCE,
            }],
        },
    },
    LeverSpec {
        name: CARD_EXPERTS,
        class: Class::A,
        kind: Kind::Words(&["tile", "expert", "slot"]),
        default: Unset::Means("the only path"),
        doc: "Was the V4.1 prompt batch's card expert walk.",
        site: Site::Retired {
            why: "a V4.1 prompt batch runs its card experts one way, by tiles",
            left: &[],
        },
    },
    LeverSpec {
        name: STEP_PAIR,
        class: Class::A,
        kind: Kind::Flag,
        default: Unset::Means("the only path"),
        doc: "Was `generate_ds41`'s forced two-row step.",
        site: Site::Retired {
            why: "`generate_ds41` runs the pair pass under `BLOOMERY_DRAFT` only",
            left: &[],
        },
    },
    LeverSpec {
        name: FLASH_MMA,
        class: Class::T,
        kind: Kind::Flag,
        default: Unset::Means("the only path"),
        doc: "Was the V2-Lite GPU flash's scalar segment pass.",
        site: Site::Retired {
            why: "the tensor-core segment pass is the only V2-Lite segment pass",
            left: &[InPlace {
                file: "crates/gpu/src/flash.rs",
                round: V2FENCE,
            }],
        },
    },
    LeverSpec {
        name: "BLOOMERY_GATE_BOUND",
        class: Class::C,
        kind: Kind::Count {
            min: 1,
            max: u64::MAX,
            trim: false,
        },
        default: Unset::Is("900"),
        doc: "Seconds a gate runner (`tools/gate.sh`, `tools/gpu-gate.sh`, \
              `tools/host-gate.sh`) lets its binary run before it kills it: a hung gate \
              ends red.",
        site: Site::Runner {
            file: "tools/gate.sh",
        },
    },
    LeverSpec {
        name: "BLOOMERY_PROFILE_DEPTH",
        class: Class::D,
        kind: Kind::Count {
            min: 0,
            max: u64::MAX,
            trim: false,
        },
        default: Unset::Means("no prompt first"),
        doc: "`tools/ref/profile-measure.sh`: the profiled run first prefills depth-decode's \
              prompt of this depth.",
        site: Site::Runner {
            file: "tools/ref/profile-measure.sh",
        },
    },
];
