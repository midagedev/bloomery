//! The rows: every `BLOOMERY_*` name the repository sets or reads — the
//! levers, the retired names, and the names that are no lever — each once.
//! A lever row with a file that reads it in place (`Site::Direct`, or a
//! non-empty `left`) is also a line of `tools/levers-direct.txt`, the list
//! `tools/check-levers.sh` holds every in-place environment read to; the test
//! `registry_and_allow_list_agree` holds the two to each other. A name that is
//! no lever names its owner (`Site::Env`), and `tools/check-levers.sh` holds
//! every `BLOOMERY_*` name under `tools/`, in the justfile and in `.cargo/` to
//! a row.

use crate::{Class, HostLanes, InPlace, Kind, LeverSpec, Phase, Readers, Site, Unset};

pub const THREADS: &str = "BLOOMERY_THREADS";
pub const SPIN: &str = "BLOOMERY_SPIN";
pub const CED: &str = "BLOOMERY_CED";
pub const PREFILL: &str = "BLOOMERY_PREFILL";
pub const PREFILL_GROUP: &str = "BLOOMERY_PREFILL_GROUP";
pub const ENGRAM_HELPER: &str = "BLOOMERY_ENGRAM_HELPER";
pub const STEP_STATS: &str = "BLOOMERY_STEP_STATS";
pub const CARD_BUDGET: &str = "BLOOMERY_CARD_BUDGET";
pub const CARD_MARGIN: &str = "BLOOMERY_CARD_MARGIN";
pub const PIN_MAIN: &str = "BLOOMERY_PIN_MAIN";
pub const DRAFT: &str = "BLOOMERY_DRAFT";
pub const CHECK_FINITE: &str = "BLOOMERY_CHECK_FINITE";
pub const HOST_POPULATE: &str = "BLOOMERY_HOST_POPULATE";
pub const HOST_LOCK: &str = "BLOOMERY_HOST_LOCK";
pub const NEED_HOOK: &str = "BLOOMERY_NEED_HOOK";
pub const CARD_DONTNEED: &str = "BLOOMERY_CARD_DONTNEED";
pub const R8: &str = "BLOOMERY_R8";
pub const MTP_HEAD_ROWS: &str = "BLOOMERY_MTP_HEAD_ROWS";
pub const MTP_DRAFT: &str = "BLOOMERY_MTP_DRAFT";
pub const MTP_WINDOWS: &str = "BLOOMERY_MTP_WINDOWS";
pub const MTP_WIDTH: &str = "BLOOMERY_MTP_WIDTH";
pub const RESIDENCY: &str = "BLOOMERY_RESIDENCY";
pub const METAL_BREADCRUMB: &str = "BLOOMERY_METAL_BREADCRUMB";
pub const HOSTSTREAM: &str = "BLOOMERY_HOSTSTREAM";
pub const XSTREAM: &str = "BLOOMERY_XSTREAM";
pub const SPLIT_WALK: &str = "BLOOMERY_SPLIT_WALK";
pub const ROUTE_TRACE: &str = "BLOOMERY_ROUTE_TRACE";
pub const WARM_PROFILE: &str = "BLOOMERY_WARM_PROFILE";
pub const QWEN38_EXPERTS: &str = "BLOOMERY_QWEN38_EXPERTS";
pub const QWEN3_KV: &str = "BLOOMERY_QWEN3_KV";
pub const LANE_PREFETCH: &str = "BLOOMERY_LANE_PREFETCH";
pub const HOST_LANES: &str = "BLOOMERY_HOST_LANES";
pub const GEN_SLOTS: &str = "BLOOMERY_GEN_SLOTS";
pub const FORCE_IDS: &str = "BLOOMERY_FORCE_IDS";
pub const HOST_ROOM: &str = "BLOOMERY_HOST_ROOM";
pub const NVTIER_BYTES: &str = "BLOOMERY_NVTIER_BYTES";
pub const NVTIER_READ: &str = "BLOOMERY_NVTIER_READ";
pub const NVTIER_MEM: &str = "BLOOMERY_NVTIER_MEM";
pub const NVTIER_EVICT: &str = "BLOOMERY_NVTIER_EVICT";
pub const TIER_LINK: &str = "BLOOMERY_TIER_LINK";
pub const TIER_NODES: &str = "BLOOMERY_TIER_NODES";
pub const TP_NODES: &str = "BLOOMERY_TP_NODES";
pub const SCREEN: &str = "BLOOMERY_SCREEN";
pub const SCHEMAJUMP: &str = "BLOOMERY_SCHEMAJUMP";
pub const PREPARE_Q8: &str = "BLOOMERY_PREPARE_Q8";

/// The largest `BLOOMERY_PREFILL_GROUP`: the batches a V4.1 prompt group
/// holds at most, which the body's buffers are sized for.
pub const PREFILL_GROUP_MAX: u64 = 8;

/// `BLOOMERY_PREFILL_GROUP` unset: the row's default, the group GLM-5.3's
/// plan reserves a prompt batch's units for and MiMo-V2.6-Flash's load runs.
pub const PREFILL_GROUP_DEFAULT: u64 = 2;

/// `BLOOMERY_LANE_PREFETCH` unset: the row's default, which the host
/// union holds in a binary that does not act on the lever
/// (`model::ops::lane_prefetch`).
pub const LANE_PREFETCH_DEFAULT: bool = false;

/// `BLOOMERY_HOST_LANES` unset: the row's default, which the host tier's row
/// dispatch holds in a binary that does not act on the lever
/// (`model::ops::Lanes::host_tier`).
pub const HOST_LANES_DEFAULT: HostLanes = HostLanes::Ccd;

/// The largest `BLOOMERY_GEN_SLOTS`: one row a slot in a pass of several
/// slots, which holds at most the pass's eight rows.
pub const GEN_SLOTS_MAX: u64 = 8;

/// The rounds that convert the levers still read in place.
const R03: &str = "[03]";
const V2FENCE: &str = "v2fence";
const SESSION: &str = "session";
const ISA_OWNER: &str = "isa-owner";

const OPS: &str = "crates/model/src/ops.rs";
const METAL_STREAM: &str = "crates/metal/src/stream.rs";
const METAL_BREADCRUMB_STAYS: &str = "stays: the Metal crash breadcrumb";
const ATTN: &str = "crates/model/src/arch/deepseek2/attn.rs";
const HYBRID: &str = "crates/gpu/src/hybrid.rs";
const DECODE: &str = "crates/model/src/bin/bloomery-decode.rs";
const QDOT: &str = "crates/qdot/src/lib.rs";

/// What an unset name that is no lever means.
const OWNERS: Unset = Unset::Means("its owner's default");

/// The row of a path: a name that is no lever and says where a file or a
/// directory is, owned by `script` under `tools/`, or by a harness.
const fn path(name: &'static str, script: Option<&'static str>, doc: &'static str) -> LeverSpec {
    LeverSpec {
        name,
        class: Class::P,
        kind: Kind::Path,
        default: OWNERS,
        doc,
        site: Site::Env { script },
    }
}

/// The row of a runner's or a harness's own variable: a name that is no
/// lever, owned by `script` under `tools/`, or by a harness.
const fn runner(name: &'static str, script: Option<&'static str>, doc: &'static str) -> LeverSpec {
    LeverSpec {
        name,
        class: Class::R,
        kind: Kind::Text,
        default: OWNERS,
        doc,
        site: Site::Env { script },
    }
}

/// Every name, in the order the tables print them: the parsed levers, the
/// ones still read in place, the retired names, the paths, the runners'
/// variables.
pub(crate) static REGISTRY: &[LeverSpec] = &[
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
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
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
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
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
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: PREFILL,
        class: Class::M,
        kind: Kind::Words(&["batch", "steps"]),
        default: Unset::Is("batch"),
        doc: "V4.1: how a binary feeds a prompt, in batches (`body::prefill`) or one \
              decode step per id, the feed the batches are bit for bit equal to. The \
              `load` line prints `prefill=`. `bloomery-serve-qwen38`: `batch` is the \
              seat's `auto` path (passes below nine ids, ubatches from nine and for a \
              call that starts or ends on the 4,096-position checkpoint grid), and `steps` \
              one captured step per id (`prefill=step`); a prompt of nine ids or more \
              leaves other bits under the ubatch walk than under steps. Steps are \
              refused by name beside the MTP draft and a `BLOOMERY_RESIDENCY` word other \
              than `off` (unset resolves `off`).",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: PREFILL_GROUP,
        class: Class::A,
        kind: Kind::Count {
            min: 1,
            max: PREFILL_GROUP_MAX,
            trim: false,
        },
        default: Unset::Is("2"),
        doc: "V4.1, GLM-5.3 and MiMo-V2.6-Flash: the batches a prompt group runs layer by \
              layer, each layer-batch's route enqueued ahead of the previous one's host \
              serve; 1 runs each batch alone, the same-binary arm, and both write the same \
              bits. The `load` line prints `group=`. GLM-5.3's plan reserves the units a \
              group of 2 holds (each about two stream buffers); a group past 2 takes its \
              further units out of the card's margin, refused by name past its free bytes; \
              the `prefill units` line prints what they took and what stayed free. \
              MiMo-V2.6-Flash's plan reserves none: every unit of its group, the default's \
              too, comes out of the card's margin and is refused by name past its free \
              bytes. GLM-5.3 and MiMo-V2.6-Flash change it between calls through \
              `set_prefill_group`; V4.1 fixes it at load, so the row's phase is `load` until \
              every family carries a setter.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: ENGRAM_HELPER,
        class: Class::A,
        kind: Kind::Flag,
        default: Unset::Is("1"),
        doc: "V4.1 `StepRows`: a helper thread reads a step's engram rows while the step \
              thread reads its embedding row; `0` has the step thread read them itself.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
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
              that split line per case; `generate_qwen3moe` on a qwen4exp file: the \
              ubatch walk's `stat prompt split` and its per layer-batch `stat \
              prompt lb`; the qwen3 seat of `bloomery-serve`: a `slots round` record \
              a round of several slots (its command, rows, the passes the seat ran \
              them as, the slots it serves). Off, nothing is read.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: "BLOOMERY_HOT_LIST",
        class: Class::C,
        kind: Kind::Path,
        default: Unset::Means("the only path"),
        doc: "Was a placement file that ranked each routed layer's card experts by router \
              frequency, learned from the test corpora, in place of the id prefix.",
        site: Site::Retired {
            why: "each routed layer's card keeps the id prefix `[0, n_l)` and adaptive residency \
                  (`BLOOMERY_RESIDENCY`) moves experts at run time; a list learned from the test \
                  corpora ranked them in-sample",
            left: &[],
        },
    },
    LeverSpec {
        name: CARD_BUDGET,
        class: Class::C,
        kind: Kind::CardBytes,
        default: Unset::Means(
            "each card's own usable bytes, or on a unified pool the budget that leaves the NVMe \
             tier's floor",
        ),
        doc: "Placement: a card of a plan plans with `min(usable, budget)` usable bytes, so a \
              large card stands in for a smaller one. One value is every card's budget; \
              `CUDA0=38G,CUDA1=20G` gives each card named its own, by the CUDA ordinal \
              `--list-devices` prints, and a card it does not name plans with its own usable \
              bytes. `CUDA0=3x,CUDA1=1x` gives each card named a share instead: the plan \
              holds the routed experts on the cards in that proportion, as many as the tightest \
              card's room lets it, and the host tier the rest, each card's budget its floor \
              (dense tensors, cache, context, scratch, reserves, margin) and its share; a list \
              is all bytes or all shares. The cards are still `--place`'s (unset: the common \
              rule's), each planned on its budget. Refused by name: an entry that is no \
              `CUDA<N>=<bytes>` or `CUDA<N>=<n>x`, a card named twice, a card the census does \
              not list, a list on a plan whose card came from no census or a tier node's, a \
              budget below a card's floor (with each term), and shares on a plan that places no \
              routed expert on a card, that name none of its cards, or whose named cards cannot \
              hold their floor. `M` and `G` are binary units. The `plan` line prints \
              `card_budget=` in the bytes the plan resolved.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: CARD_MARGIN,
        class: Class::C,
        kind: Kind::CardBytes,
        default: Unset::Means("each card's own margin, 1 GiB on this workstation's plans"),
        doc: "Placement: the bytes a card of a plan keeps free past its context and scratch, \
              which the expert rule never plans into (llama-server's `-fitt`). One value is \
              every card's margin; `CUDA0=1G,CUDA1=2G` gives each card named its own, by the \
              CUDA ordinal `--list-devices` prints, and a card it does not name keeps its own. \
              It replaces the plan's margin and is not a budget: the card's usable bytes stay \
              the budget's. Read where a `--place` is resolved against the census, so every \
              binary acts on it. Refused by name: an entry that is no `CUDA<N>=<bytes>`, a \
              card named twice, a card the census does not list, shares (`<n>x`), a list on a \
              plan whose tier cards are another node's, and a margin that leaves a card no \
              usable byte. `M` and `G` are binary units.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: HOST_ROOM,
        class: Class::C,
        kind: Kind::Bytes,
        default: Unset::Means("the host's available bytes, read from this machine"),
        doc: "Placement: the host's room in bytes (`M` and `G` binary units), read as given \
              instead of from `/proc/meminfo` and the cgroup limits — an emulation of a \
              smaller host and an override. It reaches every reader of the host's available \
              bytes: the placed loads' host-need check, the residency rules' room and the \
              plans' tiers, where the routed experts a room cannot hold are planned on the \
              NVMe tier; a room under that tier's floor is refused with the room, the floor \
              and the need. The MiMo plan reads it too and serves no NVMe expert tier, \
              so it refuses a room that cannot hold its host experts, naming the room and \
              the need, and its default context is the card's deepest, the floor where the \
              room cannot hold that one. On a device that allocates from host memory it is \
              also the device's free bytes, which the census and the load-time allocation \
              fits read.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: NVTIER_BYTES,
        class: Class::C,
        kind: Kind::Bytes,
        default: Unset::Means(
            "the split dial's default: the largest arena the room leaves above the NVMe tier's \
             floor when the room must move over half the host leg's bytes, else no arena (R1's \
             split)",
        ),
        doc: "Placement: the RAM arena the NVMe expert tier fills for the routed experts a \
              room cannot hold, in bytes (`M` and `G` binary units). The arena's bytes come out \
              of the host segments — the split leaves the room less them — and a value that \
              pushes the host segments under the tier's floor is refused by name with the room, \
              the floor and the value. A room the plan's host need already holds plans no \
              arena. Read where the plan is made, beside `BLOOMERY_HOST_ROOM`.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: NVTIER_READ,
        class: Class::C,
        kind: Kind::Words(&["direct", "buffered"]),
        default: Unset::Is("direct"),
        doc: "The NVMe expert tier's reads: `direct` opens the shards with `O_DIRECT` and \
              proves the mount with one probe read at load — a filesystem without direct IO \
              (a Windows mount under WSL2) is refused by name; `buffered` opts into reads the \
              page cache serves, exact at the same bytes and counted by the tier's stats as \
              buffered — the fallback the probe's refusal names, not a preference. Read where \
              the tier is built.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: NVTIER_MEM,
        class: Class::A,
        kind: Kind::Words(&["auto", "anon", "pinned"]),
        default: Unset::Is("auto"),
        doc: "The memory the NVMe expert tier's RAM arena is: `anon` an anonymous mapping, whose \
              slots a landing gives back to the kernel; `pinned` one `cuMemHostAlloc` the card \
              maps, priced at load and never given back, write-combined on an integrated device \
              alone (a discrete card's CPU reads write-combined pages uncached); a step's ids in \
              its slots are read on the card by the step port's card leg where the family \
              attaches it (GLM-5.3-Flash with no residency machine), which quantizes the \
              activation as q8_1 and sums in its own order, and by the host leg as `anon`'s are \
              elsewhere (a prompt's batch, every other load); `auto` is `pinned` on an \
              integrated device and `anon` on a card with its own memory. `anon` and `pinned` are \
              the same-binary arms. Read where the tier is built.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: NVTIER_EVICT,
        class: Class::A,
        kind: Kind::Words(&["tick", "heat"]),
        default: Unset::Is("tick"),
        doc: "Which filled slot of the NVMe expert tier's RAM arena a miss takes once no slot is \
              free. Both words take a prompt's transients first, oldest first. `tick` then takes \
              the least recently used hot slot. `heat` takes the hot slot whose expert has the \
              least decayed routing count, the least recently used among equal counts: the \
              tier counts per layer and expert the columns that route it, the prompt's at its \
              union call and a step's at its claim, halved every 32 columns the layer serves. \
              The two words read and compute the same bytes, so a token is the same under \
              either; only the fills differ. `tick` and `heat` are the same-binary arms. Read \
              where the tier is built.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: TIER_LINK,
        class: Class::M,
        kind: Kind::Words(&["shared", "loopback", "rdma"]),
        default: Unset::Is("shared"),
        doc: "How the expert tier cards' page crosses between the stage card and the tier \
              cards. `shared` is one host-mapped page every card reaches. `loopback` gives the \
              tier cards a page of their own and a host pump thread that carries each go and \
              done between the two pages as a generation, with the bytes it publishes; the \
              cards' operations, graphs and bits are the shared page's. It checks the relay on \
              one machine (`weekly-gpu-ds41-lost` runs the V4.1 tier gate over it). `rdma` is \
              the loopback's two pages, each held by its own end with its own pump thread and \
              queue pair, the posts sent as RDMA writes between two queue pairs on the \
              machine's first RoCE v2 port for local tiers, or on the control socket's local \
              IPv4 RoCE port for node tiers. It needs `libibverbs.so.1` and an active RoCE port. \
              Each pump spins on one core while the load lives. Read where the host tier takes \
              its tier cards.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: TIER_NODES,
        class: Class::C,
        kind: Kind::Addrs,
        default: Unset::Means("every expert tier card is a card of the loading process"),
        doc: "Where a placed load's expert tiers run: the address of each tier's node, in the \
              plan's tier order, one a tier, where a `tier_server` process serves that tier's \
              card over its control channel. A loopback node can map the stage's file under \
              `/dev/shm`. A private or 198.18.0.0/15 IPv4 node off this host requires `BLOOMERY_TIER_LINK=rdma`: \
              each process holds its own page and exchanges its endpoint on the channel. The \
              channel has no authentication and belongs on a trusted network. The plan takes \
              each tier card from its node's census. \
              Qwen3.8's load opens a tier on a node; the other families refuse one by name. \
              Read where a load takes its tier cards.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: TP_NODES,
        class: Class::C,
        kind: Kind::Addrs,
        default: Unset::Means("a `--place tp` run is refused: it has no other rank"),
        doc: "The nodes of a tensor-parallel run (`--place tp`) past this process, which is rank 0: \
              the address of each other rank's node, in rank order, where a `tier_server` \
              process answers that node's census over its control channel. Each rank plans \
              against its own node's pool. A private or 198.18.0.0/15 IPv4 node off this host \
              requires `BLOOMERY_TIER_LINK=rdma`, and `BLOOMERY_TIER_NODES` is not set beside \
              it. GLM-5.3-Flash's `--place tp --plan` reads it. Read where a load takes its \
              placement.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
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
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: DRAFT,
        class: Class::M,
        kind: Kind::Words(&["lookup", "dspark", "mtp", "off"]),
        default: Unset::Means(
            "the plain path, one token a step; `generate_qwen3moe` on a qwen4exp file and \
             `bloomery-serve-qwen38` under `--place a`: `mtp` when the MTP draft file is there \
             (on the server, on one slot), else the plain path with a `load draft=off` record \
             naming why; the GLM seat of \
             `bloomery-serve`: `mtp` under `--place a` or `bp` on a file of one next-token layer, else the \
             plain path, with a `draft unset` record naming why (`bloomery_levers::glm_unset`)",
        ),
        doc: "`generate_ds41` and `bloomery-serve-ds41`: `lookup` serves an n-gram lookup \
              draft, `dspark` the DSpark draft (`$BLOOMERY_DSPARK_MODEL`), through the skewed \
              two-row pass; the qwen3 seat of `bloomery-serve` on a \
              qwen35moe or qwen35 file: `dspark` serves the HF DSpark draft \
              (`$BLOOMERY_DSPARK_MODEL`) on a whole-card load of one slot, and `lookup` serves \
              the n-gram lookup draft on the same whole-card verifying load, one context a \
              slot, while `dspark` on a qwen3moe file and on a placed load, `lookup` on a \
              qwen3moe file and on a placed load, and several slots under `dspark` are refused \
              by name, and the seat (not the CLI) takes `off` as the plain path; \
              `generate_qwen3moe` on a \
              qwen35moe file: `dspark` serves the HF DSpark draft (`$BLOOMERY_DSPARK_MODEL`) \
              on a whole-card load of one slot, while `lookup`, `dspark` on a qwen3moe file and \
              on a placed load, and several slots under it, are refused by name, and the seat \
              (not the CLI) takes `off` as the plain path; `generate_qwen3moe` and \
              `bloomery-serve-qwen38` on a qwen4exp \
              file: `mtp` serves the file's MTP draft (`BLOOMERY_MTP_DRAFT`, else the shared \
              draft file beside the target, its head reduced under `BLOOMERY_MTP_HEAD_ROWS`) \
              through a four-row window; `generate_glm5next` and the GLM seat of `bloomery-serve` on a \
              glm5next file: `mtp` serves the target file's NextN layer as the MTP draft through a \
              two-row window, and `lookup` and `dspark` are refused by name; the greedy ids are the \
              plain run's. A server drafts a request that samples or bans an id on a seat that \
              drafts sampled requests (the Qwen3.8 and GLM seats), alone or beside other busy \
              slots, each kept id its draw from its verified row, and serves it through plain \
              steps otherwise, each step's logits row the target's. `off` is the plain path on a \
              qwen4exp file, the same-binary arm of the unset draft. Unset in \
              `generate_qwen3moe` on a qwen4exp file follows the placement \
              (`bloomery_levers::draft38_unset`): under `--place a` the MTP draft runs when a \
              regular file is where it would be opened (`BLOOMERY_MTP_DRAFT`, else the shared \
              draft beside the target, else the family's path); the plain path runs under \
              `--place gate`, beside `--logits` or `BLOOMERY_ROUTE_TRACE`, with no file there, \
              and when an arm's last four-row window would pass `--ctx` (depth + n + 2 \
              positions) — each printed as a `load draft=off (<why>)` record after the `load` \
              line (`no file at <path>[, <path>…]`, every place the pick looked), never a refusal; with the file \
              there, a target whose `token_embd` or `output` is not Q8_0, the format the draft \
              reads them in, is refused by name before the load. \
              `bloomery-serve-qwen38` follows the same rule, its conditions the seat's: under \
              `--place a` the draft when its file is there; the plain path under `--place gate`, \
              with no file there, with stores too short for one window (`--ctx-size` under \
              5: the server takes a window only while its rows fit), and when the target's \
              `token_embd` or `output` is not Q8_0, the format the draft reads them in (a set \
              `mtp` is refused there by name before the load), with the same record; it \
              has no `--logits` (a request that reads the logits row reads each kept row of \
              its pass) and takes \
              no route trace. The GLM seat of `bloomery-serve` takes `mtp` and `off`; unset \
              (`bloomery_levers::glm_unset`, a `draft unset` record before the plan) it drafts \
              under `--place a` or `bp` on a file of one next-token layer, and runs the plain path under \
              `--place gate`, on any other file and with stores too short for one window (`--ctx` \
              under 3), never a refusal. Drafting, the load carries the NextN layer as \
              `generate_glm5next`'s does, prints a `load draft=mtp` record after the `load` line, \
              and `/props`' `engine.draft` names it; the plain path prints `load draft=off \
              (<why>)`; `mtp` set is refused by name with stores too short for one window. The \
              raon1 seat of `bloomery-serve` takes `dspark` and `off`: `dspark` serves the DFlash2 \
              draft (`$BLOOMERY_DSPARK_MODEL`, else the Q8_0 file of the draft's fixed name \
              beside the target) and drafts every greedy request, the draft's file missing refused by name before the \
              plan; `off` is the plain path; unset opens the draft when its file is there and \
              drafts a request whose prompt call leaves the model inside its think span (the \
              template's thinking-on prompt) while the draft has followed the slot since its \
              reset, each other request running plain steps with no draft work, and runs the \
              plain path when no file is there or the vocabulary holds no think tags; the \
              `draft rule` record before the plan names which and why; `lookup` and `mtp` are \
              refused by name. Every other binary and \
              family refuses each word by name, `off` included (the qwen3 and raon1 seats' apart).",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: MTP_HEAD_ROWS,
        class: Class::C,
        kind: Kind::PathOr(&["full"]),
        default: Unset::Means(
            "the list the model crate ships (Qwen3.8's Korean, Chinese and Japanese list, 76,000 \
             rows) when the target's tokenizer is the one its first line names; the full head on \
             any other target, the load's `mtp head` record saying why",
        ),
        doc: "Qwen3.8's MTP draft: the head its walks score with. Unset is the shipped row list \
              (`crates/model/data/`, built in, its bytes pinned beside it: a list whose bytes \
              are not the pinned ones is refused by name); `full` is every token of the \
              vocabulary; any other value is the path of a row list (`tools/ref/draft-vocab.py`; \
              a file named `full` is `./full`). A list is the vocabulary ids its head scores, \
              read in place from the target's `output` through the head's map; its first line \
              names the list's vocabulary and the target tokenizer's digest. A given list of \
              another tokenizer, unsorted, with an id repeated or past the vocabulary, or of no \
              row is refused by name; the shipped list on a target of another tokenizer is the \
              full head, said in the `mtp head` record. The load's plan is the full head's, \
              byte for byte (the map is one word a vocabulary id either way), so the target is \
              the full head's target, and a drafted run's ids are its plain run's under any \
              head: the list moves only the drafts' acceptance, and the emitted tokens are the \
              full head's. One exception: a running residency (`BLOOMERY_RESIDENCY`) counts by \
              pass, so a list that moves the acceptance moves the passes, the experts' flips \
              with them, and the target's bits at a near tie.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: MTP_DRAFT,
        class: Class::C,
        kind: Kind::File,
        default: Unset::Means(
            "the shared draft file beside the target, by the name the qwen4exp MTP family's \
             path gives it; with no such file there, that path",
        ),
        doc: "Qwen3.8's MTP draft, when the run drafts (`BLOOMERY_DRAFT=mtp`, or unset in \
              `generate_qwen3moe` and `bloomery-serve-qwen38` under `--place a`): the draft file \
              `generate_qwen3moe` and `bloomery-serve-qwen38` open (and `gate_qwen38_serve`, \
              which starts that server). A path with no regular file is refused at `main`, and \
              the lever set on a run that drafts nothing by each of them, naming why. The MTP \
              draft gate opens the file its reference set states and does not act on it.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: MTP_WINDOWS,
        class: Class::D,
        kind: Kind::Flag,
        default: Unset::Is("0"),
        doc: "`generate_qwen3moe` drafting a qwen4exp file (`BLOOMERY_DRAFT=mtp`, or unset \
              under `--place a`): an `mtp window` record a drafted window after the arm's `mtp \
              summary` — its pass, the target's position before it, the proposal's ids, each \
              one's probability among the draft head's rows (from the chain's one readback, \
              which holds them either way) and how many the target kept. Refused by name on a \
              run that drafts nothing. Off, nothing is kept; the tokens are the same either \
              way.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: MTP_WIDTH,
        class: Class::A,
        kind: Kind::Words(&["cost", "fixed"]),
        default: Unset::Means("`cost` on a run that drafts"),
        doc: "A run that drafts (the MTP draft of `generate_qwen3moe`, `generate_glm5next` and \
              their serve seats; the lookup and DSpark drafts of `generate_ds41` and \
              `bloomery-serve-ds41`; the lookup and DSpark drafts of `bloomery-serve`'s qwen3 \
              seat; the DFlash2 draft of `bloomery-serve`'s raon1 seat): how \
              many of a proposal's ids a pass verifies. `cost` \
              (unset) puts the draft behind the width chooser (`runtime::width`): each window \
              verifies the width whose expected kept positions over its measured pass wall is \
              highest, and none while no width beats the plain step by its margin. A \
              single-sequence pass can cut a full proposal to a prefix. Batched MTP rounds \
              price all active slots together, choose proposal depths before drafting and \
              pack complete windows within the body's row bounds. Widths never change a \
              generated token. `fixed` verifies every proposal whole, the same-binary arm. A drafting \
              server prints an `mtp width` record a request under `cost`. Refused by name on a \
              run that drafts nothing. Read once at `main`, the policy every window of the \
              process runs by; no setter changes it between calls.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: WARM_PROFILE,
        class: Class::A,
        kind: Kind::Path,
        default: Unset::Means("no profile: the residency starts from its placement's seed"),
        doc: "`generate_ds41`: the residency's persisted usage profile \
              (`crates/runtime/src/warmprof.rs`, `crates/app/src/warm.rs`), keyed to the model's \
              first shard (path, length, modification time), the placement word and the \
              cards. Read at the load's end, before the first prompt: the residency is \
              reseeded to the profile's rankings (each layer's pinned experts stay, then its \
              most used), which every later reset returns to, and a `warm load` record names \
              the outcome. A missing file, one counted under another key and a corrupt one \
              leave the placement's seed, the record naming why; a file that cannot be read \
              for another reason is refused by name. `--warm-save` also folds the run's \
              usage into the profile at its end (a `warm save` record); without it the file \
              is only read, so the arms of an A/B read one profile. Refused by name on a run \
              with no residency machine.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: ROUTE_TRACE,
        class: Class::D,
        kind: Kind::Path,
        default: Unset::Means("no trace"),
        doc: "`bloomery-serve-ds41`, `bloomery-serve-qwen38`, `generate_qwen3moe` (a \
              qwen4exp file) and `generate_glm5next`: the directory, created at `main` as \
              a new directory (an existing path or a missing parent is refused by name), \
              the host tier writes a route trace into \
              (`crates/gpu/src/host/route_trace.rs`): every position's routed ids per \
              layer and the slot each ran in, as a router set, a `call` row per prompt \
              call. On a qwen4exp or glm5next load it keeps every layer's host leg, where \
              the trace is written; a layer that holds every expert on the card keeps no host \
              view, so a load of experts no host kernel reads takes a trace. The \
              servers need `BLOOMERY_PREFILL=steps`, no `BLOOMERY_DRAFT` set, \
              and one resident slot (`--parallel` past 1 is refused; unset serves one); \
              `bloomery-serve-qwen38` also refuses a `BLOOMERY_RESIDENCY` word other than \
              `off`, and its unset draft and residency resolve `off`. The generators \
              need their step feed (`--prefill step` / `--prefill steps`) and refuse \
              `--time`, each refused by name otherwise.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
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
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: HOST_POPULATE,
        class: Class::C,
        kind: Kind::Flag,
        default: Unset::Is("1"),
        doc: "Placed load: the plan's host set is read in with `MADV_POPULATE_READ`; `0` \
              leaves it to fault in, the fresh-fault arm.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: HOST_LOCK,
        class: Class::C,
        kind: Kind::Flag,
        default: Unset::Is("0"),
        doc: "Placed load: after populating, `mlock` the host set for the model's life; an \
              `RLIMIT_MEMLOCK` refusal is an error that names the limit.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: NEED_HOOK,
        class: Class::C,
        kind: Kind::Program,
        default: Unset::Means("no hook: nothing runs"),
        doc: "Placed load: an executable called once per load, after the plan and before the \
              card is opened or the host set read in, with the load's `need` record on its \
              standard input: the line the process printed at plan time (host bytes by page \
              class, the prompt cache, each card's bytes, the pool's threads and cpus). Exit \
              0 goes on; 75 ends the process with 75 naming the hook's last output line; any \
              other status or a hook that cannot start ends the load with a named error. A \
              path that is not absolute, or has no regular file, is refused at `main`. A load \
              that opens a whole model on one card with no plan states no need and consults \
              nothing, saying so on stderr. Read below every binary's `main`, like the host's \
              room.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: CARD_DONTNEED,
        class: Class::C,
        kind: Kind::Flag,
        default: Unset::Is("1"),
        doc: "Placed load: each uploaded card segment's file pages are dropped right after \
              its upload (`token_embd` and the engram table excepted); `0` keeps them in \
              the page cache.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: R8,
        class: Class::A,
        kind: Kind::OnOff,
        default: Unset::Is("on"),
        doc: "V4.1 host tier: its routed gates and ups are read from the r8 sidecar at \
              `r8file::sidecar_path` of the first shard (`just r8-sidecar` writes it) through \
              the row-lane tile, and the load's host set populates and locks those stacks' \
              pages in the sidecar instead of the source; no file there reads the source; a \
              file that does not match the source is its named `R8Error`, never a fall-back. \
              `off` reads the source, the same-binary arm; both write the same bits. The load \
              prints `load host_tier r8=on (<path>)`, `r8=off (BLOOMERY_R8=off)` or `r8=off \
              (no sidecar at <path>: just r8-sidecar)`.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: RESIDENCY,
        class: Class::C,
        kind: Kind::Residency,
        default: Unset::Means(
            "V4.1 follows the placement and the plan: `mid-p0-s1` under `--place a` and `bp` \
             where the plan's host takes its churn pool, the first P whose pool fits \
             (`mid-p<P>-s1`) where the plan's host headroom or `MemAvailable` does not, `off` \
             where the card slots leave no room for the word or no churn pool the card side \
             leaves fits; `off` under `gate`, beside \
             `BLOOMERY_CHECK_FINITE=1`, `BLOOMERY_ROUTE_TRACE` or `BLOOMERY_PREFILL=steps`; \
             `generate_qwen3moe` on a qwen4exp file follows its plan: \
             `mid-p<P>-s1` under `--place a`, P 0 where the plan's host takes the churn pool \
             and the first P whose pool fits where it does not, `off` under `--place gate`, \
             beside `BLOOMERY_ROUTE_TRACE`, `--prefill step` or `--dump-taps`, on a qwen3moe or \
             qwen35moe file, and on plan (a) when the plan \
             leaves no room for the word, or no churn pool the card side leaves fits its host \
             headroom or `MemAvailable`; `bloomery-serve-qwen38` by the same rule under \
             `--place a` and `off` under `gate`; the GLM seat of `bloomery-serve` `mid-p0-s1` \
             under `--place a` or `bp` where the plan has room, `off` under `gate` \
             (`bloomery_levers::glm_unset`, `glm_residency_at_plan`), with a `residency unset` \
             record; `off` in every other binary",
        ),
        doc: "Adaptive expert residency (`host::swap`): `off` keeps the load's slot map for the \
              model's life; `mid-p<P>-s<S>` runs the residency rule's `mid` parameters over the \
              stage card's routed stacks, the first P seed experts of each layer never a victim \
              and S slots a layer freed for flips in flight. The rule plans every 4 kept rows, \
              not every 4 passes, so a drafted pass of several kept rows swaps at the same \
              kept-token rate a plain one does; the rows past 4 carry to the next plan. Its cap \
              is a churn guard, not link-sized: the copies one planning pass issues are a few \
              percent of what the card's link and the host's staging land inside the live \
              delay, so the link never sizes it. V4.1 under `--place a` and \
              `bp` (`generate_ds41`, `bloomery-serve-ds41`): the load's host set also holds \
              the churn pool, each layer's stage card experts past the first P, refused by \
              name when the plan's host headroom cannot take it, and prints `residency host`; \
              every pass prints `residency pass`; only an explicit call resets it (an arm's \
              clear, the server's `POST /residency/reset`), with a `residency reset` record. \
              Unset follows the placement (`bloomery_levers::residency_unset`): the serving \
              default `mid-p0-s1` under `--place a` and `bp`, checked against the plan there \
              (`bloomery_levers::residency_at_plan`) — no seed expert pinned, so the churn pool \
              holds every stage card expert, P x layers x expert bytes of host RAM more than a \
              word that pins P (Qwen3.8 at a 22 % card share, 63 pinned a layer: +8.3 GB); the \
              pin is the fallback under host-RAM pressure: the first P whose churn pool fits \
              where the plan's host headroom or `MemAvailable` does not take the pool at 0, \
              `off` where the card slots leave no room for the word or no churn pool the card \
              side leaves fits — and `off` where the machine does not run — \
              `--place gate`, beside `BLOOMERY_CHECK_FINITE=1`, beside \
              `BLOOMERY_ROUTE_TRACE` (a fixed placement's routing) and beside \
              `BLOOMERY_PREFILL=steps` (each prompt id would end a pass the rule counts); both \
              binaries print the word and why as a `residency lever` record before the plan \
              record, the plan-side detail on the line beside it when the plan moved or \
              refused the word. \
              Set, `mid-…` is refused by name under `--place gate`, beside the finite probe, \
              the route trace and the step feed. Qwen3.8 (`generate_qwen3moe` on a qwen4exp \
              file, `--place a` or `gate`, plain or `BLOOMERY_DRAFT=mtp`): unset follows the \
              plan (`bloomery_levers::residency38_unset`, `residency38_at_plan`) — under \
              `--place a` `mid-p<P>-s1`, P 0 where the host room of the plan the load runs \
              (the plain or the MTP plan, at its `--ctx`) takes the churn pool and the first P \
              whose pool fits where it does not, derived at load so the default always fits the \
              plan; `off` under `--place gate`, beside \
              `BLOOMERY_ROUTE_TRACE`, with `--prefill step`, under `--dump-taps`, on a \
              qwen3moe or qwen35moe file, when the plan holds no card expert, when it leaves \
              no routed expert on the host (the pool would serve none) or its fewest \
              leave no room for one spare and one that moves, and when no churn pool the card \
              side leaves fits the plan's host headroom or what the host's \
              `MemAvailable` leaves past the plan's host need — never a refusal; the word and \
              why print as a `residency unset` record after the `plan` line (on a qwen3moe or \
              qwen35moe file, and under `--dump-taps`, before the load). \
              `mid-p0-s1` is the word a set lever names for plan (a) where its host takes the \
              pool; set, \
              `mid-…` runs the same machine over the card's routed stacks, the load's host set \
              holds the churn pool, refused by name as above, and prints `residency host`, and \
              each arm prints its `residency pass` records after its lines; an arm's clear \
              resets it, with a `residency reset` record; a set word and \
              `why=set` print as a `residency lever` record before the load. A set \
              `mid-p<P>-s<S>-unhosted` (Qwen3.8 only, refused by name on every other family) asks \
              the load to keep P with the churn pool unhosted: the pool is not read in or \
              locked, its victims are served from the file on the token path, and `residency \
              host` prints `hosted=unhosted` — measured −37 % decode on the maintainers' box at \
              plan a, so it is asked for, never a default. There `mid-…` is refused by \
              name beside `BLOOMERY_ROUTE_TRACE` (a fixed placement's routing), with a step-fed \
              prompt (`--prefill step`, at the prompt), on a qwen3moe or qwen35moe file and under \
              `--dump-taps`. `bloomery-serve-qwen38` takes it as `generate_qwen3moe` does, \
              from the same rule and records (its prompt path is never a step feed, and it takes \
              no route trace): unset `mid-p<P>-s1` under `--place a` from the plan it loads, \
              `off` under `--place gate`; running `mid-…` each call prints its `residency pass` \
              records; a request's reset keeps the residency, and only `POST /residency/reset` \
              moves it back to the seed, with a `residency reset` record. `generate_glm5next`: \
              unset is `off`; set, the word runs as given (on plan (a) `mid-p0-s1`, the GLM seat's \
              unset word, and `mid-p33-s1`, P half the fewest card experts a layer, both fit), \
              beside `BLOOMERY_DRAFT=mtp` over the NextN load, refused by name beside `--pair`, \
              `BLOOMERY_ROUTE_TRACE` and `--prefill steps`. The GLM seat of `bloomery-serve` takes \
              it as `generate_glm5next` does, plain or drafted, refused by name beside `--prefill \
              steps`. Unset (`bloomery_levers::glm_unset`, `glm_residency_at_plan`), a \
              `residency unset` record after the `plan` line names the word and why: \
              `mid-p0-s1` under `--place a` or `bp`; `off` under `--place gate`, with stores too short \
              for one window, beside `--prefill steps`, when the plan holds no card expert or its \
              fewest leave no room, and when the churn pool does not fit the plan's host \
              headroom less the NextN layer's host experts or what `MemAvailable` leaves — never \
              a refusal. Running `mid-…` each call \
              prints its `residency pass` records, a request's reset keeps the residency, and \
              `POST /residency/reset` moves it back to the seed with a `residency reset` record. \
              Every other binary refuses it set, by name.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: QWEN38_EXPERTS,
        class: Class::A,
        kind: Kind::Words(&["host", "card"]),
        default: Unset::Means("`card` on a qwen4exp file; nothing on another family's"),
        doc: "Qwen3.8 (`generate_qwen3moe`): `card` plans each layer's id prefix on the card \
              as its budget holds \
              (`place::Experts::Card`), run by the step's, the verify's and the pass's card \
              leg and the ubatch walk's card route; `host` plans every routed expert on the \
              host tier, the same-binary arm. Unset is `card` on a qwen4exp file; set to \
              `card`, a qwen3moe or qwen35moe file refuses it by name. The `plan` line prints \
              `experts=`.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: HOSTSTREAM,
        class: Class::A,
        kind: Kind::OnOff,
        default: Unset::Means(
            "V4.1: on under a residency (`BLOOMERY_RESIDENCY` not `off`), off without one",
        ),
        doc: "V4.1 prompt calls under adaptive residency (`BLOOMERY_RESIDENCY=mid-…`): `on` \
              streams each group's hottest host experts into the residency's churn pool at \
              every layer (`host::swap` call mode: a batch's counts pick them, the coldest pool \
              residents go to the host, and the call's picks stay for the decode after it), \
              at or past the prompt length the body names (`body::prefill`'s `STREAM_MIN_P`); \
              the prompt's bits are then the band's, not the decode steps'. `off` keeps a \
              prompt call bit for bit the decode steps' state, the same-binary arm. Unset follows \
              the residency: on under one, off without one. `on` is refused by name at the load \
              beside `BLOOMERY_RESIDENCY=off`. Every pick prints `call \
              stream`, every call `call stream end`. A Qwen3.8 load refuses it set by name and \
              reads `BLOOMERY_XSTREAM`.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: XSTREAM,
        class: Class::A,
        kind: Kind::Words(&["off", "admit", "split"]),
        default: Unset::Means(
            "Qwen3.8 (`generate_qwen3moe`, `bloomery-serve-qwen38`), and the Muse Spark and \
             MiniMax-M3.1 seats (`bloomery-serve`, MiniMax-M3.1's gates): `split` under `--place a` \
             with a residency (`admit` when the card has no room for the ring or a Qwen3.8 card \
             route reads NVFP4 planes), `admit` under `--place bp` with one, `off` everywhere else",
        ),
        doc: "Qwen3.8, Muse Spark and MiniMax-M3.1 prompt calls fed by ubatches or passes under adaptive residency \
              (`BLOOMERY_RESIDENCY=mid-…`): what each card layer's walk moves before its card \
              route. `admit` moves the residency's pool toward the ubatch's hottest host experts \
              (`host::swap` call mode, a count of at least the walk's floor), and the call's picks \
              stay for the decode after it. `split` admits, then streams the host experts the \
              rule (`runtime::xsplit`) sends to the card through a ring of card slots for the \
              ubatch alone (`host::xstream`): an expert streams when its columns cost the host \
              union more than its copy and the card's columns, at the measured rate of each \
              source (host memory, or the drive under the NVMe tier), kept under the device/host \
              workload key in the cost store (`BLOOMERY_COSTS`) so later loads of the key price alike. \
              The load line prints both the current probe and the priced rate. The rule cuts \
              at the ring's half and where the layer's copies would pass the \
              union it keeps; a ubatch narrower than the rule's least width streams nothing, so \
              decode and short prompts run as under `admit`. The host union leaves the streamed \
              experts out. The prompt's bits are then the band's, not the decode steps'. `off` \
              keeps a prompt call bit for bit the decode steps' state, the same-binary arm. \
              Unset is `split` under `--place a` with a residency (`admit` when the card has no \
              room for the ring or a Qwen3.8 card route reads NVFP4 planes, which its ring pass \
              does not, the load line saying so), `admit` under `--place bp` (the ring serves no \
              expert tier), `off` everywhere else; `admit` and `split` are refused by name beside \
              `BLOOMERY_RESIDENCY=off` and on a qwen3moe or qwen35moe file, `split` on a load \
              with an expert tier and on a Qwen3.8 load whose card route reads NVFP4 planes. The load line prints `xstream=`, \
              every streamed layer an `xstream` record, every call `xstream end`. A MiniMax-M3.1 \
              load (its gates' seat, `Body::set_xstream`) takes it through the common prompt walk: \
              `split` streams the host experts the rule sends to the card into the ring's half and \
              runs their slots there beside the card's stacks (NVFP4 planes staged from the \
              checkpoint folder's parts).",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: SPLIT_WALK,
        class: Class::A,
        kind: Kind::OnOff,
        default: Unset::Means(
            "Qwen3.8: the width rule's walk on a load whose routed experts sit on the cards and \
             in host memory alone, today's walk on a load with an expert tier or an NVMe-paged \
             plan",
        ),
        doc: "Qwen3.8 prompt calls fed by ubatches: `on` walks a ubatch the width rule \
              (`runtime::splitwalk::pays`) finds wide enough as two units of half its rows — \
              each unit's next front enqueued ahead of the other's card route, so the card runs \
              one unit's front while the host serves the other's union, the layer's fronts in \
              unit order, one residency pick and one expert stream a layer for both units. \
              Unstreamed, a token's bits do not move; streamed, the call's pick and stream are \
              priced from the first unit's routing. `off` walks every ubatch as one unit, the \
              same-binary arm. Unset is `on` on a load whose routed experts sit on the cards \
              and in host memory alone and `off` on a load with an expert tier (`--place bp`) or \
              an NVMe-paged plan, whose costs the rule does not price; `on` there is refused by \
              name at the load.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: QWEN3_KV,
        class: Class::A,
        kind: Kind::Words(&["f16", "q8_0"]),
        default: Unset::Means("`f16`, but `q8_0` on the Qwen3.8 seat's checkpoint folder"),
        doc: "Qwen3, Qwen3.6 and Qwen3.8 (`generate_qwen3moe`, `bloomery-serve --model qwen3`, \
              `bloomery-serve --model qwen38`): the format the load holds its K/V cache planes \
              in — `q8_0` the two-plane layout (17/16 B a value against f16's 2, both planes quantized together by the one \
              choice). On Qwen3 and Qwen3.6, a cache half the bytes wide lets the default \
              `--ctx` ladder take a higher rung on the same card. Qwen3.8's index keys stay f16, \
              so its attention store does not shrink by the same factor. Read once at load: the \
              append quantizes each row and the flash reads it through the format's \
              algebra, so a q8_0 run's bits are its own, never an f16 run's; every plan the \
              ladder probes counts the format's bytes. The seat's \
              `--cache-type-k` takes the same two words (llama-server's spelling) and \
              wins over the lever; `--cache-type-v` does not exist — both planes quantize \
              together. Unset on the Qwen3.8 seat, a checkpoint folder (FP4-Local) runs \
              `q8_0` and a GGUF file `f16`; its MTP draft's store takes the same format. The \
              qwen3 seat's `load` line prints `cache=`, the Qwen3.8 seat's `setup` line.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: LANE_PREFETCH,
        class: Class::A,
        kind: Kind::OnOff,
        default: Unset::Is("off"),
        doc: "Host union, a prompt batch's row-lane passes (Q4_K, Q5_K and Q6_K stacks, \
              calls wider than 8 columns): `on` has each 8-row group's pack prefetch the \
              next group of its participant's range toward L2, a slice a super-block \
              (`qdot::pack_lanes`' `next`); `off` packs without it, the same-binary arm. \
              Both write the same bits. `generate_glm5next` acts on it and prints the value \
              the union holds as its `load` record's `lane_prefetch`; every other binary \
              runs the default and refuses the name set.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: HOST_LANES,
        class: Class::A,
        kind: Kind::Words(&["flat", "ccd"]),
        default: Unset::Is("ccd"),
        doc: "Host tier, the decode step's one-column leg and the verify leg's step union: \
              `ccd` lays each pass's units over the pool's CCDs (each CCD's participants \
              take that CCD's share of every matrix, and take from the other CCDs' lanes \
              only once their own are done) where the pool has a CCD map, which takes the \
              calling thread pinned (`BLOOMERY_PIN_MAIN`, default on) and every worker \
              pinned — an unpinned run has none and runs flat whatever this says. `flat` \
              cuts every matrix's units end to end into one lane a participant, the \
              same-binary arm. Both write the same bits. Acted on by the binaries that pin \
              their main thread: `generate_ds41`, `bloomery-chat`, and the ds41, qwen38, \
              glm and mimo2 seats of `bloomery-serve` (`bloomery-serve-ds41`, \
              `bloomery-serve-qwen38`, `bloomery-serve-mimo2`); every other binary that parses its levers runs \
              the default and refuses the name set, and `bloomery-decode` and \
              `bench_v41_host`, which do not read it, run the default.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: GEN_SLOTS,
        class: Class::A,
        kind: Kind::Count {
            min: 1,
            max: GEN_SLOTS_MAX,
            trim: false,
        },
        default: Unset::Is("1"),
        doc: "`generate_qwen3moe` on a whole-card qwen3moe or qwen35moe file or a qwen4exp file, and \
              `generate_ds41` (V4.1): N streams decoded in one pass (`GpuModel::step_slots`). \
              Each arm's ids are N windows of equal length, window j prefilled into slot j; then \
              `-n` − 1 rounds of one pass of a row a slot, each slot fed its own argmax. Under \
              `--time` a round prints `time pass <r> ms= positions=N kind=slots` (the depth \
              runners' aggregate row: Σ positions / Σ ms). 1 is the one-sequence run, the \
              same-binary arm. N past the body's pass rows (`SlotRows::MAX_ROWS`: 8 for qwen3moe, \
              qwen35moe and qwen4exp, 2 for V4.1) and an id count N does not divide are refused by name. \
              `generate_qwen3moe` prints every slot's ids as `step` lines ending in `slot=<j>`, \
              plans a qwen4exp load for N resident sequences, and refuses N ≥ 2 on a placed \
              qwen3moe or qwen35moe load, a qwen4exp run that drafts \
              (`BLOOMERY_DRAFT=off` runs it plain), and beside `--prompt`, `--seed-depth`, \
              `--last-step`, `--logits`, `--dump-taps`, and on a qwen4exp file \
              `BLOOMERY_STEP_STATS` and `BLOOMERY_ROUTE_TRACE`. `generate_ds41` loads a plan of N \
              resident sequences, runs each slot's window as the plain run's prompt call, prints \
              each slot's records in slot order, and refuses N ≥ 2 beside `BLOOMERY_DRAFT`, \
              `BLOOMERY_CHECK_FINITE`, `BLOOMERY_STEP_STATS` and `--logits`.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: FORCE_IDS,
        class: Class::D,
        kind: Kind::FileAt,
        default: Unset::Means("each row's argmax, the greedy ids"),
        doc: "`generate_qwen3moe` on a qwen3moe, qwen35moe or qwen4exp file, greedy, one stream: \
              `<path>:<offset>` takes every generated id from a file of ids, one a line, from the \
              line `offset` on (the file's first `offset` lines are skipped; a corpus runner's \
              `continue` is its prompt's length), whatever the row's argmax. Generated token 0 is \
              the file's first id and every id after it the next, on the plain path and under \
              `BLOOMERY_DRAFT=mtp`, where a proposal is kept exactly while it matches the file, \
              so a drafted run and a plain run decode one id stream and differ only in the \
              passes they run (a verify's batched rows round differently from a one-row step, so \
              their greedy ids part otherwise, and a paged plan's cost of a token depends on its \
              routing). The file holds at least `-n` ids from `offset`, and with a draft the \
              window's rows less one more, or the run is refused by name before the load. Refused \
              by name beside `BLOOMERY_GEN_SLOTS` past 1 (the slots pass takes each slot's \
              argmax) and `--dump-taps`; this CLI has no sampler to refuse. The ids are not the \
              model's: an accuracy or text reading of a forced run is void.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: SCREEN,
        class: Class::C,
        kind: Kind::OnOff,
        default: Unset::Is("on"),
        doc: "The start screen of `bloomery-serve` (`serve::screen`): `on` draws it on stderr \
              once the server listens, a view for a person over the record lines the start \
              printed (where the model went, its context and slots, the address); `off` \
              leaves the record lines alone. It is drawn only when stderr is a terminal, so \
              piped stderr never shows it, whatever this says. `NO_COLOR` (set and not empty) \
              or `TERM=dumb` draws it without colour; `COLORTERM=truecolor` (or `24bit`) draws \
              its colours in 24 bits, any other terminal in the 256-colour cube; `TERM=dumb` \
              or a locale whose first set variable of `LC_ALL`, `LC_CTYPE` and `LANG` names \
              no UTF-8 draws it in plain ASCII. `on` also lets `d` open the live view \
              (`serve::live`) while the server serves, on the alternate screen, `q` or `Esc` \
              closing it; it takes keys only when stdin and stderr are terminals, the process \
              is the terminal's foreground group and `TERM` is not `dumb`, and only then does \
              a screen name the key. Every seat of `bloomery-serve` acts on it, and so do the \
              per-model binaries that are its seats and `bloomery-serve-metal`, which draws \
              no start screen and names the key on a line of its own.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: SCHEMAJUMP,
        class: Class::A,
        kind: Kind::OnOff,
        default: Unset::Is("on"),
        doc: "Structured output (`response_format` `json_schema`, a forced `tool_choice`) on a \
              server: `on` feeds each run of ids the program fixes (forced text, the rest of a \
              chosen option) as one prompt call, the jump arm, and reads the next decision from \
              the row of the step that closes the run; `off` is the step arm, one `next` an id. \
              Both arms pick the largest allowed logit at every decision and feed every id they \
              take, so they differ in the engine calls and, by the engine's batch rounding, \
              possibly in the bits of a decision's row. A request that asks for probabilities \
              takes the step arm whatever this says. A request's `timings.constrained` counts \
              the calls of the arm it took. A server takes it as \
              `serve::SlotConfig::schema_jump`.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
    },
    LeverSpec {
        name: PREPARE_Q8,
        class: Class::M,
        kind: Kind::Flag,
        default: Unset::Is("0"),
        doc: "`prepare`: `1` writes each BF16 or F32 matrix its checkpoint map reads as Q8_0 (the \
              attention and linear-attention projections, the shared expert, the PLE key and \
              value, the hyper-connections' down and up, the head and the token embedding) as \
              ggml's reference Q8_0; `0` keeps each single BF16 matrix as stored and refuses \
              F32 matrices read as Q8_0. BF16 and F32 expert stacks are refused, and the \
              quantized tensors (NVFP4, MXFP8, FP8) keep their stored blocks, under both \
              values. The rewrite is lossy: on \
              `nvidia/Qwen3.8-Flash-Next-NVFP4` at fc694b54, the checkpoint's own f32 forward \
              with its BF16 projections rounded to Q8_0 is at a KL divergence of 4.288e-02 from \
              the forward with them as stored, at the last of the five positions of the \
              batch set, and 4.181e-02 at position 3,000 of the D3K set (refset \
              `hf-qwen4exp-nvfp4`, the `_bf16` sets against the others), so it is asked for, \
              never a default. `--keep-bf16`, which asks for the stored form, is refused \
              beside it.",
        site: Site::Parsed {
            left: &[],
            phase: Phase::Load,
        },
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
            phase: Phase::Load,
            by: Readers::Every,
        },
    },
    LeverSpec {
        name: "BLOOMERY_DEFER_QUANT",
        class: Class::T,
        kind: Kind::Flag,
        default: Unset::Means("the only path"),
        doc: "Was the host tier's caller-side quantization pre-pass, the twin the union \
              tests compare against.",
        site: Site::Retired {
            why: "the row dispatch quantizes its activations itself; the caller-side pre-pass \
                  is a test's oracle, picked by `ops::set_defer_quant`",
            left: &[],
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
            phase: Phase::Load,
            by: Readers::Every,
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
        default: Unset::Means("the only path"),
        doc: "Was the blocks a host-tier lane is cut into for stealing.",
        site: Site::Retired {
            why: "a lane is cut into four blocks, the const `STEAL_BLOCKS` of \
                  `crates/model/src/ops.rs`",
            left: &[],
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
            phase: Phase::Load,
            by: Readers::Named,
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
            phase: Phase::Load,
            by: Readers::Every,
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
            phase: Phase::Load,
            by: Readers::Named,
        },
    },
    LeverSpec {
        name: "BLOOMERY_QDOT_AVX512",
        class: Class::A,
        kind: Kind::Flag,
        default: Unset::Is("1"),
        doc: "Host tier, Q4_K weights: `1` runs the one-column kernel, the tile and the \
              row lanes through their AVX-512 VNNI bodies on a CPU with AVX-512 F, BW and \
              VNNI; `0` keeps the AVX2 bodies, the same-binary arm. Both write the same bits. \
              Read once a process (`qdot::q4k_avx512`).",
        site: Site::Direct {
            at: &[InPlace {
                file: QDOT,
                round: ISA_OWNER,
            }],
            phase: Phase::Load,
            by: Readers::Every,
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
            phase: Phase::Load,
            by: Readers::Named,
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
            phase: Phase::Load,
            by: Readers::Named,
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
            phase: Phase::Load,
            by: Readers::Named,
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
            phase: Phase::Load,
            by: Readers::Named,
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
            phase: Phase::Load,
            by: Readers::Named,
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
            phase: Phase::Load,
            by: Readers::Named,
        },
    },
    LeverSpec {
        name: "BLOOMERY_GQA_MMA",
        class: Class::T,
        kind: Kind::Flag,
        default: Unset::Means("the only path"),
        doc: "Was the Qwen3 GQA flash's scalar segment pass.",
        site: Site::Retired {
            why: "the tensor-core segment pass is the only Qwen3 decode flash pass; the scalar \
                  pass is a gate's ruler on an f16 cache, picked by `set_flash_mma`",
            left: &[],
        },
    },
    LeverSpec {
        name: "BLOOMERY_Q3K_SPLIT",
        class: Class::A,
        kind: Kind::Words(&["1", "2", "4", "8"]),
        default: Unset::Means("the only path"),
        doc: "Was the GPU Q3_K gemv's split-K width.",
        site: Site::Retired {
            why: "a Q3_K row runs the one-warp gemv only",
            left: &[],
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
        default: Unset::Means("the only path"),
        doc: "Was the walk length from which the GPU Q3_K gemv split a row.",
        site: Site::Retired {
            why: "a Q3_K row runs the one-warp gemv only",
            left: &[],
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
                round: V2FENCE,
            }],
            phase: Phase::Load,
            by: Readers::Named,
        },
    },
    LeverSpec {
        name: "BLOOMERY_HYBRID_OVERLAP",
        class: Class::T,
        kind: Kind::Flag,
        default: Unset::Means("the only path"),
        doc: "Was the hybrid MoE's wait right after each layer's go instead of after the \
              card's experts and the shared expert.",
        site: Site::Retired {
            why: "each hybrid layer's wait sits after the card's experts and the shared expert",
            left: &[],
        },
    },
    LeverSpec {
        name: "BLOOMERY_LAUNCH_THREAD",
        class: Class::A,
        kind: Kind::Flag,
        default: Unset::Means("the only path"),
        doc: "Was `GpuModel`'s launch thread for each replay's `cuGraphLaunch`.",
        site: Site::Retired {
            why: "the decode thread issues every replay's launch",
            left: &[],
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
        doc: "Qwen3 and Qwen3.8 GEMM prefill: tokens per ubatch, read at load (Qwen3.8 sizes \
              its ubatch arena by it); a token's bits do not depend on it. A server's \
              `-ub`/`--ubatch-size`, where its seat takes the flag, replaces it. The `load` \
              line prints `ubatch=`.",
        site: Site::Direct {
            at: &[InPlace {
                file: "crates/gpu/src/arch/qwen3moe/ubatch.rs",
                round: R03,
            }],
            phase: Phase::Load,
            by: Readers::Named,
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
            phase: Phase::Load,
            by: Readers::Named,
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
            phase: Phase::Load,
            by: Readers::Named,
        },
    },
    LeverSpec {
        name: "BLOOMERY_DSPARK_CARD",
        class: Class::C,
        kind: Kind::Text,
        default: Unset::Means("the tier card under `bp`, else the smallest visible card"),
        doc: "`generate_ds41` and `bloomery-serve-ds41` under `BLOOMERY_DRAFT=dspark`: the \
              card the DSpark draft loads on, one card as a `--place` list word names it (a \
              CUDA ordinal, `1` or `cuda1`, or a card name exactly one visible device \
              carries); it may be the target's own card. Unset, the placement's first expert \
              tier card where it has one (`--place bp`), else the visible device of the \
              fewest usable bytes. The plan sets the draft's bytes aside on that card before \
              its experts: on a card of the placement at its place in the plan, on any other \
              visible device as a card that joins the plan to hold the draft and no expert. \
              A word no visible device answers, a name two carry, and a list are refused by \
              name.",
        site: Site::Direct {
            at: &[InPlace {
                file: "crates/gpu-gates/src/bin/shared/ds41_dspark.rs",
                round: SESSION,
            }],
            phase: Phase::Load,
            by: Readers::Named,
        },
    },
    LeverSpec {
        name: "BLOOMERY_CARD_EXPERTS",
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
        name: "BLOOMERY_STEP_PAIR",
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
        name: "BLOOMERY_FLASH_MMA",
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
        name: "BLOOMERY_AB_SET",
        class: Class::A,
        kind: Kind::Text,
        default: Unset::Means("the only path"),
        doc: "Was a lever set for `tools/ref/depth-gpu.sh`'s `ab:` arms, which refuses it \
              too.",
        site: Site::Retired {
            why: "`generate --ab` runs the shipped path alone",
            left: &[],
        },
    },
    // ------------------------------------------------------------ the paths
    path(
        "BLOOMERY_ARGMAX_OUT",
        Some("ref/argmax.sh"),
        "The token reference file `tools/ref/argmax.sh` writes, a file under the data \
         directory named by the profile when unset.",
    ),
    path(
        "BLOOMERY_CACHE",
        None,
        "The directory a server's `--hf <repo>[@<rev>][:<quant>]` fetches into, \
         `<dir>/<owner>/<name>[@<rev>]/<file>`; `~/.cache/bloomery/hf` when unset.",
    ),
    path(
        "BLOOMERY_COSTS",
        None,
        "The directory of the cost store, where a load keeps what it timed for the expert \
         stream and the split walk (the lane's rates, the host union's costs, the host's \
         enqueue cost) under its device/host and workload key (effective host kernels, \
         source layout and probe geometry), and where every later load of the \
         key reads them (`crates/runtime/src/costs.rs`); `~/.cache/bloomery/costs` when unset. \
         A key's first timing stands: delete its file (or the directory) and the next load \
         times again, e.g. after a load timed under `taskset` or a busy host. A load whose \
         stream or split rule needs costs the store cannot keep is refused by name. \
         The split walk requests them at its first rule-driven multi-row ubatch; loads, \
         decode, passes and split-walk `off` request no split-walk costs. \
         The expert stream requests its costs when it starts.",
    ),
    path(
        "BLOOMERY_DATA",
        Some("ref/ref-paths.sh"),
        "The data directory: reference sets, dumps, corpora and the reference binaries. \
         `tools/box.sh` exports `tools/ref/ref-paths.sh`'s into every box command; a \
         caller's own wins.",
    ),
    path(
        "BLOOMERY_DECODE_BIN",
        Some("ref/decode-measure.sh"),
        "The `bloomery-decode` binary the CPU runners (`decode-measure.sh`, \
         `perf-decode.sh`, `profile-measure.sh`) time.",
    ),
    path(
        "BLOOMERY_DEPINFO_REMOTE",
        Some("recipes.py"),
        "Mac side: the box directory whose dep-info `tools/recipes.py affected` reads.",
    ),
    path(
        "BLOOMERY_DSPARK_MODEL",
        Some("ref/timing-card.sh"),
        "The DSpark draft file: the V4.1 profile's `DSPARK_MODEL` \
         (`tools/ref/models/deepseek41.sh`), which the DSpark recipes and \
         `tools/ref/timing-card.sh` export; `generate_ds41` and `bloomery-serve-ds41` \
         under `BLOOMERY_DRAFT=dspark` read it, and a binary that needs it and finds it \
         unset says so. `bloomery-serve`'s raon1 seat reads it as its DFlash2 draft's file \
         before the one beside the target, and refuses a file that is not a DFlash2 draft \
         by name.",
    ),
    path(
        "BLOOMERY_ENGRAM_BIN",
        Some("ref/engram-rate.sh"),
        "The `engram-rate` binary `tools/ref/engram-rate.sh` runs.",
    ),
    path(
        "BLOOMERY_ENV_FILE",
        Some("box.sh"),
        "Local mode (`BLOOMERY_BOX_LOCAL=1`): the environment file `tools/box.sh` sources \
         before the command, `~/bloomery-env.sh` when unset.",
    ),
    path(
        "BLOOMERY_FIXTURE_MODEL",
        Some("box.sh"),
        "The fixture file `tools/box.sh` resolved the fixture tier's model to (the first shard), set only \
         when the tier resolved one. `tools/gpu-gate.sh` takes no V4.1 load lock for a run whose \
         `BLOOMERY_REF_MODEL` is this file and asks for none otherwise.",
    ),
    path(
        "BLOOMERY_FIXTURE_ROOT",
        Some("ref/ref-paths.sh"),
        "Box side: the directory the fixture tier's files live in, one directory a family \
         (`/models/fixtures` when unset); `tools/ref/ref-paths.sh` reads it, and `tools/box.sh` does \
         not carry it from the Mac.",
    ),
    path(
        "BLOOMERY_FIXTURE_VARIANT",
        Some("ref/ref-paths.sh"),
        "The fixture variant, a tag of `tools/fixture-variants.tsv`: the family's fixture requantized to \
         the tag's type map (`just fixture-requant`), whose files live in `fixture-variants/<tag>` beside \
         the fixture root and whose sets carry the tag in their names. `tools/box.sh` reads it from the \
         environment or `BLOOMERY_BOX_ENV` before the profile, as it does `BLOOMERY_TIER`; only the \
         fixture tier has variants (the real tier refuses one, 64), and `tools/ref/ref-paths.sh` refuses \
         a tag the table does not name. refset reads it in place (`fixture::root`).",
    ),
    path(
        "BLOOMERY_GATE_LEDGER",
        Some("gate-batch.sh"),
        "Mac side: the green ledger `tools/gate-batch.sh --ledger` reads and writes.",
    ),
    path(
        "BLOOMERY_GATE_ROUND_LEDGER",
        Some("gate-batch.sh"),
        "Mac side: the rounds' green ledger `tools/gate-batch.sh --round-ledger` writes; every \
         batch reads it, the lead's only under `--trust-rounds`.",
    ),
    path(
        "BLOOMERY_GATE_TIMES",
        Some("gate-batch.sh"),
        "Mac side: the gate wall-time file `tools/gate-batch.sh` balances its lanes by.",
    ),
    path(
        "BLOOMERY_GEN_BIN",
        Some("ref/depth-ds41.sh"),
        "The generator binary a GPU runner (the depth, nsys and ncu runners) times, each \
         runner's own when unset.",
    ),
    path(
        "BLOOMERY_IQ3XXS_MODEL",
        Some("ref/build-qdot-ref.sh"),
        "The file IQ3_XXS's qdot harness and gate read, the Qwen3.8 UD-Q3_K_XL second shard when unset.",
    ),
    path(
        "BLOOMERY_KLD_FILE",
        None,
        "A KLD base file the KLD gate judges instead of its set's \
         (`crates/gpu-gates/src/kld.rs`); the gate's FAIL-first points it at a copy.",
    ),
    path(
        "BLOOMERY_LEASE_CARD",
        Some("ref/lease.sh"),
        "The prediction card (`docs/cards/<slug>.card`, format in `tools/ref/card.py`) a \
         run that takes the timing lease carries; `lease_take` refuses a run without one.",
    ),
    path(
        "BLOOMERY_LEASE_HOLDS",
        Some("ref/lease-probe.sh"),
        "The hold files the box guard reads, a glob; the tools' stub tests point it at \
         their own.",
    ),
    path(
        "BLOOMERY_LEASE_LOCK",
        Some("ref/lease-probe.sh"),
        "The timing lease's lock file; the tools' stub tests point it at their own, and \
         `lease_take` refuses another on the box.",
    ),
    path(
        "BLOOMERY_LEASE_PROC",
        Some("ref/lease-probe.sh"),
        "The process tree the lease's holders are read from, `/proc` when unset; the \
         tools' stub tests point it at their own.",
    ),
    path(
        "BLOOMERY_LEASE_TRACE",
        Some("ref/lease.sh"),
        "The lease's trace file, `bloomery-sittings.log` in the lock directory under the \
         machine lease when unset (`/root/bloomery-sittings.log` on the box); the tools' stub \
         tests point it at their own.",
    ),
    path(
        "BLOOMERY_LLAMA_BPE_GGUF",
        None,
        "The reference's vocabulary test file for that pre-tokenizer, its `.inp` and `.out` \
         beside it, whose texts the tokenizer crate's vector gate encodes.",
    ),
    path(
        "BLOOMERY_LOCK_DIR",
        Some("ref/lease-probe.sh"),
        "The directory of the timing lease, the holds and the gate locks, `/root` when \
         unset; `tools/box.sh`'s local mode sets it, `~/.cache/bloomery/locks` by default.",
    ),
    path(
        "BLOOMERY_MM31_FOLDER",
        Some("ref/mm31-dump.sh"),
        "The MiniMax-M3.1-Flash checkpoint folder or its fixture cut: `tools/ref/mm31-dump.sh` \
         dumps it (the tool profile's model when unset) and the folder reader's gate opens it by \
         its map.",
    ),
    path(
        "BLOOMERY_MM31_HF",
        None,
        "The MiniMax-M3.1-Flash folder whose tokenizer files, config.json and chat template the \
         vocabulary and encode gates of the hf crate read.",
    ),
    path(
        "BLOOMERY_MODEL",
        Some("box.sh"),
        "Mac side, the tool profile a box command runs under (`tools/ref/models/`), which \
         `tools/box.sh` reads and does not export; on the box, the model crate's test \
         harnesses read it as a model file's path.",
    ),
    path(
        "BLOOMERY_MTP_MODEL",
        None,
        "The Qwen3.6 file that carries the model's next-token layer (unsloth's \
         `UD-Q4_K_M`), which `gate_qwen35moe_mtp` opens and pins by its `general.name` and \
         size; the gate refuses to run without it.",
    ),
    path(
        "BLOOMERY_MXFP4_MODEL",
        Some("ref/build-qdot-ref.sh"),
        "The file MXFP4's qdot harness and gate read, the V4.1 DSpark draft's tl37 file when unset.",
    ),
    path(
        "BLOOMERY_NCU_BIN",
        Some("ref/ncu-gpu.sh"),
        "Another tree's `generate_qwen3moe` (an absolute path under its `target/`) the ncu \
         runner's q3pp form profiles.",
    ),
    path(
        "BLOOMERY_NCU_GEMM_BIN",
        Some("ref/ncu-gpu.sh"),
        "The `gate_p8` binary the ncu runner's gemm form profiles.",
    ),
    path(
        "BLOOMERY_NCU_OUT",
        Some("ref/ncu-gpu.sh"),
        "The directory the ncu runner writes its reports into.",
    ),
    path(
        "BLOOMERY_NCU_TRACE",
        Some("ref/ncu-gpu.sh"),
        "The nsys prefill trace the ncu runner's ds41pp form reads its launch skip from, \
         the newest of its prompt length when unset.",
    ),
    path(
        "BLOOMERY_NSYS_OUT",
        Some("ref/nsys-gpu.sh"),
        "The directory the nsys runners write their traces into.",
    ),
    path(
        "BLOOMERY_PROMPTS",
        Some("ref/argmax.sh"),
        "The prompt file `tools/ref/argmax.sh` takes its prompts from, the profile's when \
         unset.",
    ),
    path(
        "BLOOMERY_Q5K_MODEL",
        Some("ref/build-qdot-ref.sh"),
        "The file q5_K's qdot harness and gate read, the V4.1 first shard when unset.",
    ),
    path(
        "BLOOMERY_QWEN38_FOLDER",
        None,
        "The Qwen3.8-Flash-Next checkpoint folder with its MTP layer that the MTP input gate loads.",
    ),
    path(
        "BLOOMERY_QWEN38_HF",
        None,
        "The checkpoint folder whose tokenizer files the prepare corpus test reads.",
    ),
    path(
        "BLOOMERY_QWEN_Q4_MODEL",
        Some("ref/build-qdot-ref.sh"),
        "The Qwen3.8 UD-Q4_K_XL shard IQ4_NL's qdot harness and gate read, the second shard when unset.",
    ),
    path(
        "BLOOMERY_QWEN_Q3_MODEL",
        Some("ref/build-qdot-ref.sh"),
        "The Qwen3.8 UD-Q3_K_XL shard IQ4_XS's qdot harness and gate read, the second shard when unset; its `.done` \
         sentinel gates the real rows.",
    ),
    path(
        "BLOOMERY_QWEN3MOE_VOCAB",
        None,
        "The Qwen3-MoE file the tokenizer gate reads its vocabulary from, the box's when \
         unset.",
    ),
    path(
        "BLOOMERY_REF_CPU_SET",
        None,
        "The CPU reference set `gate_block` compares against instead of its oracle \
         table's.",
    ),
    path(
        "BLOOMERY_REF_CUDA",
        None,
        "An absolute path to the CUDA reference set the gates read instead of the one \
         `BLOOMERY_REF_SET` or the oracle table names.",
    ),
    path(
        "BLOOMERY_REF_DIR",
        Some("ref/dump.sh"),
        "The directory the C++ dumpers write a reference set into: the staging directory \
         `tools/ref/dump.sh` and `dump-draft.sh` hand them.",
    ),
    path(
        "BLOOMERY_REF_MODEL",
        Some("box.sh"),
        "The model file the gates and the GPU binaries open: `tools/box.sh` exports the \
         tool profile's `MODEL` (`tools/ref/ref-paths.sh`) into every box command; a \
         caller's own wins.",
    ),
    path(
        "BLOOMERY_REF_SET",
        Some("ref/dump.sh"),
        "The reference set a gate reads instead of its oracle table's, and the one \
         `tools/ref/dump.sh` writes.",
    ),
    path(
        "BLOOMERY_REMOTE",
        Some("box.sh"),
        "Mac side: the box directory `tools/box.sh` syncs a tree into and runs in, a \
         track's own in a parallel round.",
    ),
    path(
        "BLOOMERY_SCALES",
        None,
        "The directory of the scale store, where a load of a checkpoint folder whose NVFP4 \
         experts store their block scales compressed (`csf`, `csf_32x2_v1`, as FP4-Local's do) \
         keeps those scales read back, one file per folder generation, so the host reads its \
         experts in place (`crates/model-hf/src/folder/scales.rs`); `~/.cache/bloomery/scales` \
         when unset. The \
         open sizes the file and writes no scale; the first load whose host reads an expert \
         decodes them, and later loads of the unchanged folder read them as written. A changed \
         folder opens another path; older files stay available to loads holding their paths.",
    ),
    path(
        "BLOOMERY_TOKENIZE_BIN",
        Some("ref/ik-draft.sh"),
        "The `bloomery-tokenize` binary `tools/ref/ik-draft.sh` runs.",
    ),
    path(
        "BLOOMERY_V41_DIR",
        Some("box.sh"),
        "The V4.1 file's directory, exported beside `BLOOMERY_V41_MODEL`.",
    ),
    path(
        "BLOOMERY_V41_ENGRAM",
        None,
        "The V4.1 Engram companion file: a set that stores its Engram tables in a GGUF of their own \
         (`deepseek41.engram.file` names it) opens that file as a further shard of the model. Unset, \
         the open looks for the named file beside the first shard, then in the sibling directory of \
         the file's stem; set, this path is taken instead, and a model that needs no companion refuses it.",
    ),
    path(
        "BLOOMERY_V41_MODEL",
        Some("box.sh"),
        "The V4.1 first shard: `tools/box.sh` exports the deepseek41 profile's \
         `V41_MODEL` into every box command whatever profile it picked; a caller's own \
         wins.",
    ),
    path(
        "BLOOMERY_V4_MODEL",
        Some("ref/build-qdot-ref.sh"),
        "Was the one file IQ3_XXS's and MXFP4's qdot harnesses and gates read; the harness \
         script and the gate refuse it when set, and `BLOOMERY_IQ3XXS_MODEL` and \
         `BLOOMERY_MXFP4_MODEL` name a file each.",
    ),
    path(
        "BLOOMERY_VISION_SET",
        Some("ref/vision/dump-vision.sh"),
        "The vision reference set (a directory under `ref-vision`) the vision gates read, \
         and the one `tools/ref/vision/dump-vision.sh` writes.",
    ),
    // ---------------------------------------------- the runners' variables
    runner(
        "BLOOMERY_AB_AA",
        Some("ref/bin-label.sh"),
        "`tools/ref/depth-ds41.sh` and `depth-qwen3moe.sh` (through `tools/ref/bin-label.sh`): `1` \
         declares the run a deliberate A/A -- a bin: arm may then live in the ours tree or share \
         the ours build's sha; any other run is refused before the lease.",
    ),
    runner(
        "BLOOMERY_AB_DEPTH",
        Some("ref/ab-decode.sh"),
        "`tools/ref/ab-decode.sh`: the depth whose prompt is prefilled before each arm is \
         timed.",
    ),
    runner(
        "BLOOMERY_AB_ENVS",
        Some("ref/ab-decode.sh"),
        "The A/B runners: same-binary arms that differ by levers alone, `K=V;K=V`.",
    ),
    runner(
        "BLOOMERY_AB_IK",
        Some("ref/ab-decode.sh"),
        "`tools/ref/ab-decode.sh`: `1` adds the reference engine at its fastest flags as \
         an arm of every round.",
    ),
    runner(
        "BLOOMERY_AB_INNER",
        Some("ref/depth-gpu.sh"),
        "`tools/ref/depth-gpu.sh`: the inner rounds of an `ab:` arm.",
    ),
    runner(
        "BLOOMERY_AB_ORDER",
        Some("ref/cold-blocks.sh"),
        "`tools/ref/depth-ds41.sh` and `depth-qwen3moe.sh` (through `cold-blocks.sh`): `rotate` \
         (unset) runs every arm once a round in rotated order; `blocks` runs each engine's arms \
         together after one discarded process.",
    ),
    runner(
        "BLOOMERY_AB_LOAD",
        Some("ref/load-groups.sh"),
        "`tools/ref/depth-ds41.sh` and `depth-qwen3moe.sh`: `key` (unset) runs a round's arms of \
         one load key in one process, the engine cleared between them; `arm` runs each arm in a \
         process of its own. An arm's own `@BLOOMERY_AB_LOAD=arm` runs that arm alone.",
    ),
    runner(
        "BLOOMERY_AB_ROUNDS",
        Some("ref/ab-decode.sh"),
        "The A/B and depth runners: the rounds each arm runs.",
    ),
    runner(
        "BLOOMERY_AB_WARMUP",
        Some("ref/depth-ds41.sh"),
        "`tools/ref/depth-ds41.sh` and `depth-qwen3moe.sh`: `1` runs the first arm once, \
         discarded, before round 1 under `rotate`, and is the blocks' discards under `blocks`; \
         `0` skips them. Unset is `1` in `depth-ds41.sh`; in `depth-qwen3moe.sh` it is `0` under \
         `rotate` and `1` under `blocks`.",
    ),
    runner(
        "BLOOMERY_ARCH",
        Some("arch.sh"),
        "The CUDA arch the device builds target: `sm_` and the capability's digits, with an \
         optional `a` or `f` (`sm_86`, `sm_121`, `sm_121a`). Unset, `tools/arch.sh` picks it \
         from the visible cards: `sm_120a` when every one is compute capability 12.0, `sm_121a` \
         when every one is 12.1, else `sm_86`. The justfile's `arch` variable reads it for every \
         `--arch`, `tools/arch.sh` parses it for the tools that run on the box, and \
         `tools/box.sh` forwards it there.",
    ),
    runner(
        "BLOOMERY_ARM_BOUND",
        Some("ref/lease.sh"),
        "Seconds one arm of a timed runner may run under the lease before `timeout` ends \
         it.",
    ),
    runner(
        "BLOOMERY_BATCH_OWNER",
        Some("gpu-gate.sh"),
        "The owner of the GPU hold a landing batch put up (letters, digits, `_`): \
         `tools/gate-batch.sh --ledger` exports it into each item's box env, and \
         `tools/gpu-gate.sh` lets a run with this owner pass the hold while every other \
         run waits.",
    ),
    runner(
        "BLOOMERY_BATCH_START_HOOK",
        Some("gate-batch.sh"),
        "Mac side: a command line (split on blanks, no shell syntax) that `tools/gate-batch.sh` \
         starts once in the background at the start of a real run; its output goes to \
         `start-hook.log` and its exit code to `start-hook.rc` in the batch's out directory, and a \
         code other than 0 is one stderr line at the end of the run. It never delays the batch or \
         changes its code. Unset or empty: nothing runs.",
    ),
    runner(
        "BLOOMERY_BOX",
        Some("box.sh"),
        "Mac side: the box's ssh host.",
    ),
    runner(
        "BLOOMERY_BOX_ENV",
        Some("box.sh"),
        "Mac side: `NAME=value` pairs `tools/box.sh` exports into the box command, so a \
         lever reaches a binary through an unchanged recipe.",
    ),
    runner(
        "BLOOMERY_BOX_LOCAL",
        Some("box.sh"),
        "`1`: `tools/box.sh` runs the command in this tree on this machine, with no ssh and \
         no rsync, for a contributor's Linux machine with a card of its own; its locks live \
         under `BLOOMERY_LOCK_DIR`.",
    ),
    runner(
        "BLOOMERY_BOX_PIDDIR",
        Some("box.sh"),
        "Mac side: a directory `tools/box.sh` files its own pid in (an empty file named for the pid) \
         while it runs, for the caller to end it by: `tools/gate-batch.sh` sets it per item \
         (`lane-<L>.box` under its out directory) and its stop and dead-lane paths send TERM to \
         the pids there that still read as a box.sh. Unset: nothing is filed. Local and queue \
         mode (`BLOOMERY_BOX_LOCAL`, `BLOOMERY_BOX_QUEUE`) file nothing.",
    ),
    runner(
        "BLOOMERY_BOX_QUEUE",
        Some("box.sh"),
        "Mac side: `1` makes a box command a job of the box queue: the tree syncs under the remote \
         directory's lock and `tools/boxq.py` runs the job, which exits with the job's code. A read \
         (`BLOOMERY_BOX_READONLY=1`) never queues; unset or `0` runs the command directly.",
    ),
    runner(
        "BLOOMERY_BOX_READONLY",
        Some("box.sh"),
        "Mac side: `1` runs a read on the box beside a sitting, with no guard and no sync.",
    ),
    runner(
        "BLOOMERY_BOX_WAIT",
        Some("box.sh"),
        "Seconds the box guard waits for the timing lease and the holds before a command \
         gives up; `0` does not wait.",
    ),
    runner(
        "BLOOMERY_BUILD_BOUND",
        Some("ref/build-dump-draft.sh"),
        "Seconds `tools/ref/build-dump-draft.sh` lets the reference build run.",
    ),
    runner(
        "BLOOMERY_BUNDLE_BAND_DUMP",
        None,
        "The model attention gate's handshake with the child it runs of its own test \
         binary: the file the bundle-band child writes.",
    ),
    runner(
        "BLOOMERY_CARD",
        Some("box.sh"),
        "Mac side: the card a box command runs on, `3090`, `a6000` or `both`; \
         `tools/box.sh` sets `CUDA_VISIBLE_DEVICES` from it.",
    ),
    runner(
        "BLOOMERY_CARD_ALL_EXPERTS",
        Some("ref/needs-env.sh"),
        "Set on a machine whose one card holds every routed expert of Qwen3.8-Flash-Next's NVFP4 split (a GB10 Spark, \
         128 GB): `gate-gpu-qwen4exp-nvfp4` runs only where it is set, and a batch defers it by name elsewhere.",
    ),
    runner(
        "BLOOMERY_CHAIN_ATTN_LAYERS",
        None,
        "`gate_deepseek41_chain_attn`: the layers it checks, all of them when unset.",
    ),
    runner(
        "BLOOMERY_CHAIN_ATTN_MUTATE",
        None,
        "`gate_deepseek41_chain_attn`'s FAIL-first tool: the one thing a pin covers that it \
         breaks.",
    ),
    runner(
        "BLOOMERY_CHECK_TARGET",
        Some("mac-check.sh"),
        "Mac side: the Linux triple `tools/mac-check.sh`'s `check` and `combos` cross-check \
         for, `x86_64-unknown-linux-gnu` when unset; its `lint` and a `combos` ledger refuse \
         another.",
    ),
    runner(
        "BLOOMERY_CPU_BUSY_COMMS",
        Some("ref/lease.sh"),
        "The process names the lease's cpu guard counts as another tenant's work.",
    ),
    runner(
        "BLOOMERY_CPU_BUSY_PCT",
        Some("ref/lease.sh"),
        "The percent of one cpu past which the lease's cpu guard marks a row \
         `[cpu-busy]`.",
    ),
    runner(
        "BLOOMERY_DECODE_N",
        Some("ref/decode-measure.sh"),
        "The CPU and GPU runners: the decode steps each timed run generates.",
    ),
    runner(
        "BLOOMERY_DECODE_TOKENS",
        Some("ref/decode-measure.sh"),
        "The CPU runners: the prompt ids, the profile's `REF_TOKENS` when unset.",
    ),
    runner(
        "BLOOMERY_DEPTHS",
        Some("ref/depth-decode.sh"),
        "`tools/ref/depth-decode.sh`: the depths both engines run at.",
    ),
    runner(
        "BLOOMERY_DRY",
        Some("ref/depth-ds41.sh"),
        "The timing runners: `1` prints each command line (and a derivation) and exits \
         before the lease; the recipes then build nothing.",
    ),
    runner(
        "BLOOMERY_DUMP_BOUND",
        Some("ref/dump.sh"),
        "Seconds the reference dumpers (`tools/ref/dump.sh`, `dump-draft.sh`) may run.",
    ),
    runner(
        "BLOOMERY_ENGRAM_LEASE",
        Some("ref/engram-rate.sh"),
        "`tools/ref/engram-rate.sh`'s handshake with `engram-rate`: `1` says the run holds \
         the lease; without it every line is stamped outside one.",
    ),
    runner(
        "BLOOMERY_FLASH_SIMD_CHILD_DUMP",
        None,
        "The model attention gate's handshake with the child it runs of its own test \
         binary: the file the lever child writes.",
    ),
    LeverSpec {
        name: "BLOOMERY_GATE_BOUND",
        class: Class::R,
        kind: Kind::Count {
            min: 1,
            max: u64::MAX,
            trim: false,
        },
        default: Unset::Is("900"),
        doc: "Seconds a gate runner (`tools/gate.sh`, `tools/gpu-gate.sh`, \
              `tools/host-gate.sh`) lets its binary run before it kills it: a hung gate \
              ends red. The runners' one parser is `tools/gate-bound.sh`; this row takes \
              what it takes and unset means what it means.",
        site: Site::Env {
            script: Some("gate-bound.sh"),
        },
    },
    LeverSpec {
        name: "BLOOMERY_GATE_BUILD_BOUND",
        class: Class::R,
        kind: Kind::Count {
            min: 1,
            max: u64::MAX,
            trim: false,
        },
        default: Unset::Is("2700"),
        doc: "Seconds `tools/gate.sh` lets its cargo build run before it kills it, apart from \
              the test's own bound (`BLOOMERY_GATE_BOUND`): a build that waits on the \
              build-directory lock and compiles a cold cone is slow, not hung. The parser is \
              `tools/gate-bound.sh`'s, as for the test's bound.",
        site: Site::Env {
            script: Some("gate-bound.sh"),
        },
    },
    runner(
        "BLOOMERY_GATE_CARD",
        Some("gpu-gate.sh"),
        "The card a GPU gate runs on and whose gate lock it takes: `3090`, `a6000` or \
         `any`.",
    ),
    runner(
        "BLOOMERY_GATE_V41_LOAD",
        Some("gpu-gate.sh"),
        "`1`: the run loads the whole V4.1 model, so it also takes the box-wide V4.1 load lock, \
         queued in arrival order and taken before its card lock(s); a recipe in the \
         justfile's `v41-load` group exports it.",
    ),
    runner(
        "BLOOMERY_MAC_REMOTE",
        Some("mac-remote.sh"),
        "`tools/mac-remote.sh`: the alias of the pool Mac to run on, a file of \
         `~/.config/bloomery/mac-remote.d`; unset, the runner picks a free slot across the pool.",
    ),
    runner(
        "BLOOMERY_V41_CLASS",
        Some("gpu-gate.sh"),
        "The V4.1 load queue's class for a loader (`tools/gpu-gate.sh`): `1` the lead's landing \
         batch (`tools/gate-batch.sh --ledger` sets it), `2` round gates (the default), `3` \
         sittings, `4` idle fill. The queue serves (class, arrival) ascending and nothing \
         preempts; any other value is refused (64).",
    ),
    runner(
        "BLOOMERY_V41_PRED_S",
        Some("gpu-gate.sh"),
        "Whole seconds: the calling recipe's median wall, written into a loader's V4.1 queue \
         ticket as `pred_s`; unset, `-`.",
    ),
    runner(
        "BLOOMERY_GATE_STACKS",
        Some("gpu-gate.sh"),
        "Whole seconds: the GPU gate's binary runs under `tools/ref/stack-watch.sh`, which dumps \
         its thread stacks and ends it once its output has stopped that long; unset, no watch.",
    ),
    runner(
        "BLOOMERY_GATE_GDB",
        Some("gpu-gate.sh"),
        "`1`: the GPU gate's binary runs under gdb, which prints every thread's stack when a \
         signal ends it, and the runner exits 128 + the signal; `0` or unset, no gdb.",
    ),
    runner(
        "BLOOMERY_BOX_CARD",
        Some("box.sh"),
        "The card `tools/box.sh` put in view, passed to the box side: `3090`, `a6000`, \
         `both`, or `local` in its local mode; `tools/gpu-gate.sh` takes that card's gate lock \
         (both locks for `both`, the one local lock for `local`), and under `local` \
         `tools/ref/timing-card.sh` times this machine's card.",
    ),
    runner(
        "BLOOMERY_GB10_BOX",
        Some("release/publish.sh"),
        "Mac side: the ssh host of the GB10 that built the release's GB10 archive; \
         `tools/release/publish.sh` verifies, fetches and uploads it from there. Unset: the \
         release carries no GB10 archive.",
    ),
    runner(
        "BLOOMERY_GEN_CTX",
        Some("ref/depth-qwen3moe.sh"),
        "The Qwen3 depth, ncu and nsys runners: the context every arm of ours runs at.",
    ),
    runner(
        "BLOOMERY_GEN_WARM",
        Some("ref/depth-ds41.sh"),
        "The depth runners: steps our arm runs before its statistics (`--warm`).",
    ),
    runner(
        "BLOOMERY_GEN_PAIR",
        Some("ref/depth-glm5next.sh"),
        "The GLM depth runner: our arm runs --pair after its steps (the verify's V2/S1).",
    ),
    runner(
        "BLOOMERY_GEN_PLACE",
        Some("ref/depth-ds41.sh"),
        "The depth runners: our arms' `--place` — `a` (plan (a), the A6000; the \
         default), `gate` (the gate plan, the 3090) or `bp` (plan (b′), both cards; only \
         under `BLOOMERY_TIMING_CARDS=a6000+3090`, which refuses `a` and `gate`); anything \
         else is refused.",
    ),
    runner(
        "BLOOMERY_PREHEAT",
        Some("ref/depth-ds41.sh"),
        "The depth runners: `0` turns off the preheat of each reference arm's host \
         set (its same-lease A/B); `1` is the default; anything else is refused.",
    ),
    runner(
        "BLOOMERY_GIT_COMMIT",
        Some("box.sh"),
        "The commit a build names, in `bloomery-serve --version` and `/props` \
         (`serve::COMMIT`, set by the serve crate's build script): `tools/box.sh` exports \
         the synced tree's commit (`-dirty` when the Mac tree held changes) into every box \
         command, and `tools/release/build.sh` sets it for its cargo build; unset, the \
         serve build script reads `git rev-parse --short=8 HEAD`, else `unknown`. \
         `tools/ref/depth-ds41.sh` reads it too. A running binary never reads it.",
    ),
    runner(
        "BLOOMERY_GPU_ARMS",
        Some("ref/depth-gpu.sh"),
        "`tools/ref/depth-gpu.sh`: its arms, in the order a lease runs them.",
    ),
    runner(
        "BLOOMERY_GPU_HOLD_BEAT",
        Some("gate-batch.sh"),
        "Mac side: seconds between `tools/gate-batch.sh --ledger`'s refreshes of its GPU \
         hold on the box (default 60; the box calls a hold stale after 300); the \
         self-test shortens it.",
    ),
    runner(
        "BLOOMERY_HOLD_OWNER",
        Some("box.sh"),
        "The owner of the hold a sitting put up (`/root/bloomery-<owner>-hold`); a box \
         command that names it passes that hold.",
    ),
    runner(
        "BLOOMERY_HOLD_YIELD",
        Some("gpu-gate.sh"),
        "`1`: the run yields to a batch's GPU hold at its first probe (rc 75 naming the \
         hold's owner) instead of waiting for it; `tools/gate-batch.sh --round-ledger` sets it \
         in each item's box env. Unset or `0`: the run waits for the hold, as today.",
    ),
    runner(
        "BLOOMERY_HOST_BOUND",
        Some("ref/host-rate.sh"),
        "Seconds one thread count of `tools/ref/host-rate.sh` may run.",
    ),
    runner(
        "BLOOMERY_HOST_LEASE",
        Some("ref/host-rate.sh"),
        "`tools/ref/host-rate.sh`'s handshake with `bench_v41_host`: `1` says the run \
         holds the lease.",
    ),
    runner(
        "BLOOMERY_IK_CTX",
        Some("ref/ik-draft.sh"),
        "`tools/ref/ik-draft.sh`: both engines' context.",
    ),
    runner(
        "BLOOMERY_IK_DRAFT_PARAMS",
        Some("ref/ik-draft.sh"),
        "`tools/ref/ik-draft.sh`: the reference engine's `--draft-params`.",
    ),
    runner(
        "BLOOMERY_IK_NCMOE",
        Some("ref/ik-draft.sh"),
        "`tools/ref/ik-draft.sh`: replaces `--n-cpu-moe` in the reference engine's GPU \
         flags for one run.",
    ),
    runner(
        "BLOOMERY_IK_SPEC",
        Some("ref/ik-draft.sh"),
        "`tools/ref/ik-draft.sh`: the reference engine's DSpark stage, `dspark[:k=v,...]`.",
    ),
    runner(
        "BLOOMERY_KV_PREFETCH_CHILD",
        None,
        "The model attention gate's handshake with the child it runs of its own test \
         binary: set, the process is the prefetch child.",
    ),
    runner(
        "BLOOMERY_LCPP_NCMOE",
        Some("ref/ik-draft.sh"),
        "`tools/ref/ik-draft.sh`: replaces `--n-cpu-moe` in llama.cpp's flags for one run.",
    ),
    runner(
        "BLOOMERY_LEASE_HELD",
        Some("ref/lease-hold.sh"),
        "The pid of the `tools/ref/lease-hold.sh` a command runs under: `lease_take` \
         refuses a second lease inside it.",
    ),
    runner(
        "BLOOMERY_LEASE_POLL",
        Some("ref/lease-probe.sh"),
        "Seconds between the box guard's polls; the tools' stub tests shorten it.",
    ),
    runner(
        "BLOOMERY_MIN_FREE_GIB",
        Some("gate-batch.sh"),
        "Mac side: the disk floor in GiB `tools/gate-batch.sh` checks before its first lane \
         and `tools/mac-check.sh` before its first cargo command; unset means each script's \
         own constant.",
    ),
    runner(
        "BLOOMERY_MT_CHILD_DUMP",
        None,
        "The model threading gate's handshake with the child it runs of its own test \
         binary: the file the child writes.",
    ),
    runner(
        "BLOOMERY_NCU_COUNT",
        Some("ref/ncu-gpu.sh"),
        "The ncu runner: launches of each kernel it profiles.",
    ),
    runner(
        "BLOOMERY_NCU_DEPTHS",
        Some("ref/ncu-gpu.sh"),
        "The ncu runner's depth form: the depths it profiles.",
    ),
    runner(
        "BLOOMERY_NCU_FORM",
        Some("ref/ncu-gpu.sh"),
        "The ncu runner's form: `generate`, `gemm`, `ds41pp` or `q3pp`.",
    ),
    runner(
        "BLOOMERY_NCU_GEMM_ARM",
        Some("ref/ncu-gpu.sh"),
        "The ncu runner's gemm form: the `gate_p8` grouped-GEMM arm it profiles.",
    ),
    runner(
        "BLOOMERY_NCU_KERNEL",
        Some("ref/ncu-gpu.sh"),
        "The ncu runner's q3pp form: the kernel entry it profiles.",
    ),
    runner(
        "BLOOMERY_NCU_KERNELS",
        Some("ref/ncu-gpu.sh"),
        "The ncu runner: the kernel names it profiles, a regular expression.",
    ),
    runner(
        "BLOOMERY_NCU_LAYER",
        Some("ref/ncu-gpu.sh"),
        "The ncu runner's prompt forms: the layer whose launch it profiles.",
    ),
    runner(
        "BLOOMERY_NCU_METRICS",
        Some("ref/ncu-gpu.sh"),
        "The ncu runner: metrics asked for by name instead of the stall reasons.",
    ),
    runner(
        "BLOOMERY_NCU_MODE",
        Some("ref/ncu-gpu.sh"),
        "The ncu runner's depth form: `graph` or `eager`.",
    ),
    runner(
        "BLOOMERY_NCU_PER_STEP",
        Some("ref/ncu-gpu.sh"),
        "The ncu runner's depth form: launches its filter takes per step, which the \
         launch skip is counted in.",
    ),
    runner(
        "BLOOMERY_NCU_PROMPT",
        Some("ref/ncu-gpu.sh"),
        "The ncu runner's prompt forms: the prompt length.",
    ),
    runner(
        "BLOOMERY_NCU_SECTIONS",
        Some("ref/ncu-gpu.sh"),
        "The ncu runner: the report sections it asks for.",
    ),
    runner(
        "BLOOMERY_NCU_SKIP",
        Some("ref/ncu-gpu.sh"),
        "The ncu runner: launches it skips before the first it profiles, instead of the \
         skip it derives.",
    ),
    runner(
        "BLOOMERY_NCU_SKIP_STEPS",
        Some("ref/ncu-gpu.sh"),
        "The ncu runner's depth form: steps past the depth it skips before it profiles.",
    ),
    runner(
        "BLOOMERY_NCU_SOURCE",
        Some("ref/ncu-gpu.sh"),
        "The ncu runner's gemm and q3pp forms: `1` adds the per-instruction source page.",
    ),
    runner(
        "BLOOMERY_NCU_STALLS",
        Some("ref/ncu-gpu.sh"),
        "The ncu runner: the stall reasons it asks for.",
    ),
    runner(
        "BLOOMERY_NCU_TOTALS",
        Some("ref/ncu-gpu.sh"),
        "The ncu runner's summary: set, the kernels' total time, largest first, instead \
         of each kernel's medians.",
    ),
    runner(
        "BLOOMERY_NCU_UNIT",
        Some("ref/ncu-gpu.sh"),
        "The ncu runner's q3pp form: the ubatch it profiles.",
    ),
    runner(
        "BLOOMERY_NSYS_BLOCKED_US",
        Some("ref/nsys-ds41.sh"),
        "`tools/ref/nsys-ds41.sh`: the microseconds past which a host call is tabled as \
         blocked.",
    ),
    runner(
        "BLOOMERY_NSYS_DEPTHS",
        Some("ref/nsys-gpu.sh"),
        "The nsys runner: the depths, or prompt lengths in its prefill form, it traces.",
    ),
    runner(
        "BLOOMERY_NSYS_FORM",
        Some("ref/nsys-gpu.sh"),
        "The nsys runners: `prefill` traces a prompt instead of decode steps.",
    ),
    runner(
        "BLOOMERY_NSYS_LAST",
        Some("ref/nsys-ds41.sh"),
        "`tools/ref/nsys-ds41.sh`: the replays it tables.",
    ),
    runner(
        "BLOOMERY_NSYS_LAYER",
        Some("ref/nsys-ds41.sh"),
        "`tools/ref/nsys-ds41.sh`'s prefill form: the layer-batch it tables alone.",
    ),
    runner(
        "BLOOMERY_NSYS_MARKER",
        Some("ref/nsys-gpu.sh"),
        "`tools/ref/nsys-gpu.sh`: the kernel whose launches cut the trace into steps.",
    ),
    runner(
        "BLOOMERY_NSYS_MODE",
        Some("ref/nsys-gpu.sh"),
        "`tools/ref/nsys-gpu.sh`: `graph` or `eager`.",
    ),
    runner(
        "BLOOMERY_NSYS_N",
        Some("ref/nsys-gpu.sh"),
        "The nsys runners: the steps each traced run generates.",
    ),
    runner(
        "BLOOMERY_NSYS_TOP",
        Some("ref/nsys-gpu.sh"),
        "The nsys runners: the kernel rows they table.",
    ),
    runner(
        "BLOOMERY_OTHER_STRICT",
        Some("ref/lease.sh"),
        "The lease's guards: `1` ends a timed run (75) where another tenant's work would \
         only mark its row.",
    ),
    runner(
        "BLOOMERY_OXIDE_BACKENDS",
        Some("oxide-backend.sh"),
        "The directory the patched cuda-oxide backends are built into and read from, one \
         `<rev>/` each; unset, `${CARGO_HOME:-~/.cargo}/cuda-oxide-bloomery`.",
    ),
    runner(
        "BLOOMERY_PACK_BUDGET",
        Some("gate-batch.sh"),
        "Mac side: the host byte budget `tools/gate-batch.sh` gives the pack of lane X (bytes, a \
         positive integer); unset means the script's `PACK_BUDGET` constant; the self-test \
         shrinks it.",
    ),
    runner(
        "BLOOMERY_PLACEMENT_TABLE",
        None,
        "The placement gate: `1` prints the per-tensor table too.",
    ),
    runner(
        "BLOOMERY_PROFILE_CHILD_DUMP",
        None,
        "The model profile gate's handshake with the child it runs of its own test \
         binary: the file the child writes.",
    ),
    LeverSpec {
        name: "BLOOMERY_PROFILE_DEPTH",
        class: Class::R,
        kind: Kind::Count {
            min: 0,
            max: u64::MAX,
            trim: false,
        },
        default: Unset::Means("no prompt first"),
        doc: "`tools/ref/profile-measure.sh`: the profiled run first prefills depth-decode's \
              prompt of this depth.",
        site: Site::Env {
            script: Some("ref/profile-measure.sh"),
        },
    },
    runner(
        "BLOOMERY_PROFILE_LEVELS",
        Some("ref/profile-measure.sh"),
        "`tools/ref/profile-measure.sh`: the `BLOOMERY_PROFILE` levels it runs.",
    ),
    runner(
        "BLOOMERY_Q_CARD",
        Some("box.sh"),
        "Mac side, with `BLOOMERY_BOX_QUEUE=1`: the queue lane of the job, `3090`, `a6000`, `both`, \
         `any` (the daemon picks a card; the job reads it from `$SIGNALBOX_CARD`) or `none`. Unset: \
         `BLOOMERY_CARD`, else the card a `BLOOMERY_GATE_CARD` entry of `BLOOMERY_BOX_ENV` names, else `any` for a \
         command whose every device run is an `any`-form `tools/gpu-gate.sh` call, else `3090`; both set to \
         different values is refused.",
    ),
    runner(
        "BLOOMERY_Q_CLASS",
        Some("box.sh"),
        "Mac side, with `BLOOMERY_BOX_QUEUE=1`: the job's class, `lead`, `round`, `sitting` or \
         `fill`, which sets its priority on the daemon; unset means `round`.",
    ),
    runner(
        "BLOOMERY_Q_JOB",
        Some("ref/lease.sh"),
        "The box queue job a runner started under; the lease's trace names it.",
    ),
    runner(
        "BLOOMERY_Q_PRED_S",
        Some("box.sh"),
        "Mac side, with `BLOOMERY_BOX_QUEUE=1`: the job's predicted wall in whole seconds (at \
         least 1); unset prints `predicted unknown` and the job goes without one.",
    ),
    runner(
        "BLOOMERY_RATE_CORE",
        Some("ref/qdot-rate.sh"),
        "`tools/ref/qdot-rate.sh`: the core its bench runs on.",
    ),
    runner(
        "BLOOMERY_REF_BACKEND",
        Some("ref/argmax.sh"),
        "The reference dumpers: `cuda` runs the reference engine's GPU build instead of \
         its CPU one.",
    ),
    runner(
        "BLOOMERY_REF_BATCH_PREFILL",
        Some("ref/argmax.sh"),
        "`tools/ref/argmax.sh`: `1` lets the reference engine prefill in a batch instead \
         of a step per id.",
    ),
    runner(
        "BLOOMERY_REF_BUILD",
        Some("ref/dump.sh"),
        "The reference build a dumper's set records in its manifest; the dump runners \
         hand it to the C++ dumpers.",
    ),
    runner(
        "BLOOMERY_REF_CTX",
        Some("ref/argmax.sh"),
        "`tools/ref/argmax.sh`: the reference engine's context, the profile's when unset.",
    ),
    runner(
        "BLOOMERY_REF_GEN",
        Some("ref/argmax.sh"),
        "`tools/ref/argmax.sh`: greedy steps appended after the prompt's.",
    ),
    runner(
        "BLOOMERY_REF_MODEL_PROFILE",
        Some("box.sh"),
        "The profile `BLOOMERY_REF_MODEL` was exported under: `tools/box.sh` exports it, \
         and `tools/ref/ref-paths.sh` refuses a script that picks another.",
    ),
    runner(
        "BLOOMERY_REF_TOKENS",
        Some("ref/dump.sh"),
        "The token ids `tools/ref/dump.sh` dumps a reference set of, instead of the \
         profile's.",
    ),
    runner(
        "BLOOMERY_REF_TOKENS_SHA256",
        Some("ref/dump.sh"),
        "The sha256 of the ids file a dumper's set names, which the dump runners hand the \
         C++ dumpers.",
    ),
    runner(
        "BLOOMERY_REF_WRITE",
        Some("ref/dump.sh"),
        "The C++ dumpers' write guard: `1` writes the set, `0` or unset refuses; the dump \
         runners set it.",
    ),
    runner(
        "BLOOMERY_ROUTER_BOUND",
        Some("ref/router-trace.sh"),
        "Seconds `tools/ref/router-trace.sh` lets its harness run.",
    ),
    runner(
        "BLOOMERY_ROUTER_IDS_MD5",
        Some("ref/router-trace.sh"),
        "The md5 of the ids a router trace ran over, which `tools/ref/router-trace.sh` \
         hands its harness.",
    ),
    runner(
        "BLOOMERY_ROUTER_WRITE",
        Some("ref/router-trace.sh"),
        "The router trace harness's write guard: `1` writes the set, `0` or unset \
         refuses.",
    ),
    runner(
        "BLOOMERY_SELFTEST_LOAD",
        Some("bloomery/selftest.py"),
        "The self-test's synthetic 1-minute load. `tools/bloomery/selftest.py` reads it \
         instead of the host load, and `tools/boxq.py`'s load-ready case sets it. The \
         shell caller does not.",
    ),
    runner(
        "BLOOMERY_SELFTEST_NCPU",
        Some("bloomery/selftest.py"),
        "The self-test's synthetic CPU count, read with the synthetic load. Unset, the \
         host's CPU count.",
    ),
    runner(
        "BLOOMERY_SPLITK_BAND_DUMP",
        None,
        "The model attention gate's handshake with the child it runs of its own test \
         binary: the file the split-K band child writes.",
    ),
    runner(
        "BLOOMERY_SPLITK_CHILD_DUMP",
        None,
        "The model attention gate's handshake with the child it runs of its own test \
         binary: the file the split-K child writes.",
    ),
    LeverSpec {
        name: "BLOOMERY_TIER",
        class: Class::R,
        kind: Kind::Words(&["real", "fixture"]),
        default: Unset::Is("real"),
        doc: "The tier a box command runs in: `real` is the files the model profiles name, \
              `fixture` the small files `fixture generate` writes, one directory a family. \
              `tools/box.sh` reads it from the environment or `BLOOMERY_BOX_ENV` and resolves the \
              model file and `BLOOMERY_CARD_BUDGET` from the family's fixture \
              (`tools/ref/ref-paths.sh`; a family with none is refused, never run on its real file); \
              `tools/gate-batch.sh --tier` passes it to every item, `tools/gpu-gate.sh` takes no \
              V4.1 load lock for a resolved fixture, and a gate's clauses run by their tag \
              (`crates/gpu-gates/src/tier.rs`). It is part of every ledger key.",
        site: Site::Env {
            script: Some("box.sh"),
        },
    },
    runner(
        "BLOOMERY_TIMING_CARDS",
        Some("ref/timing-card.sh"),
        "`a6000+3090`: a depth runner times both cards for the separate A6000+3090 table \
         (`tools/ref/timing-card.sh`); unset, one card.",
    ),
    runner(
        "BLOOMERY_TIMING_GPU",
        Some("ref/timing-card.sh"),
        "The card a timed run takes, the A6000 when unset; in `tools/box.sh`'s local mode a \
         UUID `nvidia-smi` lists, the machine's one card when unset \
         (`tools/ref/timing-card.sh`).",
    ),
    runner(
        "BLOOMERY_VISION_BOUND",
        Some("ref/vision/dump-vision.sh"),
        "Seconds each vision dump run may take.",
    ),
    runner(
        "BLOOMERY_WARM_ROWS",
        Some("ref/cold-blocks.sh"),
        "`tools/ref/depth-ds41.sh` and `depth-qwen3moe.sh` (through `cold-blocks.sh`): `1` runs \
         each of our arms once more on the same ids right before its row (a discarded `PRIME` \
         row) and runs a counted row the cold tag marks once more (`COLD`, then its row or `FAIL \
         rc=cold`); `0` (unset) runs and prints as before; anything else is refused.",
    ),
    runner(
        "BLOOMERY_SESSION_URL",
        Some("release/pass_checks.py"),
        "`tools/release/pass_checks.py`: the URL of the live `bloomery-serve` a release sitting's \
         cell checks, when no URL argument is given.",
    ),
    runner(
        "BLOOMERY_SESSION_PID",
        Some("release/pass_checks.py"),
        "`tools/release/pass_checks.py`: the default of `--pid`, that server's process, probed \
         after every check (alive and not a zombie).",
    ),
    runner(
        "BLOOMERY_SESSION_STDERR",
        Some("release/pass_checks.py"),
        "`tools/release/pass_checks.py`: the default of `--stderr`, the file that server's stderr \
         goes to; its `plan` and `load` records and an off cell's refusal are read from it.",
    ),
    runner(
        "BLOOMERY_SESSION_DIR",
        Some("release/pass_checks.py"),
        "`tools/release/pass_checks.py`: the default of `--dir`, where each check's request and \
         reply are kept, one JSON a check.",
    ),
    LeverSpec {
        name: METAL_BREADCRUMB,
        class: Class::D,
        kind: Kind::Path,
        default: Unset::Means("off: no breadcrumb written"),
        doc: "Metal host crash breadcrumb: names a file each driver-facing call class \
              (`queue`, `set`, `event`, `commit`, `wait`, `teardown`) and each e2e-gate clause \
              appends one fsync'd line to before the call, so the file's last line names the call \
              in flight after a kernel panic; unset it is one branch off the step path.",
        site: Site::Direct {
            at: &[InPlace {
                file: METAL_STREAM,
                round: METAL_BREADCRUMB_STAYS,
            }],
            phase: Phase::Load,
            by: Readers::Every,
        },
    },
];
