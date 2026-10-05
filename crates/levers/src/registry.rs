//! The rows: every `BLOOMERY_*` name the repository sets or reads — the
//! levers, the retired names, and the names that are no lever — each once.
//! A lever row with a file that reads it in place (`Site::Direct`, or a
//! non-empty `left`) is also a line of `tools/levers-direct.txt`, the list
//! `tools/check-levers.sh` holds every in-place environment read to; the test
//! `registry_and_allow_list_agree` holds the two to each other. A name that is
//! no lever names its owner (`Site::Env`), and `tools/check-levers.sh` holds
//! every `BLOOMERY_*` name under `tools/`, in the justfile and in `.cargo/` to
//! a row.

use crate::{Class, InPlace, Kind, LeverSpec, Site, Unset};

pub const THREADS: &str = "BLOOMERY_THREADS";
pub const SPIN: &str = "BLOOMERY_SPIN";
pub const CED: &str = "BLOOMERY_CED";
pub const PREFILL: &str = "BLOOMERY_PREFILL";
pub const PREFILL_GROUP: &str = "BLOOMERY_PREFILL_GROUP";
pub const ENGRAM_HELPER: &str = "BLOOMERY_ENGRAM_HELPER";
pub const STEP_STATS: &str = "BLOOMERY_STEP_STATS";
pub const CARD_BUDGET: &str = "BLOOMERY_CARD_BUDGET";
pub const PIN_MAIN: &str = "BLOOMERY_PIN_MAIN";
pub const DRAFT: &str = "BLOOMERY_DRAFT";
pub const CHECK_FINITE: &str = "BLOOMERY_CHECK_FINITE";
pub const HOST_POPULATE: &str = "BLOOMERY_HOST_POPULATE";
pub const HOST_LOCK: &str = "BLOOMERY_HOST_LOCK";
pub const CARD_DONTNEED: &str = "BLOOMERY_CARD_DONTNEED";
pub const R8: &str = "BLOOMERY_R8";
pub const MTP_HEAD_ROWS: &str = "BLOOMERY_MTP_HEAD_ROWS";
pub const MTP_DRAFT: &str = "BLOOMERY_MTP_DRAFT";
pub const MTP_WINDOWS: &str = "BLOOMERY_MTP_WINDOWS";
pub const RESIDENCY: &str = "BLOOMERY_RESIDENCY";
pub const HOSTSTREAM: &str = "BLOOMERY_HOSTSTREAM";
pub const ROUTE_TRACE: &str = "BLOOMERY_ROUTE_TRACE";
pub const QWEN38_EXPERTS: &str = "BLOOMERY_QWEN38_EXPERTS";
pub const QWEN3_KV: &str = "BLOOMERY_QWEN3_KV";
pub const LANE_PREFETCH: &str = "BLOOMERY_LANE_PREFETCH";

/// The largest `BLOOMERY_PREFILL_GROUP`: the batches a V4.1 prompt group
/// holds at most, which the body's buffers are sized for.
pub const PREFILL_GROUP_MAX: u64 = 8;

/// `BLOOMERY_PREFILL_GROUP` unset: the row's default, the group GLM-5.3's
/// plan reserves a prompt batch's units for.
pub const PREFILL_GROUP_DEFAULT: u64 = 2;

/// `BLOOMERY_LANE_PREFETCH` unset: the row's default, which the host
/// union holds in a binary that does not act on the lever
/// (`model::ops::lane_prefetch`).
pub const LANE_PREFETCH_DEFAULT: bool = false;

/// The rounds that convert the levers still read in place.
const R03: &str = "[03]";
const V2FENCE: &str = "v2fence";
const SESSION: &str = "session";

const OPS: &str = "crates/model/src/ops.rs";
const ATTN: &str = "crates/model/src/arch/deepseek2/attn.rs";
const HYBRID: &str = "crates/gpu/src/hybrid.rs";
const DECODE: &str = "crates/model/src/bin/bloomery-decode.rs";

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
            max: PREFILL_GROUP_MAX,
            trim: false,
        },
        default: Unset::Is("2"),
        doc: "V4.1 and GLM-5.3: the batches a prompt group runs layer by layer, each \
              layer-batch's route enqueued ahead of the previous one's host serve; 1 runs \
              each batch alone, the same-binary arm, and both write the same bits. The \
              `load` line prints `group=`. GLM-5.3's plan reserves the units a group of 2 \
              holds (each about two stream buffers); a group past 2 takes its further units \
              out of the card's margin, refused by name past its free bytes; the `prefill \
              units` line prints what they took and what stayed free.",
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
              that split line per case; `generate_qwen3moe` on a qwen4exp file: the \
              ubatch walk's `stat prompt split` and its per layer-batch `stat \
              prompt lb`. Off, nothing is read.",
        site: Site::Parsed { left: &[] },
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
        kind: Kind::Words(&["lookup", "dspark", "mtp", "off"]),
        default: Unset::Means(
            "the plain path, one token a step; `generate_qwen3moe` on a qwen4exp file and \
             `bloomery-serve-qwen38` under `--place a`: `mtp` when the MTP draft file is there, \
             else the plain path with a `load draft=off` record naming why; the GLM seat of \
             `bloomery-serve`: `mtp` under `--place a` or `bp` on a file of one next-token layer, else the \
             plain path, with a `draft unset` record naming why (`bloomery_levers::glm_unset`)",
        ),
        doc: "`generate_ds41` and `bloomery-serve-ds41`: `lookup` serves an n-gram lookup \
              draft, `dspark` the DSpark draft (`$BLOOMERY_DSPARK_MODEL`), through the skewed \
              two-row pass; `generate_qwen3moe` and `bloomery-serve-qwen38` on a qwen4exp \
              file: `mtp` serves the file's MTP draft (`BLOOMERY_MTP_DRAFT`, else the shared \
              draft file beside the target, its head reduced under `BLOOMERY_MTP_HEAD_ROWS`) \
              through a four-row window; `generate_glm5next` and the GLM seat of `bloomery-serve` on a \
              glm5next file: `mtp` serves the target file's NextN layer as the MTP draft through a \
              two-row window, and `lookup` and `dspark` are refused by name; the greedy ids are the \
              plain run's. A server serves a \
              request that samples or bans an id through plain steps, each step's logits row the \
              target's, and drafts only its greedy requests. `off` is the plain path on a \
              qwen4exp file, the same-binary arm of the unset draft. Unset in \
              `generate_qwen3moe` on a qwen4exp file follows the placement \
              (`bloomery_levers::draft38_unset`): under `--place a` the MTP draft runs when a \
              regular file is where it would be opened (`BLOOMERY_MTP_DRAFT`, else the shared \
              draft beside the target, else the family's path); the plain path runs under \
              `--place gate`, beside `--logits` or `BLOOMERY_ROUTE_TRACE`, with no file there, \
              and when an arm's last four-row window would pass `--ctx` (depth + n + 2 \
              positions) — each printed as a `load draft=off (<why>)` record after the `load` \
              line (`no file at <path>` for the missing file), never a refusal. \
              `bloomery-serve-qwen38` follows the same rule, its conditions the seat's: under \
              `--place a` the draft when its file is there; the plain path under `--place gate`, \
              with no file there, and with stores too short for one window (`--ctx-size` under \
              5: the server takes a window only while its rows fit), with the same record; it \
              has no `--logits` (a request that reads the logits row steps plainly) and takes \
              no route trace. The GLM seat of `bloomery-serve` takes `mtp` and `off`; unset \
              (`bloomery_levers::glm_unset`, a `draft unset` record before the plan) it drafts \
              under `--place a` or `bp` on a file of one next-token layer, and runs the plain path under \
              `--place gate`, on any other file and with stores too short for one window (`--ctx` \
              under 3), never a refusal. Drafting, the load carries the NextN layer as \
              `generate_glm5next`'s does, prints a `load draft=mtp` record after the `load` line, \
              and `/props`' `engine.draft` names it; the plain path prints `load draft=off \
              (<why>)`; `mtp` set is refused by name with stores too short for one window. Every \
              other binary and \
              family refuses each word by name, `off` included.",
        site: Site::Parsed { left: &[] },
    },
    LeverSpec {
        name: MTP_HEAD_ROWS,
        class: Class::C,
        kind: Kind::PathOr(&["full"]),
        default: Unset::Means(
            "the list the model crate ships (Qwen3.8's hangul family, 65,536 rows) when the \
             target's tokenizer is the one its first line names; the full head on any other \
             target, the load's `mtp head` record saying why",
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
        site: Site::Parsed { left: &[] },
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
        site: Site::Parsed { left: &[] },
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
              run that drafts nothing. Off, the chain hands back its ids alone and nothing is \
              kept; the tokens are the same either way.",
        site: Site::Parsed { left: &[] },
    },
    LeverSpec {
        name: ROUTE_TRACE,
        class: Class::D,
        kind: Kind::Path,
        default: Unset::Means("no trace"),
        doc: "`bloomery-serve-ds41`, `generate_qwen3moe` (a qwen4exp file) and \
              `generate_glm5next`: the directory, created at `main` as a new directory \
              (an existing path or a missing parent is refused by name), the host tier \
              writes a route trace into (`crates/gpu/src/host/route_trace.rs`): every \
              position's routed ids per layer and the slot each ran in, as a router set, \
              a `call` row per prompt call. The server needs `BLOOMERY_PREFILL=steps` \
              and no `BLOOMERY_DRAFT`; the generators need their step feed \
              (`--prefill step` / `--prefill steps`) and refuse `--time`, each refused \
              by name otherwise.",
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
        name: HOST_POPULATE,
        class: Class::C,
        kind: Kind::Flag,
        default: Unset::Is("1"),
        doc: "Placed load: the plan's host set is read in with `MADV_POPULATE_READ`; `0` \
              leaves it to fault in, the fresh-fault arm.",
        site: Site::Parsed { left: &[] },
    },
    LeverSpec {
        name: HOST_LOCK,
        class: Class::C,
        kind: Kind::Flag,
        default: Unset::Is("0"),
        doc: "Placed load: after populating, `mlock` the host set for the model's life; an \
              `RLIMIT_MEMLOCK` refusal is an error that names the limit.",
        site: Site::Parsed { left: &[] },
    },
    LeverSpec {
        name: CARD_DONTNEED,
        class: Class::C,
        kind: Kind::Flag,
        default: Unset::Is("1"),
        doc: "Placed load: each uploaded card segment's file pages are dropped right after \
              its upload (`token_embd` and the engram table excepted); `0` keeps them in \
              the page cache.",
        site: Site::Parsed { left: &[] },
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
        site: Site::Parsed { left: &[] },
    },
    LeverSpec {
        name: RESIDENCY,
        class: Class::C,
        kind: Kind::Residency,
        default: Unset::Means(
            "V4.1 follows the placement: `mid-p40-s1` under `--place a` and `bp`, `off` under \
             `gate`, beside `BLOOMERY_CHECK_FINITE=1`, `BLOOMERY_ROUTE_TRACE` or \
             `BLOOMERY_PREFILL=steps`; `generate_qwen3moe` on a qwen4exp file follows its plan: \
             `mid-p<P>-s1` under `--place a`, P half the plan's fewest card experts a layer, \
             `off` under `--place gate`, beside `BLOOMERY_ROUTE_TRACE`, `--prefill step` or \
             `--dump-taps`, on a qwen3moe or qwen35moe file, and on plan (a) when the plan \
             leaves no room for the word, or its host headroom or `MemAvailable` none for its \
             churn pool; `bloomery-serve-qwen38` by the same rule under `--place a` and `off` \
             under `gate`; the GLM seat of `bloomery-serve` `mid-p0-s1` under `--place a` or `bp` where \
             the plan has room, `off` under `gate` (`bloomery_levers::glm_unset`, \
             `glm_residency_at_plan`), with a `residency unset` record; `off` in every other \
             binary",
        ),
        doc: "Adaptive expert residency (`host::swap`): `off` keeps the load's slot map for the \
              model's life; `mid-p<P>-s<S>` runs the residency rule's `mid` parameters over the \
              stage card's routed stacks, the first P seed experts of each layer never a victim \
              and S slots a layer freed for flips in flight. V4.1 under `--place a` and \
              `bp` (`generate_ds41`, `bloomery-serve-ds41`): the load's host set also holds \
              the churn pool, each layer's stage card experts past the first P, refused by \
              name when the plan's host headroom cannot take it, and prints `residency host`; \
              every pass prints `residency pass`; only an explicit call resets it (an arm's \
              clear, the server's `POST /residency/reset`), with a `residency reset` record. \
              Unset follows the placement (`bloomery_levers::residency_unset`): the serving \
              default `mid-p40-s1` under `--place a` and `bp`, and `off` where the machine \
              does not run — `--place gate`, beside `BLOOMERY_CHECK_FINITE=1`, beside \
              `BLOOMERY_ROUTE_TRACE` (a fixed placement's routing) and beside \
              `BLOOMERY_PREFILL=steps` (each prompt id would end a pass the rule counts); both \
              binaries print the word and why as a `residency lever` record before the load. \
              Set, `mid-…` is refused by name under `--place gate`, beside the finite probe, \
              the route trace and the step feed. Qwen3.8 (`generate_qwen3moe` on a qwen4exp \
              file, `--place a` or `gate`, plain or `BLOOMERY_DRAFT=mtp`): unset follows the \
              plan (`bloomery_levers::residency38_unset`, `residency38_at_plan`) — under \
              `--place a` `mid-p<P>-s1`, P half the fewest card experts a layer of the plan \
              the load runs (the plain or the MTP plan, at its `--ctx`), derived at load so \
              the default always fits the plan; `off` under `--place gate`, beside \
              `BLOOMERY_ROUTE_TRACE`, with `--prefill step`, under `--dump-taps`, on a \
              qwen3moe or qwen35moe file, when the plan holds no card expert or its fewest \
              leave no room for P pinned, one spare and one that moves, and when the churn \
              pool at P does not fit the plan's host headroom or what the host's \
              `MemAvailable` leaves past the plan's host need — never a refusal; the word and \
              why print as a `residency unset` record after the `plan` line (on a qwen3moe or \
              qwen35moe file, and under `--dump-taps`, before the load). \
              `mid-p148-s1` is the word a set lever names for plan (a) at the plain plan's \
              count; set, \
              `mid-…` runs the same machine over the card's routed stacks, the load's host set \
              holds the churn pool, refused by name as above, and prints `residency host`, and \
              each arm prints its `residency pass` records after its lines; an arm's clear \
              resets it, with a `residency reset` record; a set word and \
              `why=set` print as a `residency lever` record before the load. There `mid-…` is refused by \
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
        site: Site::Parsed { left: &[] },
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
        site: Site::Parsed { left: &[] },
    },
    LeverSpec {
        name: HOSTSTREAM,
        class: Class::A,
        kind: Kind::OnOff,
        default: Unset::Means(
            "V4.1: on under a residency (`BLOOMERY_RESIDENCY` not `off`), off without one; \
             Qwen3.8: on under `--place a` with a residency, off everywhere else",
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
              stream`, every call `call stream end`. Qwen3.8 (`generate_qwen3moe`) runs the \
              same call mode in its ubatch walk: `on` moves each card layer's pool toward the \
              ubatch's hottest host experts before the layer's card route (a count of at least \
              the walk's `STREAM_FLOOR`), the union and the route under the moved map, and the \
              call's picks stay for the decode; a pass-fed prompt, and one of fewer ids than the \
              floor, opens no call (its rows give an expert fewer counts than the floor, so \
              its pick could admit nothing). \
              Unset there is on under `--place a` with a residency, off everywhere else; \
              `off` is the same-binary arm; `on` is refused by name beside \
              `BLOOMERY_RESIDENCY=off` and on a qwen3moe or qwen35moe file.",
        site: Site::Parsed { left: &[] },
    },
    LeverSpec {
        name: QWEN3_KV,
        class: Class::A,
        kind: Kind::Words(&["f16", "q8_0"]),
        default: Unset::Is("f16"),
        doc: "Qwen3 and Qwen3.6 (`generate_qwen3moe`, `bloomery-serve --model qwen3`): the \
              format the load holds its K/V cache planes in — `q8_0` the two-plane layout \
              (17/16 B a value against f16's 2, both planes quantized together by the one \
              choice), a cache half the bytes wide, so the seat's auto `--ctx` search \
              reaches about 1.88x the f16 answer on the same card. Read once at load: the \
              append quantizes each row and the flash reads it through the format's \
              algebra, so a q8_0 run's bits are its own, never an f16 run's; the budget \
              (`whole_ctx` and the placed searches) counts the format's bytes. The seat's \
              `--cache-type-k` takes the same two words (llama-server's spelling) and \
              wins over the lever; `--cache-type-v` does not exist — both planes quantize \
              together. Refused by name on a qwen4exp (Qwen3.8) file, whose stores carry \
              no q8_0 form. The `load` line prints `cache=`.",
        site: Site::Parsed { left: &[] },
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
              its ubatch arena by it); a token's bits do not depend on it. The `load` line \
              prints `ubatch=`.",
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
        name: "BLOOMERY_DSPARK_CARD",
        class: Class::C,
        kind: Kind::Text,
        default: Unset::Means("the 3090"),
        doc: "`generate_ds41` and `bloomery-serve-ds41` under `BLOOMERY_DRAFT=dspark`: the \
              card the DSpark draft loads on, one card as a `--place` list word names it (a \
              CUDA ordinal, `1` or `cuda1`, or a card name exactly one visible device \
              carries); it may be the target's own card. Unset, the visible device of the \
              fewest usable bytes (the 3090). A word no visible device answers, or a name two \
              carry, is refused by name. Under `--place bp` the draft sits on the expert tier \
              card (the 3090), whose plan reserves its bytes, and a word naming another device \
              is refused.",
        site: Site::Direct {
            at: &[InPlace {
                file: "crates/gpu-gates/src/bin/shared/ds41_dspark.rs",
                round: SESSION,
            }],
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
        "The directory a server's `--hf <repo>[:<quant>]` fetches into, \
         `<dir>/<owner>/<name>/<file>`; `~/.cache/bloomery/hf` when unset.",
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
         unset says so.",
    ),
    path(
        "BLOOMERY_ENGRAM_BIN",
        Some("ref/engram-rate.sh"),
        "The `engram-rate` binary `tools/ref/engram-rate.sh` runs.",
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
        "BLOOMERY_MODEL",
        Some("box.sh"),
        "Mac side, the tool profile a box command runs under (`tools/ref/models/`), which \
         `tools/box.sh` reads and does not export; on the box, the model crate's test \
         harnesses read it as a model file's path.",
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
        "BLOOMERY_V41_MODEL",
        Some("box.sh"),
        "The V4.1 first shard: `tools/box.sh` exports the deepseek41 profile's \
         `V41_MODEL` into every box command whatever profile it picked; a caller's own \
         wins.",
    ),
    path(
        "BLOOMERY_V4_MODEL",
        Some("ref/build-qdot-ref.sh"),
        "The V4-Flash shard the MXFP4 qdot harnesses and gates read.",
    ),
    path(
        "BLOOMERY_VISION_SET",
        Some("ref/vision/dump-vision.sh"),
        "The vision reference set (a directory under `ref-vision`) the vision gates read, \
         and the one `tools/ref/vision/dump-vision.sh` writes.",
    ),
    // ---------------------------------------------- the runners' variables
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
        "BLOOMERY_ARM_BOUND",
        Some("ref/lease.sh"),
        "Seconds one arm of a timed runner may run under the lease before `timeout` ends \
         it.",
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
        "BLOOMERY_BUILD_COMMIT",
        Some("release/build.sh"),
        "The commit a release build's `--version` names, read by cargo at build time \
         (`model_file::version`); `unknown` when unset. A running binary never reads it.",
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
    runner(
        "BLOOMERY_GATE_CARD",
        Some("gpu-gate.sh"),
        "The card a GPU gate runs on and whose gate lock it takes: `3090`, `a6000` or \
         `any`.",
    ),
    runner(
        "BLOOMERY_GATE_V41_LOAD",
        Some("gpu-gate.sh"),
        "`1`: the run loads the whole V4.1 model, so it also takes the box-wide V4.1 load lock \
         after its card lock(s); a recipe in the justfile's `v41-load` group exports it.",
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
        "The card `tools/box.sh` put in view, passed to the box side: `3090`, `a6000` or \
         `both`; `tools/gpu-gate.sh` takes that card's gate lock (both locks for `both`).",
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
        "The commit of the synced tree (`-dirty` when the Mac tree held changes): \
         `tools/box.sh` exports it into every box command; `bloomery-serve`'s build \
         script and `tools/ref/depth-ds41.sh` read it.",
    ),
    runner(
        "BLOOMERY_GPU_ARMS",
        Some("ref/depth-gpu.sh"),
        "`tools/ref/depth-gpu.sh`: its arms, in the order a lease runs them.",
    ),
    runner(
        "BLOOMERY_HOLD_OWNER",
        Some("box.sh"),
        "The owner of the hold a sitting put up (`/root/bloomery-<owner>-hold`); a box \
         command that names it passes that hold.",
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
    runner(
        "BLOOMERY_TIMING_CARDS",
        Some("ref/timing-card.sh"),
        "`a6000+3090`: a depth runner times both cards for the separate A6000+3090 table \
         (`tools/ref/timing-card.sh`); unset, one card.",
    ),
    runner(
        "BLOOMERY_TIMING_GPU",
        Some("ref/timing-card.sh"),
        "The card a timed run takes, the A6000 when unset (`tools/ref/timing-card.sh`).",
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
];
