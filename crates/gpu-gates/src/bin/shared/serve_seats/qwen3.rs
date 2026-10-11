//! `bloomery-serve --model qwen3` — the llama-server-compatible HTTP API on
//! the single-card Qwen engines: a qwen3moe file (Qwen3-30B-A3B) or a
//! qwen35moe file (Qwen3.6-35B-A3B), the whole model on one card, under
//! `--place` placed by its plan on the card the word names with the routed
//! experts the card's budget does not hold on the host tier.
//!
//!     bloomery-serve --model qwen3 [-m PATH | --hf <repo>[:<quant>]]
//!                    [--host 127.0.0.1] [--port 8080] [--ctx C] [--place W]
//!                    [--cache-type-k f16|q8_0] [--parallel N] [--queue-depth Q] [--plan]
//!                    [--mmproj FILE | --no-mmproj] [--image-min-tokens N]
//!                    [--image-max-tokens N]
//!
//! The server takes `-m`/`--hf` out before this seat parses (its module
//! doc); `--ctx-size` and `-c` are `--ctx` under llama-server's spellings.
//! Unset, a slot's context is the common rule's answer
//! (`placement::ctx::default_slot_ctx`, the rule every serving seat takes):
//! 4096 a slot, or — where the card holds more — the largest multiple of 1024
//! up to the largest the load fits (the file's trained context at most, 4096
//! for a file that states none) whose plan holds at most `MARGIN` fewer card
//! expert bytes than the plan at 4096 and no more NVMe expert bytes. A
//! whole-card load, which the whole-fit verdict at 4096 a slot takes (its
//! arena and reserve counted), keeps every expert on the card at every
//! context, so the verdict alone bounds it; a placed load, under `--place` or
//! the plan a run without it falls to, is the plan's own trade of experts for
//! positions, held to that margin. One `ctx` line on stderr names the rule
//! (`card`, `base`, `fit` or `margin`), the slot context, the slots and their
//! total, the cap, the fit and the margin's terms. The model is opened as
//! `generate_qwen3moe` opens it — qwen3moe through `Qwen3moeModel::open` at
//! its levers' options, qwen35moe through `Qwen35moeModel::open` with the
//! tensor-core decode flash and the levers' ubatch — in graph mode with the
//! step and every pass captured, so a request's ids are the CLI's: the
//! server feeds the prompt less its last id as one prompt call — both bodies
//! by the seat's schedule (the ubatch walk from nine rows on — qwen3moe's
//! GEMM, qwen35moe's wide, no unit under the gemv arm's cut — passes below)
//! — then steps the last, which `generate_qwen3moe --last-step` does too.
//! Its first shard gives the vocabulary, `tokenizer.chat_template` the chat
//! template and `general.name` the alias. One slot at the whole `--ctx`
//! while `--parallel` names no count beside a set flag, else two slots by
//! default (`--parallel N`), one only at `--parallel 1`; `--plan` prints the
//! records the decisions before the load make (the `ctx` line, the whole-fit
//! verdict or the `plan` record, the `parallel` line) and exits 0 before the
//! model is opened on a card; sampling is the sampler
//! crate's chain with no repetition penalty, `temperature <= 0` the
//! engine's argmax.
//!
//! `--place W` opens the model as `generate_qwen3moe --place W` does
//! (`qwen3moe_place`): `W` a placement word of `generate::Place` (`a`,
//! `gate`, or a card list of one card; a tier card is refused by name), the
//! plan's `plan` record on stderr before the `load` record, the card budget
//! `BLOOMERY_CARD_BUDGET`'s and the host set's load the host levers' (each
//! of them refused by name without `--place`). A
//! plan with no host expert loads the whole model on the plan's card when the
//! whole-fit verdict takes it; any other runs the step graph through the
//! host tier (a plan with no host expert whose whole load does not fit runs
//! it with nothing to serve) and every prompt as eager units through the
//! host tier's batch port (`app::Prompt for Body`: the wide units from
//! `GEMM_FROM` rows on a placed load, passes below), no pass captured.
//!
//! With `--place` unset the model opens as `generate_qwen3moe` opens it,
//! on `a`'s card, the largest visible card: the whole file while the
//! whole-fit verdict there takes it (one stderr line with its terms), else
//! the placed plan on that card, its `plan` record naming why
//! (`whole_does_not_fit`). Each `--ctx` search prints one line more: its
//! probes and the census readings it took. The seat takes the common rule
//! every serving seat takes (`generate::Place::choose_untiered`: the body
//! serves no tier card): a `place unset` record names the word `a`
//! and why — `one card`, or on two `the body serves no tier card` — before
//! the first search.
//!
//! A request keeps the longest prefix it shares with what the slot holds:
//! the session over the model answers the server's keep queries and cuts
//! (`app::Keep` — a qwen3moe file's [`Body`] grants every held position,
//! the caches being per-position; a qwen35moe file's [`Body35`] grants the
//! nearest checkpoint a marked prompt call took at or below the ask, every
//! ubatch of positions and each call's end, and its prompt calls take those
//! checkpoints), so the resend of a conversation keeps everything up to
//! where it diverges and a request that extends the held sequence keeps all
//! of it. The prompt call runs the ubatch walk from
//! `app::arch::qwen3moe::GEMM_FROM` rows on and passes below, so the rows a
//! kept prefix leaves behind are a whole fresh run's. The seat runs no
//! prompt cache: a save and a resume are refused by name. The
//! `load` record (architecture, resident bytes — every slot's sequences —,
//! context, slots, each slot's context, layers, ubatch,
//! the step graph's nodes, the load's wall) then the `listening` record go
//! to stderr (`record::BLOOMERY_SERVE_QWEN3`). An engine error ends the
//! process with the crash block and exit code 70, as every seat's.
//!
//! `--parallel N` (`-np N`) serves N slots, on both files N
//! resident sequences inside the one model (`GpuModel::add_slots` on the
//! session: a qwen3moe file's per-position caches, a qwen35moe file's
//! `Slot35` — its K/V planes, its delta layers' recurrent states and conv
//! rings, its held count and its checkpoints). `--parallel` naming no
//! count, the seat serves one slot beside a set `--ctx` (the flag is one
//! request's context, `placement::ctx::slots_of`; one line on stderr says so) and its
//! default two over the auto choice. Under more slots than one the context
//! is split across them
//! as llama-server splits it with `-np N` and no `-kvu`: the `--ctx` the
//! flags named (or, when unset, the default slot context times N) is the
//! total, each slot `total / N` rows rounded down — a cache row is the granularity, so the
//! floor is exact, and N slots never hold more rows than the one-sequence
//! load — and `--parallel N` is the slot count itself, the split's bytes the
//! bound. A whole-card load and a placed one — under `--place`, or the plan a
//! run without it falls to when the whole file does not fit the card's free
//! bytes — split alike: what the file loads as is decided once, before the
//! open, at the total (the whole-fit verdict there, which counts every
//! slot's stores at its share, or the placed plan made there, whose card
//! bytes count every slot's rows and, past the live slot, every other slot's
//! fixed stores as the `resident slots` reserve), and the open makes no fit
//! call of its own, so a total whose whole load does not fit never opens
//! whole because one slot's share would. A placed open holds the slots'
//! stores to what the plan counts for them by name and prints both
//! (`q3place::open_qwen3_slots`, `q3place::open_qwen35_slots`). A qwen35moe
//! sequence's checkpoints pin host bytes, up to the body's budget at a slot's
//! context (`q3place::checkpoints_fit`'s one line names it): a placed plan
//! carries every sequence's as its
//! host's `checkpoints` reserve, and a whole-card load, which has no host
//! plan, is refused by name before it loads when the host's available bytes
//! do not hold them (`q3place::checkpoints_fit`, one stderr line either way).
//! One token a slot a round, the sequences switched by pointer exchange, no
//! park and no re-prefill; the engine is the seat's own (`SeatEngine`), and
//! the server steps every running slot in one call — one pass of the busy
//! rows on a whole-card load, whose body runs several slots' rows as one
//! pass (a qwen3moe file's `Body` and a qwen35moe file's `Body35`,
//! `GpuModel::step_slots`, every row-wise launch once over them all), else a
//! select and a step a row (the seat's fallback loop), a placed load's.
//! Under `BLOOMERY_STEP_STATS=1` each round of several slots
//! prints a `slots round` record naming its command, rows, passes and the
//! slots the seat serves. `--parallel 1` is exactly the one-sequence server.
//! A `parallel` line on stderr before the load names the rule (`slots`), the
//! slots, a slot's context, the total, what set the count (`from`: the
//! `--parallel` flag, a set `--ctx`, the default) and the shape of a round of
//! several slots (`pass`: `one`, or `turns` on a load that keeps routed
//! experts on the host tier). `--queue-depth Q` bounds
//! the requests that wait for a slot; `--park-ram`
//! is refused by name: the resident slots hold no parked state the flag
//! could budget.
//!
//! `--mmproj FILE` (llama-server's spelling, with `--no-mmproj` and the two token limits:
//! [`vision_seat::Flags`]) gives a qwen35moe file (Qwen3.6-35B-A3B, whose projector is a
//! `qwen3vl_merger`) its image input. The encoder file is held against the text model — its rows'
//! width, its vocabulary's three image ids, the IMROPE sections — and refused by name when they
//! differ ([`vision_seat::open_seat`]); on a qwen3moe file (Qwen3-30B-A3B) `--mmproj` is refused
//! by name at the open, before any plan. The tower sits on the smallest card outside the plan, else
//! as a named reserve of the stage card ([`vision_seat::EncoderSeat`]), which every plan and every
//! whole-fit verdict counts and the `--ctx` and `whole fit` lines name. The load is a media load
//! (`Open35::media`): every sequence holds the rotation table an image's positions rewrite, counted
//! a position in the verdict's cache and as a reserve of a placed plan (`q3place::TABLES_RESERVE`).
//! The `vision` record follows the `load` record. The server renders an image part as the model's
//! placeholder and expands it to its grid's span (`serve::media` over [`Seat::media_model`]); a
//! prompt call that holds images encodes each on the tower and reads its rows back, then feeds the
//! prompt with the rows in place of their positions' embeddings ([`Seat::prefill_media`]). An image
//! with no `--mmproj` is the server's own 400 for every seat that takes none.
//!
//! The seat sits behind the `deepseek41` feature, the server surface's
//! scope (`bind`, the `serve` and `sampler` crates); it runs no V4.1 code;
//! a placed load acts on the plan's and the host tier's levers
//! (`BLOOMERY_QWEN3_UBATCH` is read where the load sizes its arena).

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

#[path = "../qwen3moe_place.rs"]
mod q3place;
use app::arch::qwen3moe::GEMM_FROM;
use q3place::{PlaceQ3, Seqs};

use bloomery_gpu::arch::qwen3moe::router::MAX_TOKENS;
use bloomery_gpu::arch::qwen3moe::ubatch::ubatch_size;
use bloomery_gpu::arch::qwen3moe::{Body, Body35, KvQ8, Open35, Qwen35moeModel};
use bloomery_gpu::model::{GpuModel, MAX_PASS_ROWS, Slots, StepMode};
use bloomery_gpu::{Gpu, Qwen3moeModel};
use bloomery_gpu_gates::bind::{
    ChatSurface, Listen, Pass, Seat, SeatEngine, SlotStep, Vocab, bind_server, note_line,
    parallel_line, slots_given,
};
use bloomery_gpu_gates::generate::{CTX_GRAN, Place};
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::vision_seat::{self, Opened, PleImage, TextFacts};
use bloomery_gpu_gates::{GateError, gpu_census, ref_model_path};
use bloomery_levers::Levers;
use gguf::Split;
use model::arch::Arch;
use model::arch::qwen35moe::hparams::Hparams;
use model::placement::PlanLevers;
use model::placement::ctx::SlotCtx;
use model::placement::workstation::{CardSpec, MARGIN};
use runtime::Target as _;
#[cfg(feature = "vision")]
use serve::media::{MediaFeed, SharedMediaModel};
use serve::{CacheNote, EngineProps, Saved, ServeError};
use tokenizer::Tokenizer;
#[cfg(feature = "vision")]
use {
    bloomery_gpu::media::{ImageSpan, MediaSpan},
    runtime::Want,
    std::ops::Range,
};

const NAME: &str = "bloomery-serve-qwen3";

const USAGE: &str = "usage: bloomery-serve --model qwen3 [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C] [--place W] [--cache-type-k f16|q8_0] \
                     [--parallel N] [--queue-depth Q] [--plan] [--mmproj FILE | --no-mmproj] \
                     [--image-min-tokens N] [--image-max-tokens N] [--api-key KEY] \
                     [--api-key-file FNAME]";

/// The context a slot takes before the rule grows it: the base the margin
/// counts against, `generate_qwen3moe`'s default.
const BASE_SLOT_CTX: usize = 4096;

/// The context a slot takes when `--ctx` is unset: the common rule's answer
/// (`placement::ctx::default_slot_ctx`) for `parallel` resident slots, the
/// constants its own — [`BASE_SLOT_CTX`] a slot, steps of
/// [`CTX_GRAN`], a margin of `MARGIN` card expert bytes — over the
/// file's trained context as the cap (the base when it states none). The
/// probes plan the load the file takes on `a`'s card: the whole model while
/// its whole-fit verdict takes the base a slot — weights, the cache of every
/// resident slot, the program's arena at the load's ubatch and the reserve
/// its load keeps free past it, against what the card had — where every
/// routed expert stays on the card and the verdict alone bounds the fit; else
/// the placed plan, under `--place` or the one a run without it falls to,
/// whose card and NVMe expert bytes the margin counts (a plan that keeps
/// every expert opens placed like any other). The
/// load is then made at the total, `parallel` slots of the answer
/// ([`decide`]). One line names the answer, and one more the search's probes
/// and the census readings it took ([`searched_line`]); a qwen35moe file's
/// plans carry its sequences' rows ([`seqs35`]). Under `--mmproj` (`vision`)
/// the sequences hold their rotation tables too, and the encoder's reserve,
/// when its tower sits on the stage card, is in every plan and verdict, named
/// in the line that answers ([`encoder_words`]).
fn default_ctx(
    split: &Split,
    arch: Arch,
    place: Option<Place>,
    parallel: usize,
    levers: &Levers,
    (kv, vision): (KvQ8, Option<&Opened>),
) -> Result<SlotCtx, GateError> {
    let (media, encoder) = images_of(vision);
    let plan_levers = PlanLevers::from_levers(levers)?;
    let rule = q3place::SlotRule {
        base: BASE_SLOT_CTX,
        step: CTX_GRAN,
        margin: MARGIN,
    };
    let before = q3place::reads();
    let found = match arch {
        Arch::Qwen3moe => {
            q3place::default_ctx_qwen3(split, place, rule, (parallel, &plan_levers), kv)?
        }
        _ => q3place::default_ctx_qwen35(
            split,
            &open35(BASE_SLOT_CTX, kv, media)?,
            place,
            rule,
            (seqs35(split, parallel)?, &plan_levers, encoder),
        )?,
    };
    searched_line(found.basis, before);
    Ok(found.slot)
}

/// The line a `--ctx` search (`kind`, whole or placed) prints once it
/// answers: the probes it ran and the census readings it took since
/// `before` — one reading serves every probe, so a search that read the
/// census per probe shows here.
fn searched_line(kind: &str, before: q3place::Reads) {
    let r = q3place::reads().since(before);
    eprintln!(
        "{NAME}: the {kind} --ctx search ran {} probes on {} census reading(s)",
        r.probes, r.censuses
    );
}

/// What a qwen35moe file's load opens at with a `ctx`-row cache: the
/// tensor-core decode flash, the levers' ubatch and the `kv` planes, as
/// `generate_qwen3moe` opens it; `media` when `--mmproj` gave the load an
/// image input: every sequence then holds the rotation table an image's
/// positions rewrite.
fn open35(ctx: usize, kv: KvQ8, media: bool) -> Result<Open35, GateError> {
    Ok(Open35 {
        ctx,
        mma: true,
        ubatch: ubatch_size()?,
        kv,
        media,
    })
}

/// What the plans and the open take of `--mmproj` ([`vision_of`]): the load's `media` flag
/// ([`open35`]) and the bytes the image encoder reserves on the stage card, when its tower sits
/// there.
fn images_of(vision: Option<&Opened>) -> (bool, Option<u64>) {
    (vision.is_some(), vision.and_then(|v| v.seat.reserve))
}

/// What a plan's line says of the encoder's reserve: the reserve named, `""` when the tower sits
/// off the plan or there is none, so a load without `--mmproj` prints its lines as it did.
fn encoder_words(encoder: Option<u64>) -> String {
    encoder.map_or_else(String::new, |bytes| {
        format!(
            ", the {} reserve of {bytes} B counted",
            vision_seat::RESERVE
        )
    })
}

/// A qwen35moe file's `parallel` resident sequences: each takes the
/// checkpoints the seat's kept prefixes come back to ([`Body3::open`] arms
/// them), so each pins its own on the host, priced by the body's own
/// derivation at the ubatch the load is asked for
/// ([`q3place::seqs_qwen35`]).
fn seqs35(split: &Split, parallel: usize) -> Result<Seqs, GateError> {
    q3place::seqs_qwen35(split, ubatch_size()?, parallel)
}

/// Why the seat takes no sequence state: it runs no prompt cache.
const NO_CACHE: &str = "the qwen3 seat runs no prompt cache";

struct Args {
    host: String,
    port: u16,
    ctx: Option<usize>,
    place: Option<Place>,
    /// `--cache-type-k`: the K/V planes' format (llama-server's spelling),
    /// over the `BLOOMERY_QWEN3_KV` lever's word when given.
    cache_type_k: Option<String>,
    /// `--parallel`: the slots the server serves, resident sequences on
    /// either file's load; `None` takes one slot beside a set `--ctx`, else
    /// the default 2 ([`model::placement::ctx::slots_of`]).
    parallel: Option<usize>,
    queue_depth: Option<usize>,
    /// The image-input flags: `--mmproj` is the encoder file whose images a
    /// qwen35moe file's seat serves, none serves no image.
    vision: vision_seat::Flags,
    /// `--api-key`/`--api-key-file`: the keys every request is checked
    /// against ([`serve::flag::ApiKeys`]).
    api_keys: serve::flag::ApiKeys,
    /// `--plan`: the records before the load, then exit.
    plan_only: bool,
}

/// The K/V planes' format this run loads: `--cache-type-k`'s word when given
/// (refused by name on any other), else the `BLOOMERY_QWEN3_KV` lever's,
/// else f16.
fn cache_k(a: &Args, levers: &Levers) -> Result<KvQ8, GateError> {
    let word = a
        .cache_type_k
        .as_deref()
        .or(levers.qwen3_kv_set())
        .unwrap_or("f16");
    KvQ8::parse(word)
        .ok_or_else(|| format!("--cache-type-k takes f16 or q8_0, not {word:?}").into())
}

fn parse_args(args: &[String]) -> Result<Args, GateError> {
    let mut a = Args {
        host: "127.0.0.1".to_owned(),
        port: 8080,
        ctx: None,
        place: None,
        cache_type_k: None,
        // `None` resolves through `placement::ctx::slots_of`: one slot beside a set
        // `--ctx` (the flag is one request's context), else the fixed
        // default 2, not an elastic one — the second slot is a resident
        // sequence the context split bounds (no byte budget to size it
        // from). A lone request pays nothing for it beyond its share of the
        // context: the other slot sits parked empty; `--parallel 1` keeps
        // the one sequence.
        parallel: None,
        queue_depth: None,
        vision: vision_seat::Flags::default(),
        api_keys: serve::flag::ApiKeys::default(),
        plan_only: false,
    };
    let mut it = args.iter().map(String::as_str);
    while let Some(flag) = it.next() {
        if flag == "--help" || flag == "-h" {
            return Err(USAGE.into());
        }
        if flag == "--plan" {
            a.plan_only = true;
            continue;
        }
        // `--no-mmproj` has no value: the image flags read theirs from `it` themselves.
        if a.vision.take(flag, &mut it)? {
            continue;
        }
        let v = it
            .next()
            .ok_or_else(|| format!("{flag} needs a value, or is unknown: {USAGE}"))?;
        match flag {
            "--host" => a.host = v.to_owned(),
            "--port" => a.port = v.parse().map_err(|e| format!("--port {v:?}: {e}"))?,
            f if serve::flag::CTX.contains(&f) => match v.parse::<usize>() {
                Ok(n) if n > 0 => a.ctx = Some(n),
                _ => {
                    return Err(
                        format!("{flag} takes a whole number of at least 1, not {v:?}").into(),
                    );
                }
            },
            "--place" => a.place = Some(Place::parse(v)?),
            "--cache-type-k" => a.cache_type_k = Some(v.to_owned()),
            "--cache-type-v" => {
                return Err(
                    "--cache-type-v does not exist: the qwen3 cache quantizes its K and V \
                     planes together, so --cache-type-k names the one format both take"
                        .into(),
                );
            }
            "--parallel" | "-np" => match v.parse::<usize>() {
                Ok(n) if n > 0 => a.parallel = Some(n),
                _ => {
                    return Err(
                        format!("{flag} takes a whole number of at least 1, not {v:?}").into(),
                    );
                }
            },
            "--queue-depth" => match v.parse::<usize>() {
                Ok(q) => a.queue_depth = Some(q),
                _ => return Err(format!("{flag} takes a whole number, not {v:?}").into()),
            },
            f if serve::flag::KEYS.contains(&f) => a.api_keys.add(f, v)?,
            "--park-ram" => {
                return Err(
                    "--park-ram names a budget of parked states; the qwen3 seat parks none — \
                     its resident slots keep their sequences on the card — so it reads no budget"
                        .into(),
                );
            }
            other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
        }
    }
    Ok(a)
}

/// One of the two bodies the seat serves, opened as `generate_qwen3moe`
/// opens it, its prompt schedule and its keep rule. Both hold resident
/// sequences ([`Slots`]): the session adds and selects them alike.
trait Body3: Slots<Seq: 'static> + app::Prompt + app::Keep + Sized + 'static {
    /// The file's `general.architecture`, as the `load` record names it.
    const ARCH: &'static str;
    /// Whether the body runs several slots' rows as one pass on a
    /// whole-card load (`SlotRows`), through [`Body3::one_pass_round`]:
    /// `Body` (a qwen3moe file) and `Body35` (a qwen35moe file) both do.
    const ONE_PASS: bool = false;
    /// The model of `file` with a `ctx`-row cache by `load`, decided before
    /// the open at the total its `slots` sequences of that shape split
    /// ([`decide`]), the `plan` record on stderr first when a plan loads it;
    /// `planes` is the cache planes' format and whether `--mmproj` gave the load an image input
    /// ([`open35`]'s `media`), which only a qwen35moe body takes.
    fn open(
        file: Split,
        ctx: usize,
        slots: usize,
        load: Q3Load,
        levers: &Levers,
        planes: (KvQ8, bool),
    ) -> Result<GpuModel<Self>, GateError>;
    /// The load runs its routed experts on the host tier too.
    fn placed(m: &GpuModel<Self>) -> Result<bool, GateError>;
    /// Every pass the prompt call replays, captured after the step.
    fn capture_passes(m: &mut GpuModel<Self>) -> Result<(), GateError>;
    /// The prompt call's ubatch size.
    fn ubatch(m: &GpuModel<Self>) -> Result<usize, GateError>;
    /// `ids` from where the model stands by the body's schedule; the argmax
    /// after the last.
    fn prompt(m: &mut GpuModel<Self>, ids: &[u32]) -> Result<u32, GateError>;
    /// Where to cut a prompt call of `first .. end` at `marks` so every run
    /// the cut makes still holds what the body's prompt schedule treats as
    /// one whole call; empty where the body cuts nowhere.
    fn splits(first: usize, end: usize, marks: &[usize]) -> Vec<usize>;
    /// One round of several slots' steps as one pass of the busy rows
    /// ([`Q3::step_slots`]'s one-pass arm), each row's answer into `next`
    /// and each lent logits row filled; the passes it ran back. Refused by
    /// the default: only a body of [`Body3::ONE_PASS`] takes it — `Body` (a
    /// qwen3moe file) and `Body35` (a qwen35moe file), through
    /// [`super::rounds::step_rows_one_pass`] (the seat decides at its open
    /// from [`Body3::ONE_PASS`]).
    fn one_pass_round(
        s: &mut app::Session<Self>,
        rows: &mut [SlotStep],
    ) -> Result<usize, GateError> {
        let _ = (s, rows);
        Err("a round of several slots as one pass: this body runs its slots in turns".into())
    }
    /// One prompt call of `ids` with `images` spliced in, the greedy token after the last id.
    /// Refused by the default: only a qwen35moe file's body takes images.
    #[cfg(feature = "vision")]
    fn prompt_media(
        s: &mut app::Session<Self>,
        ids: &[u32],
        images: &[Placed<'_>],
    ) -> Result<u32, GateError> {
        let _ = (s, ids);
        Err(format!(
            "a prompt with {} images: a {} file takes no image input",
            images.len(),
            Self::ARCH
        )
        .into())
    }
}

/// One image of a media call as a body takes it: the positions its span holds in the call's ids,
/// its rows (`n_embd` bf16 values a position) and its token grid `(nx, ny)`, `nx` tokens along a
/// line and `ny` lines (the merger's `n_llm_w` by `n_llm_h`).
#[cfg(feature = "vision")]
struct Placed<'a> {
    at: Range<usize>,
    rows: &'a [u16],
    grid: (usize, usize),
}

impl Body3 for Body {
    const ARCH: &'static str = "qwen3moe";
    const ONE_PASS: bool = true;

    /// As `load` decided before the open, making no fit call of its own: the
    /// whole model on its card, or the placed plan made at the total, its
    /// load holding a sequence of `ctx` rows for each of `slots`
    /// (`q3place::open_qwen3_slots`).
    fn open(
        file: Split,
        ctx: usize,
        slots: usize,
        load: Q3Load,
        levers: &Levers,
        (kv, media): (KvQ8, bool),
    ) -> Result<Qwen3moeModel, GateError> {
        // `vision_of` refuses `--mmproj` on this file before anything is planned.
        if media {
            return Err("a qwen3moe load takes no image input".into());
        }
        let opts = Qwen3moeModel::lever_opts(ctx, kv)?;
        match load {
            Q3Load::Whole(card) => Ok(Qwen3moeModel::open(
                Gpu::open_card(card.name, card.device)?,
                file,
                opts,
            )?),
            Q3Load::Placed {
                q,
                total,
                why,
                whole,
            } => {
                let plan = q.plan(total, &PlanLevers::from_levers(levers)?)?;
                q.record(&plan, why).eprint();
                match whole {
                    Some(card) => Ok(Qwen3moeModel::open(
                        Gpu::open_card(card.name, card.device)?,
                        file,
                        opts,
                    )?),
                    None => q3place::open_qwen3_slots(file, &plan, slots, opts, levers.host()),
                }
            }
        }
    }

    fn placed(m: &Qwen3moeModel) -> Result<bool, GateError> {
        Ok(m.body(NAME)?.placed().is_some())
    }

    fn capture_passes(m: &mut Qwen3moeModel) -> Result<(), GateError> {
        m.capture_prefill()?;
        Ok(())
    }

    fn ubatch(m: &Qwen3moeModel) -> Result<usize, GateError> {
        Ok(m.ubatch()?)
    }

    /// The session's own schedule (`app::Prompt for Body`): the GEMM walk
    /// from [`GEMM_FROM`] rows on, passes below; on a placed load passes
    /// at every length.
    fn prompt(m: &mut Qwen3moeModel, ids: &[u32]) -> Result<u32, GateError> {
        Ok(<Body as app::Prompt>::prompt(m, ids)?)
    }

    /// Nowhere: the caches are per-position, so every position the body
    /// holds is keepable as it stands and a cut never needs the calls on
    /// either side of it to be whole prompt calls of their own — a token's
    /// values do not depend on its ubatch.
    fn splits(_first: usize, _end: usize, _marks: &[usize]) -> Vec<usize> {
        Vec::new()
    }

    /// The busy rows as one pass ([`super::rounds::step_rows_one_pass`]).
    fn one_pass_round(
        s: &mut app::Session<Body>,
        rows: &mut [SlotStep],
    ) -> Result<usize, GateError> {
        super::rounds::step_rows_one_pass(s, rows)
    }
}

impl Body3 for Body35 {
    const ARCH: &'static str = "qwen35moe";
    const ONE_PASS: bool = true;

    /// As `load` decided before the open, making no fit call of its own: the
    /// whole model on its card, or the placed plan made at the total, its
    /// machine carrying the sequences' rows, its load holding a sequence of
    /// `ctx` rows for each of `slots` (`q3place::open_qwen35_slots`).
    fn open(
        file: Split,
        ctx: usize,
        slots: usize,
        load: Q3Load,
        levers: &Levers,
        (kv, media): (KvQ8, bool),
    ) -> Result<Qwen35moeModel, GateError> {
        let o = open35(ctx, kv, media)?;
        let mut m = match load {
            Q3Load::Whole(card) => {
                Qwen35moeModel::open(Gpu::open_card(card.name, card.device)?, file, o)?
            }
            Q3Load::Placed {
                q,
                total,
                why,
                whole,
            } => {
                let plan = q.plan(total, &PlanLevers::from_levers(levers)?)?;
                q.record(&plan, why).eprint();
                match whole {
                    Some(card) => {
                        Qwen35moeModel::open(Gpu::open_card(card.name, card.device)?, file, o)?
                    }
                    None => q3place::open_qwen35_slots(file, &plan, slots, o, levers.host())?,
                }
            }
        };
        // The seat keeps prefixes, so its prompt calls take the checkpoints
        // of their marks.
        let (_, _, body) = m.body_parts(NAME)?;
        body.set_checkpoints(true);
        Ok(m)
    }

    fn placed(m: &Qwen35moeModel) -> Result<bool, GateError> {
        Ok(m.body(NAME)?.placed().is_some())
    }

    fn capture_passes(m: &mut Qwen35moeModel) -> Result<(), GateError> {
        m.capture_rows::<2>()?;
        m.capture_rows::<3>()?;
        m.capture_rows::<4>()?;
        m.capture_rows::<5>()?;
        m.capture_rows::<6>()?;
        m.capture_rows::<7>()?;
        m.capture_rows::<8>()?;
        Ok(())
    }

    fn ubatch(m: &Qwen35moeModel) -> Result<usize, GateError> {
        Ok(m.ubatch()?)
    }

    /// The session's own schedule (`app::Prompt for Body35`): the wide
    /// ubatch walk from [`GEMM_FROM`] rows on, passes below.
    fn prompt(m: &mut Qwen35moeModel, ids: &[u32]) -> Result<u32, GateError> {
        Ok(<Body35 as app::Prompt>::prompt(m, ids)?)
    }

    /// The marks inside the call where both runs hold at least
    /// [`GEMM_FROM`] ids, so each run is the wide walk whose bits are the
    /// uncut call's.
    fn splits(first: usize, end: usize, marks: &[usize]) -> Vec<usize> {
        let mut at = Vec::new();
        let mut last = first;
        for &u in marks {
            if u >= last + GEMM_FROM && u + GEMM_FROM <= end {
                at.push(u);
                last = u;
            }
        }
        at
    }

    /// The busy rows as one pass ([`super::rounds::step_rows_one_pass`]).
    fn one_pass_round(
        s: &mut app::Session<Body35>,
        rows: &mut [SlotStep],
    ) -> Result<usize, GateError> {
        super::rounds::step_rows_one_pass(s, rows)
    }

    /// The session's media call ([`app::Session::prompt_media`]): each image's span of the call
    /// takes its rows in place of its positions' embeddings and shifts the sequence's rotation
    /// table by its grid, an `(nx, ny)` of positions in raster order. The architecture's own fact
    /// is the grid; the rows are the tower's.
    #[cfg(feature = "vision")]
    fn prompt_media(
        s: &mut app::Session<Body35>,
        ids: &[u32],
        images: &[Placed<'_>],
    ) -> Result<u32, GateError> {
        let spans: Vec<ImageSpan<'_>> = images
            .iter()
            .map(|p| ImageSpan {
                span: MediaSpan {
                    at: p.at.clone(),
                    rows: p.rows,
                },
                grid: p.grid,
            })
            .collect();
        Ok(s.prompt_media(ids, &spans, Want::Argmax)?.argmax())
    }
}

/// The session on the engine thread, the positions its cache was sized
/// for, and what the seat decided once at its open: whether it runs every
/// round of several slots as one pass, and whether it counts those rounds.
struct Q3<B: Body3> {
    s: app::Session<B>,
    ctx: usize,
    /// Whether a round of several slots runs as one pass of the busy rows
    /// ([`Q3::step_slots`]): a whole-card load of more than one resident
    /// slot of a body that runs several slots' rows as one pass
    /// ([`Body3::ONE_PASS`]) — a placed load keeps the fallback loop (its
    /// body serves each slot's rows in its own pass), as does a body with no
    /// one pass of its slots, and a load of one slot never runs a round of
    /// several.
    one_pass: bool,
    /// [`Seat::step_stats`]: the `BLOOMERY_STEP_STATS` the binary parsed.
    stats: bool,
    /// The resident image encoder under `--mmproj` ([`open_encoder`]).
    #[cfg(feature = "vision")]
    vision: Option<vision_seat::Resident>,
}

impl<B: Body3> Q3<B> {
    /// The model of the file at `path` in graph mode by `load`
    /// ([`Body3::open`]), its step captured and, unless placed, its passes,
    /// `slots` resident sequences parked after them (each capturing on its
    /// first use), the session over it; the `load` record on stderr. The
    /// opened body is held to `pass`, the shape the `parallel` line named.
    fn open(
        path: &Path,
        ctx: usize,
        slots: usize,
        pass: Pass,
        load: Q3Load,
        levers: &Levers,
        (kv, vision): (KvQ8, Option<Opened>),
    ) -> Result<Q3<B>, GateError> {
        let t = Instant::now();
        let file = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mut m = B::open(file, ctx, slots, load, levers, (kv, vision.is_some()))?;
        m.set_mode(StepMode::Graph);
        let nodes = m.capture_step()?;
        let placed = B::placed(&m)?;
        let ran = pass_of::<B>(slots, placed);
        if ran != pass {
            return Err(format!(
                "the `parallel` line named pass={} and the load runs pass={}: the plan's host \
                 experts and the opened body's placement disagree",
                pass.word(),
                ran.word()
            )
            .into());
        }
        if !placed {
            B::capture_passes(&mut m)?;
        }
        let at = u32::try_from(ctx).map_err(|_| format!("--ctx {ctx} passes u32"))?;
        let mut s = app::Session::from_model(m, at);
        if slots > 1 {
            s.add_slots(slots)?;
        }
        Record::new(&record::LOAD_QWEN3)
            .w("arch", B::ARCH)
            .u("resident_bytes", s.model().resident_bytes())
            .w("cache", kv.name())
            .u("ctx", ctx)
            .u("slots", slots)
            .u("slot_ctx", ctx)
            .u("layers", s.model().layers().len())
            .u("ubatch", B::ubatch(s.model())?)
            .u("graph_nodes", nodes)
            .f("load_s", t.elapsed().as_secs_f64())
            .eprint();
        // The tower after the model's captures, as the V4.1 seat opens its encoder.
        #[cfg(feature = "vision")]
        let vision = open_encoder(&s, vision)?;
        #[cfg(not(feature = "vision"))]
        let _ = vision;
        Ok(Q3 {
            s,
            ctx,
            one_pass: ran.together(slots),
            stats: levers.step_stats(),
            #[cfg(feature = "vision")]
            vision,
        })
    }
}

/// The resident image encoder of `--mmproj` ([`vision_seat::Resident`]), its tower loaded on the
/// card the plan seated; the `vision` record that follows the load names the card, which of the
/// two seats it is and the tower's bytes.
#[cfg(feature = "vision")]
fn open_encoder<B: Body3>(
    s: &app::Session<B>,
    at: Option<Opened>,
) -> Result<Option<vision_seat::Resident>, GateError> {
    let Some(v) = at else {
        return Ok(None);
    };
    let t = Instant::now();
    let resident = vision_seat::Resident::open(s.model().gpu().context(), &v)?;
    resident.announce(&v, vision_seat::ScratchHome::Own, t);
    Ok(Some(resident))
}

impl<B: Body3> Seat for Q3<B> {
    fn pos(&self) -> usize {
        self.s.model().pos() as usize
    }

    fn ctx_max(&self) -> usize {
        self.ctx
    }

    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        <B as Body3>::prompt(self.s.model_mut(), ids)
    }

    /// The resident encoder's media description ([`vision_seat::Resident::media`]); `None`
    /// without `--mmproj`, a seat that takes no image.
    #[cfg(feature = "vision")]
    fn media_model(&self) -> Option<SharedMediaModel> {
        self.vision.as_ref().map(vision_seat::Resident::media)
    }

    /// Each feed's image encoded on the resident encoder and read back as bf16, one row a cell of
    /// its grid and nothing around them ([`vision_seat::merged_cells`]: the tokens carry the
    /// delimiters; [`vision_seat::Resident::merged_rows`]), then one prompt call with the images
    /// spliced in ([`Body3::prompt_media`]), the model's card bound back between the two.
    #[cfg(feature = "vision")]
    fn prefill_media(&mut self, ids: &[u32], feeds: &[MediaFeed]) -> Result<u32, GateError> {
        let Some(v) = self.vision.as_mut() else {
            return Err("no --mmproj: this seat serves no image".into());
        };
        let vision_seat::SpanRows { rows, ats, .. } = v.merged_rows(
            self.s.model().gpu().context(),
            feeds
                .iter()
                .map(|f| vision_seat::Image::of(f.at, &f.prepared)),
        )?;
        // Each span borrows its own range of the call's one buffer of rows; its grid is the
        // image's merged-token plan, `nx` along a line (the width).
        let images: Vec<Placed<'_>> = ats
            .iter()
            .zip(feeds)
            .map(|(a, f)| {
                let plan = &f.prepared.patches.plan;
                Placed {
                    at: a.positions.clone(),
                    rows: &rows[a.rows.clone()],
                    grid: (plan.n_llm_w, plan.n_llm_h),
                }
            })
            .collect();
        B::prompt_media(&mut self.s, ids, &images)
    }

    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        Ok(self.s.model_mut().step(&[last])?)
    }

    fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError> {
        Ok(self.s.model().logits_into(row)?)
    }

    /// The resident sequences the seat made at its open
    /// (`app::Session::slots`): `--parallel`.
    fn slots(&self) -> usize {
        self.s.slots()
    }

    /// The session's slot (`app::Session::select_slot`): the model exchanges
    /// its live sequence with the slot's parked state — pointer moves.
    fn select(&mut self, slot: usize) -> Result<(), GateError> {
        Ok(self.s.select_slot(slot)?)
    }

    /// The lever the binary parsed ([`Q3::stats`]).
    fn step_stats(&self) -> bool {
        self.stats
    }

    /// One round of several slots ([`super::rounds::step_round`]): one pass
    /// of the busy rows while the open decided so ([`Q3::one_pass`], through
    /// [`Body3::one_pass_round`] — a whole-card load of a body with one pass,
    /// the rows' answers and lent logits rows from the pass's own per-row
    /// heads, slot 0 left selected), else the seat's fallback — a select and
    /// a step a row — the placed load's round (its body serves each slot's
    /// rows in its own pass), every round of a body with no one pass of its
    /// slots, and the round a load of one slot never runs.
    fn step_slots(&mut self, rows: &mut [SlotStep]) -> Result<(), String> {
        let one_pass = self.one_pass;
        super::rounds::step_round(
            self,
            one_pass,
            rows,
            |q, rows| B::one_pass_round(&mut q.s, rows),
            |_| Ok(()),
        )
    }

    /// The model's reset ([`GpuModel::reset`]): these bodies never hold a
    /// verify's rows, so the session's own reset adds nothing to it.
    fn reset(&mut self) -> Result<(), GateError> {
        Ok(self.s.model_mut().reset()?)
    }

    /// The session's cut (`app::Keep`), which takes only a position
    /// [`Seat::keep`] granted; any other is refused by name.
    fn rollback(&mut self, pos: u32) -> Result<(), GateError> {
        Ok(self.s.cut(pos)?)
    }

    /// The body's rule: every held position, or for a body with checkpoints
    /// the nearest at or below `n`, with the rule's sentence when it keeps
    /// less.
    fn keep(&self, n: usize) -> (usize, Option<String>) {
        self.s.keep_query(n)
    }

    fn splits(&self, first: usize, end: usize, marks: &[usize]) -> Vec<usize> {
        B::splits(first, end, marks)
    }

    /// Never asked: the seat's prompt cache budget is 0.
    fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
        Err(format!("a snapshot: {NO_CACHE}").into())
    }

    fn resume(&mut self, _state: &dyn Saved) -> Result<(), GateError> {
        Err(format!("a resume: {NO_CACHE}").into())
    }

    /// A prefix kept less of than shared is a `cache reuse` record; every
    /// other note prints as its line.
    fn note(note: &CacheNote) {
        note_line(NAME, note);
    }
}

/// What the file loads as, decided once before the open at the total
/// context — the context its whole-fit verdict and its plan count — and
/// carried into it ([`Body3::open`]), so the open makes no fit call of its
/// own ([`decide`]).
enum Q3Load {
    /// The whole model on this card, `a`'s: `--place` unset, and the
    /// whole-fit verdict at the total on it takes the whole load.
    Whole(CardSpec),
    /// The placed plan made at `total` positions on `q`'s card: `--place`'s
    /// (no `why`), or `a`'s card's when the whole load does not fit at the
    /// total (`why` names that, `q3place::WHY_NOT_WHOLE`). Its open is the
    /// placed body, its host set whatever the plan holds, none included;
    /// under `--place` whose plan holds no host expert, `whole` is the card
    /// the whole model opens on instead, when the whole-fit verdict takes it
    /// (the plan record prints either way).
    Placed {
        q: Box<PlaceQ3>,
        total: usize,
        why: Option<&'static str>,
        whole: Option<CardSpec>,
    },
}

impl Q3Load {
    /// The `plan` record a placed load prints before its open
    /// ([`Body3::open`]); a whole load has none.
    fn print_plan(&self, levers: &Levers) -> Result<(), GateError> {
        if let Q3Load::Placed { q, total, why, .. } = self {
            let plan = q.plan(*total, &PlanLevers::from_levers(levers)?)?;
            q.record(&plan, *why).eprint();
        }
        Ok(())
    }

    /// Whether the open is the placed body ([`Body3::placed`] reads the same
    /// fact off the opened body): a placed load, whether its plan holds host
    /// experts or none. A load that opens the whole model is not.
    fn placed_body(&self) -> bool {
        matches!(self, Q3Load::Placed { whole: None, .. })
    }
}

/// The shape of a round of `slots` slots on body `B` whose load `placed` its
/// experts on the host tier or not: one pass of the busy rows on a body that
/// runs one ([`Body3::ONE_PASS`]) beside no host tier, else the fallback loop
/// (a select and a step a slot). The one owner of the rule, which the
/// `parallel` line names ahead of the load and the open holds the load to.
fn pass_of<B: Body3>(slots: usize, placed: bool) -> Pass {
    Pass::of(slots, B::ONE_PASS && !placed)
}

/// What the `arch` file loads as at `total` positions for `parallel`
/// resident sequences ([`Q3Load`]): under `--place`, the plan on its card,
/// opening the whole model on that card when the plan holds no host expert
/// and the whole-fit verdict takes the load (`PlaceQ3::whole_card_qwen3`,
/// `PlaceQ3::whole_card_qwen35`: no line on stderr), else the placed body;
/// else the whole-fit verdict at the total (`q3place::unplaced_qwen3_slots`,
/// `q3place::unplaced_qwen35_slots`, its line on stderr) — the whole model on
/// `a`'s card, or the placed plan on it. A qwen35moe file's plans carry
/// its sequences' rows ([`seqs35`]) and their arena is a slot's load's
/// (`q3place::share`). Made at the total, never at a slot's share: a card
/// whose whole load fits a slot's rows and not the total's would open whole,
/// and the slots added after the open would hold stores the verdict never
/// counted; the verdict's own cache term is every slot's stores at its share
/// of the total. Under `--mmproj` (`vision`) the load is a media load and the
/// encoder's reserve, when its tower sits on the stage card, is in the
/// verdict and in the plan.
fn decide(
    split: &Split,
    arch: Arch,
    total: usize,
    (place, levers): (Option<Place>, &PlanLevers),
    parallel: usize,
    (kv, vision): (KvQ8, Option<&Opened>),
) -> Result<Q3Load, GateError> {
    let (media, encoder) = images_of(vision);
    let unplaced = match (arch, place) {
        (Arch::Qwen35moe, Some(p)) => {
            let o = q3place::share(&open35(total, kv, media)?, total, parallel);
            let q = PlaceQ3::qwen35_seqs(split, p, o, seqs35(split, parallel)?, encoder)?;
            let plan = q.plan(total, levers)?;
            let whole = q.whole_card_qwen35(split, &plan, (total, parallel), o)?;
            return Ok(Q3Load::Placed {
                q: Box::new(q),
                total,
                why: None,
                whole,
            });
        }
        (Arch::Qwen35moe, None) => q3place::unplaced_qwen35_slots(
            split,
            &open35(total, kv, media)?,
            seqs35(split, parallel)?,
            encoder,
        )?,
        (_, Some(p)) => {
            let q = PlaceQ3::qwen3(split, p, total, kv)?;
            let plan = q.plan(total, levers)?;
            let whole = q.whole_card_qwen3(split, &plan, (total, parallel))?;
            return Ok(Q3Load::Placed {
                q: Box::new(q),
                total,
                why: None,
                whole,
            });
        }
        (_, None) => q3place::unplaced_qwen3_slots(split, total, parallel, kv)?,
    };
    Ok(match unplaced {
        q3place::Unplaced::Whole(card) => Q3Load::Whole(card),
        q3place::Unplaced::Placed(q) => Q3Load::Placed {
            q,
            total,
            why: Some(q3place::WHY_NOT_WHOLE),
            whole: None,
        },
    })
}

/// The least context a resident slot loads at, one rule for both loads and
/// both files: [`MAX_TOKENS`] rows. A whole-card load captures its widest
/// pass (`capture_prefill`, `capture_rows`, [`MAX_TOKENS`] rows), whose
/// capture writes that many rows into the cache — a cache of fewer rows is
/// refused by the launcher, deep in the load. A placed load captures no
/// pass; its prompt runs as eager passes of up to [`MAX_PASS_ROWS`] ids, and
/// a slot of fewer rows cannot hold one whole pass.
const SLOT_CTX_MIN: usize = MAX_TOKENS;
const _: () = assert!(SLOT_CTX_MIN == MAX_PASS_ROWS);

/// The rows each of `parallel` slots loads with at a total context of
/// `ctx` (the module doc's `--parallel` half): the context split across
/// them, each `ctx / parallel` rows rounded down (a cache row the
/// granularity, the floor exact; N slots never more rows than the
/// one-sequence load, so never more stores than the verdict or the plan at
/// the total counts), a split that leaves a slot under [`SLOT_CTX_MIN`] rows
/// refused by name before anything is decided or loaded — one slot (the one
/// sequence of the whole context, the count a set `--ctx` with no
/// `--parallel` serves) held to the same floor, so a too-small flag is a
/// named refusal on every count.
fn slot_ctx(ctx: usize, parallel: usize) -> Result<usize, GateError> {
    let slot_ctx = if parallel <= 1 { ctx } else { ctx / parallel };
    if slot_ctx < SLOT_CTX_MIN {
        return Err(format!(
            "--parallel {parallel} of a --ctx of {ctx}: a slot's context of {slot_ctx} rows is \
             below the {SLOT_CTX_MIN} rows a slot loads at (the widest pass a whole-card load \
             captures, one whole prompt pass of a placed load's eager passes of up to \
             {MAX_PASS_ROWS} ids); give a larger --ctx or a smaller --parallel"
        )
        .into());
    }
    Ok(slot_ctx)
}

/// What `--mmproj` opened for the plans: the encoder file held against the text model's
/// embedding width, its vocabulary's image ids and its IMROPE sections, and seated
/// ([`vision_seat::open_seat`]); `None` without `--mmproj`. A qwen3moe file (Qwen3-30B-A3B)
/// has no image input, so `--mmproj` there is refused by name, and so is a build without the
/// `vision` feature, whose tower is not built.
fn vision_of(
    a: &Args,
    arch: Arch,
    (path, split): (&Path, &Split),
    token: &dyn Fn(u32) -> Option<String>,
) -> Result<Option<Opened>, GateError> {
    if matches!(arch, Arch::Qwen3moe) {
        return match a.vision.wanted()? {
            Some(file) => Err(format!(
                "--mmproj {}: {} is a qwen3moe file (Qwen3-30B-A3B), which has no image input; \
                 this seat serves images on a qwen35moe file (Qwen3.6-35B-A3B)",
                file.display(),
                path.display()
            )
            .into()),
            None => Ok(None),
        };
    }
    #[cfg(not(feature = "vision"))]
    if a.vision.wanted()?.is_some() {
        return Err(
            "--mmproj: this build takes no images (built without the vision feature)".into(),
        );
    }
    let hp = Hparams::read(split)?;
    // Every qwen35moe attention layer rotates by IMROPE over the file's sections
    // (`qwen35moe::spec::read`).
    let facts = TextFacts {
        n_embd: hp.n_embd,
        imrope: Some(hp.rope_sections),
        ple: PleImage::Absent,
    };
    vision_seat::open_seat(
        &a.vision,
        a.vision.mmproj()?,
        a.place.unwrap_or(Place::A),
        &[vision::arch::qwen3vl::PROJECTOR_TYPE],
        &facts,
        token,
    )
}

/// Loads the model and serves until the listener or the engine fails;
/// `Ok` carries why the server ended.
pub fn run(args: &[String]) -> Result<ServeError, GateError> {
    // A placed load (`--place`) acts on the placement's levers; set beside
    // an unplaced one, they are refused by name.
    let placed = args.iter().any(|a| a == "--place");
    let mut acts_on = vec![bloomery_levers::QWEN3_KV, bloomery_levers::STEP_STATS];
    if placed {
        acts_on.extend(q3place::PLACED_LEVERS);
    }
    let levers = bloomery_levers::at_main(&acts_on)?;
    record::at_main(NAME, record::BLOOMERY_SERVE_QWEN3);
    let a = parse_args(args)?;
    let kv = cache_k(&a, &levers)?;
    let path = ref_model_path()?;
    let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let first = split
        .shard(0)
        .ok_or_else(|| format!("{} opened with no shard", path.display()))?;
    let arch = Arch::detect(first).map_err(|e| format!("{}: {e}", path.display()))?;
    if !matches!(arch, Arch::Qwen3moe | Arch::Qwen35moe) {
        return Err(format!(
            "{} is a {} file; the qwen3 seat serves qwen3moe and qwen35moe files",
            path.display(),
            arch.name()
        )
        .into());
    }
    // The vocabulary before the plans: `--mmproj`'s encoder is held against it, and its bytes
    // are a reserve of every plan that follows.
    let vocab = Arc::new(Vocab::new(Tokenizer::from_gguf(&path)?)?);
    let token = |id| vocab.tokenizer().text(id).map(str::to_owned);
    let vision = vision_of(&a, arch, (&path, &split), &token)?;
    let ChatSurface {
        template,
        name: alias,
    } = ChatSurface::read(
        &path,
        split.value("tokenizer.chat_template"),
        split.value("general.name"),
        None,
        "qwen3",
    )?;
    // The slot count before the context it sizes: a set `--ctx` with no
    // `--parallel` is one request's context — one slot at the whole of it
    // (`placement::ctx::slots_of`).
    let (slots, from) = slots_given(a.parallel, a.ctx, 2)?;
    // The common unset rule, once, on one census reading: the load below
    // still takes `a`'s card (`PlaceQ3::unplaced_on`), the rule's own pick
    // for a body with no tier card; its `place unset` record is printed here.
    let census = gpu_census::census()?;
    q3place::choose(a.place, &census)?.record().eprint();
    // A set `--ctx` is the total the slots split; the default is a slot's
    // context, the load's total the slots of it.
    let ctx = match a.ctx {
        Some(c) => c,
        None => {
            let found = default_ctx(&split, arch, a.place, slots, &levers, (kv, vision.as_ref()))?;
            eprintln!("{}", found.line(slots));
            // The `ctx` line's words stay the rule's; the reserve the
            // search counted is named on a line of its own.
            let (_, encoder) = images_of(vision.as_ref());
            eprintln!(
                "{NAME}: --ctx defaults to {} positions a slot, {} in all{}",
                found.ctx,
                slots * found.ctx,
                encoder_words(encoder)
            );
            slots * found.ctx
        }
    };
    // Resident slots on both files: the model loads with a slot's share and
    // parks the rest of the slots after its captures.
    let slot_ctx = slot_ctx(ctx, slots)?;
    // What the file loads as is decided here, once, at the total: the open
    // makes no fit call of its own (`decide`).
    let load = decide(
        &split,
        arch,
        ctx,
        (a.place, &PlanLevers::from_levers(&levers)?),
        slots,
        (kv, vision.as_ref()),
    )?;
    if matches!(arch, Arch::Qwen35moe) {
        q3place::checkpoints_fit(seqs35(&split, slots)?, slot_ctx, load.placed_body())?;
    }
    // The shape of a round of several slots, from the load's body
    // ([`Q3Load::placed_body`]); the open holds the opened body to it.
    let placed = load.placed_body();
    let pass = match arch {
        Arch::Qwen3moe => pass_of::<Body>(slots, placed),
        _ => pass_of::<Body35>(slots, placed),
    };
    // The slots the seat serves, one line before the load (`from` names what
    // set the count).
    parallel_line(slots, slot_ctx, from, pass);
    if a.plan_only {
        // The records the load would print first are out; nothing was
        // opened on a card.
        load.print_plan(&levers)?;
        std::process::exit(0);
    }
    drop(split);
    let open = path.clone();
    let device = match &load {
        Q3Load::Whole(card) => card.name,
        Q3Load::Placed { q, .. } => q.name(),
    }
    .to_owned();
    let engine = match arch {
        Arch::Qwen3moe => SeatEngine::spawn(
            move || Q3::<Body>::open(&open, slot_ctx, slots, pass, load, &levers, (kv, vision)),
            slot_ctx,
            vocab,
            device,
            EngineProps::default(),
            0,
        )?,
        _ => SeatEngine::spawn(
            move || Q3::<Body35>::open(&open, slot_ctx, slots, pass, load, &levers, (kv, vision)),
            slot_ctx,
            vocab,
            device,
            EngineProps::default(),
            0,
        )?,
    };
    // The seat's own slots are the engine's: the seat made them at its open,
    // and the server steps them together.
    let server = bind_server(
        engine,
        Listen {
            host: &a.host,
            port: a.port,
            alias,
            path: &path,
            template,
            slot_save_path: None,
            api_keys: a.api_keys,
            parallel: slots,
            queue_depth: a.queue_depth,
        },
    )?;
    Record::new(&record::LISTENING_QWEN3)
        .w("arch", arch.name())
        .u("ctx", slot_ctx)
        .u("slots", slots)
        .u("slot_ctx", slot_ctx)
        .w("addr", server.local_addr()?)
        .eprint();
    Ok(server.run())
}
