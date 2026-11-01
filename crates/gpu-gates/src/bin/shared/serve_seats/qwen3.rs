//! `bloomery-serve --model qwen3` — the llama-server-compatible HTTP API on
//! the single-card Qwen engines: qwen3moe (Qwen3-30B-A3B), qwen35moe
//! (Qwen3.6-35B-A3B), or dense qwen35, the whole model on one card, under
//! `--place` placed by its plan on the card the word names with the routed
//! experts the card's budget does not hold on the host tier.
//!
//!     bloomery-serve --model qwen3 [-m PATH | --hf <repo>[:<quant>]]
//!                    [--host 127.0.0.1] [--port 8080] [--ctx C] [--place W]
//!                    [--cache-type-k f16|q8_0] [--parallel N] [--slot-ctx S] [--queue-depth Q] [--plan]
//!                    [-ub N | --ubatch-size N]
//!                    [--chat-template-file PATH] [--mmproj FILE | --no-mmproj]
//!                    [--image-min-tokens N] [--image-max-tokens N]
//!
//! The server takes `-m`/`--hf` out before this seat parses (its module
//! doc); `--ctx-size` and `-c` are `--ctx` under llama-server's spellings.
//! The context is the user's ask and the plan fills the card around it
//! (`generate::choose_ctx`, every seat's): a set `--ctx` is the total the slots
//! split, planned once at its share, and a share the card cannot hold stops
//! the load by name with the largest it would hold. Unset, a slot is planned
//! at the file's trained context and, only where the card cannot hold the
//! load's resident bytes plus that context's sequence state, at the largest
//! context under it that the card holds, a multiple of 1024 never below 4096.
//! Each candidate is the load
//! itself: the whole model on the card where the whole-fit verdict takes it,
//! else the placed plan, under `--place` or the plan a run without it falls
//! to, its routed experts on whatever the card has left. One `ctx` line on
//! stderr names the rule (`set`, `target` or `fitted`), the slot context,
//! the slots, their total and the model's limit. The model is opened as
//! `generate_qwen3moe` opens it — qwen3moe through `Qwen3moeModel::open` at
//! its levers' options, qwen35moe through `Qwen35moeModel::open` with the
//! file's head shape selecting decode flash and the levers' ubatch — in graph mode with the
//! step and every pass captured, so a request's ids are the CLI's: the
//! server feeds the prompt less its last id as one prompt call — both bodies
//! by the seat's schedule (the ubatch walk from nine rows on — qwen3moe's
//! GEMM, qwen35moe's wide, no unit under the gemv arm's cut — passes below)
//! — then steps the last, which `generate_qwen3moe --last-step` does too.
//! Its first shard gives the vocabulary, `tokenizer.chat_template` the chat
//! template (`--chat-template-file` replaces it) and `general.name` the
//! alias. One slot unless `--parallel N`
//! names more; `--plan` prints the
//! records the decisions before the load make (the `ctx` line, the whole-fit
//! verdict or the `plan` record, the `parallel` line) and exits 0 before the
//! model is opened on a card; sampling is the sampler
//! crate's chain with no repetition penalty, `temperature <= 0` the
//! engine's argmax.
//!
//! A file with no routed expert (a dense trunk, Nanbeige4.3, `qwen35`) has only the whole load, so
//! the prompt ubatch its arena sizes is part of that load's verdict (`q3place::serve_load`):
//! `-ub N` (`--ubatch-size`, llama-server's spelling), else `BLOOMERY_QWEN3_UBATCH`, is taken
//! exactly or refused by name with the verdict's terms; unset, the load takes the largest of
//! 4096, 2048, 1024 and 512 whose whole need fits at the asked context and slots, and one `ubatch`
//! line on stderr (`ubatch_pick`) names the sizes tried, their needs and the pick, beside the
//! `whole fit` line of the size taken. A routed file takes the levers' ubatch and refuses the flag
//! by name.
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
//! (`whole_does_not_fit`). Each `--ctx` choice prints one line more: its
//! probes and the census readings it took. The seat takes the common rule
//! every serving seat takes (`generate::Place::choose_untiered`: the body
//! serves no tier card): a `place unset` record names the word `a`
//! and why — `one card`, or on two `the body serves no tier card` — before
//! the first choice.
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
//! request's context, `bind::slots_of`; one line on stderr says so) and one
//! slot by default; a count the flags set is never lowered.
//! Under more slots than one the context
//! is split across them
//! as llama-server splits it with `-np N` and no `-kvu`: the `--ctx` the
//! flags named (or, when unset, the chosen slot context times N) is the
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
//! rows when the body says the load can ([`Body3::serves_one_pass`]: a
//! qwen3moe file's `Body` and a qwen35moe file's `Body35`, whole-card and
//! placed with no expert tier, `GpuModel::step_slots`), else a select and a
//! step a row (the seat's fallback loop) — beside an expert tier card the
//! placed body runs no pass of several slots (its page is one column), so a
//! tiered load's rounds run a slot at a time.
//! Under `BLOOMERY_STEP_STATS=1` each round of several slots
//! prints a `slots round` record naming its command, rows, passes and the
//! slots the seat serves. `--parallel 1` is exactly the one-sequence server.
//! A `parallel` line on stderr before the load names the rule (`slots`), the
//! slots, a slot's context, the total, what set the count (`from`: the
//! `--parallel` flag, a set `--ctx`, the default) and the shape of a round of
//! several slots (`pass`: `one`, or `turns` under the lookup draft on a
//! body that keeps a pass's rows whole, so cannot verify several slots'
//! windows together, `bind::verifies_slots`, or beside an expert tier card,
//! whose page is one column). `--queue-depth Q` bounds
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
//! `BLOOMERY_DRAFT` unset on a qwen35moe or qwen35 file is the in-file
//! next-token layer when `draft38_unset` says the MTP draft and the file
//! carries the layer, else the plain path with the reason on `load draft=off`.
//! `off` is the plain path. `mtp` is that layer, refused by name where the
//! load cannot draft, and served undrafted with the reason named where the
//! file has no layer. A qwen3moe file keeps the plain path: `mtp` is refused
//! by name (no such layer, no verify pass).
//! `dspark` is the draft a qwen35moe file serves (`$BLOOMERY_DSPARK_MODEL`),
//! held by the seat as the serve seats' one owner of a served draft
//! (`serve_seats::served`): one slot while `--parallel` names none, several
//! refused by name, every greedy token after a request's first out of a pass.
//! The draft needs the whole-card verifying load, so a placed load (routed
//! experts on the host tier) is refused by name once the load is decided. A
//! whole-card load serves the draft: a full-vocabulary HF draft (MiniCPM-V
//! 4.7) borrows the target's Q6_K `output.weight` in place and its
//! `token_embd.weight` rows, and a reduced one (RedHat's Qwen3.6 speculator)
//! carries its own head and table and borrows nothing of the target's. The
//! draft's ids, mask and taps are held to the target before any weight is
//! read, and its bytes against the card's free room after the target's load.
//! The draft fits a target by its facts, not by the file's name, so a
//! fine-tune that carries the same shapes takes it too. On a qwen3moe file
//! the word is refused by name before any plan.
//! `lookup` is the n-gram draft a qwen35moe or qwen35 file serves on the same
//! whole-card verifying load, one context a slot, and a qwen3moe file refuses
//! it by name (its body runs no pass of several rows).
//!
//! A model path that is a directory is a dense Qwen3.5-family checkpoint folder (ModelOpt's
//! Qwen3.8-27B NVFP4), read as stored (`qwen3_folder::open`, no prepare step): its located header
//! plans as a file's first shard does, its tokenizer and chat template are the folder's, and the
//! engine loads every card tensor through the folder's part source, whole on the card. The folder's
//! `mtp.*` tensors are not read: unset or `off` serves the plain path, and `BLOOMERY_DRAFT=mtp` and
//! `BLOOMERY_MTP_WIDTH` are refused by name. A folder of another architecture, and a build without
//! the `checkpoint` feature, are refused by name.
//!
//! The seat sits behind the `deepseek41` feature, the server surface's
//! scope (`bind`, the `serve` and `sampler` crates); it runs no V4.1 code;
//! a placed load acts on the plan's and the host tier's levers
//! (`BLOOMERY_QWEN3_UBATCH` is read where the load sizes its arena).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

#[path = "../qwen3moe_place.rs"]
mod q3place;
use app::arch::qwen3moe::{GEMM_FROM, Prompt35};
use app::dspark::CardDspark;
use app::mtp::{MtpBody, MtpDraft};
use q3place::Seqs;

use bloomery_gpu::arch::qwen3moe::ubatch::{UBATCH_ENV, ubatch_size};
use bloomery_gpu::arch::qwen3moe::{Body, Body35, KvQ8, Qwen35moeModel};
use bloomery_gpu::dspark::DsparkBody;
use bloomery_gpu::folder::Model;
use bloomery_gpu::model::{GpuModel, MAX_PASS_ROWS, SlotRows, StepMode};
use bloomery_gpu::route_softmax::MAX_TOKENS;
use bloomery_gpu::weights::{DevWeight, head_tensor};
use bloomery_gpu::{Gpu, Qwen3moeModel};
use bloomery_gpu_gates::beside::Beside;
use bloomery_gpu_gates::bind::{
    self, ChatSurface, Listen, Pass, Seat, SeatEngine, SlotFlags, SlotStep, TemplateFile, Vocab,
    arm_screen, bind_server, call_token, note_line, parallel_line, placement_of, slots_given,
};
use bloomery_gpu_gates::generate::Place;
use bloomery_gpu_gates::qwen_seat::Family;
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::side_seat;
use bloomery_gpu_gates::vision_seat::{self, Opened, PleImage, TextFacts};
use bloomery_gpu_gates::{GateError, gpu_census, ref_model_path};
use bloomery_levers::{Draft38At, Draft38Off, Levers, draft38_unset};
use gguf::Split;
use gguf::quant::GgmlType;
use model::arch::models::{Mixer, RopeMode};
use model::arch::qwen35moe::hparams::Hparams;
use model::arch::qwen35moe::mtp::{InFile, in_file};
use model::placement::PlanLevers;
use model::placement::ctx_ask::{Chosen, DEFAULT_SLOTS};
use runtime::Target as _;
use runtime::prompt::CallEnd;
use runtime::seqstate::Why;
use runtime::width::Mode as WidthMode;
#[cfg(feature = "vision")]
use serve::media::{MediaFeed, SharedMediaModel};
use serve::{CacheNote, DraftProps, Drafted, EngineProps, Saved, ServeError};
use tokenizer::Tokenizer;

use super::drafted::{LOOKUP_ROWS, SlotDrafts};
use super::served::{self, ServedDraft, Word};
use super::{folder, qwen3_folder};
use crate::draft::Draft;
use crate::dspark_hf;
use crate::quiet::Quiet;

const NAME: &str = "bloomery-serve-qwen3";

const USAGE: &str = "usage: bloomery-serve --model qwen3 [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C] [--place W] [--cache-type-k f16|q8_0] \
                     [--parallel N] [--slot-ctx S] [--queue-depth Q] [--plan] \
                     [-ub N | --ubatch-size N] [--chat-template-file PATH] \
                     [--mmproj FILE | --no-mmproj] \
                     [--image-min-tokens N] \
                     [--image-max-tokens N] [--api-key KEY] [--api-key-file FNAME]";

/// What shapes the load beside its context and slots ([`q3place::ServeOpts`]),
/// `--mmproj`'s part read by [`vision_seat::images_of`], `embedding` the side model's reserve on
/// the stage card.
fn serve_opts<'a>(
    (place, levers): (Option<Place>, &'a PlanLevers),
    (kv, ubatch): (KvQ8, Option<usize>),
    (vision, embedding): (Option<&Opened>, Option<u64>),
) -> q3place::ServeOpts<'a> {
    let (media, image) = vision_seat::images_of(vision);
    q3place::ServeOpts {
        place,
        levers,
        kv,
        media,
        beside: Beside { image, embedding },
        ubatch,
    }
}

/// What a plan's line says of the reserves beside the model: each named, `""` when none sits on
/// the plan, so a load with nothing beside it prints its lines as it did.
fn beside_words(held: Beside) -> String {
    held.named()
        .map(|(name, bytes)| format!(", the {name} reserve of {bytes} B counted"))
        .collect()
}

/// A qwen35moe file's `parallel` resident sequences: each takes the
/// checkpoints the seat's kept prefixes come back to ([`Body3::open`] arms
/// them), so each pins its own on the host, priced by the body's own
/// derivation at the ubatch the load opens at: the size its verdict settled
/// ([`Q3Load::ubatch`]), else the levers' ([`q3place::seqs_qwen35`]).
fn seqs35(split: &Split, load: &Q3Load, parallel: usize) -> Result<Seqs, GateError> {
    let ubatch = match load.ubatch() {
        Some(u) => u,
        None => ubatch_size()?,
    };
    q3place::seqs_qwen35(split, ubatch, parallel)
}

/// Why the seat takes no sequence state: it runs no prompt cache.
const NO_CACHE: &str = "the qwen3 seat runs no prompt cache";

/// The rows the DSpark draft's verify runs on this seat: the token at the
/// target's position and its proposal.
const DSPARK_ROWS: usize = 8;

pub(crate) struct Args {
    host: String,
    port: u16,
    ctx: Option<usize>,
    place: Option<Place>,
    /// `--cache-type-k`: the K/V planes' format (llama-server's spelling),
    /// over the `BLOOMERY_QWEN3_KV` lever's word when given.
    cache_type_k: Option<String>,
    /// `--parallel`: the slots the server serves, resident sequences on
    /// either file's load; `None` takes one slot beside a set `--ctx`, else
    /// `ctx_ask::DEFAULT_SLOTS` (one).
    parallel: Option<usize>,
    queue_depth: Option<usize>,
    /// `-ub` / `--ubatch-size`: the prompt ubatch of a file with no routed expert, taken exactly or
    /// refused by name; unset, the load takes the largest size the card holds
    /// ([`q3place::serve_load`]).
    ubatch: Option<usize>,
    /// The image-input flags: `--mmproj` is the encoder file whose images a
    /// qwen35moe file's seat serves, none serves no image.
    vision: vision_seat::Flags,
    /// `--api-key`/`--api-key-file`: the keys every request is checked
    /// against ([`serve::flag::ApiKeys`]).
    api_keys: serve::flag::ApiKeys,
    /// `--plan`: the records before the load, then exit.
    plan_only: bool,
    /// `--chat-template-file`, replacing the file's own template.
    template_file: TemplateFile,
}

fn parse_args(args: &[String]) -> Result<Args, GateError> {
    let mut a = Args {
        host: "127.0.0.1".to_owned(),
        port: 8080,
        ctx: None,
        place: None,
        cache_type_k: None,
        // `None` resolves through `slots_given`: one slot beside a set
        // `--ctx` (the flag is one request's context), else
        // `ctx_ask::DEFAULT_SLOTS` (one). A count the flags set is never
        // lowered.
        parallel: None,
        queue_depth: None,
        ubatch: None,
        vision: vision_seat::Flags::default(),
        api_keys: serve::flag::ApiKeys::default(),
        plan_only: false,
        template_file: TemplateFile::default(),
    };
    let mut slots = SlotFlags::default();
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
            "--place" => a.place = Some(bind::place_word(v, None)?),
            "--cache-type-k" => a.cache_type_k = Some(v.to_owned()),
            TemplateFile::FLAG => a.template_file.set(v),
            "--cache-type-v" => {
                return Err(
                    "--cache-type-v does not exist: the qwen3 cache quantizes its K and V \
                     planes together, so --cache-type-k names the one format both take"
                        .into(),
                );
            }
            "--queue-depth" => match v.parse::<usize>() {
                Ok(q) => a.queue_depth = Some(q),
                _ => return Err(format!("{flag} takes a whole number, not {v:?}").into()),
            },
            f if serve::flag::UBATCH.contains(&f) => match v.parse::<usize>() {
                Ok(n) if n > 0 => a.ubatch = Some(n),
                _ => {
                    return Err(
                        format!("{flag} takes a whole number of at least 1, not {v:?}").into(),
                    );
                }
            },
            f if serve::flag::KEYS.contains(&f) => a.api_keys.add(f, v)?,
            "--park-ram" => {
                return Err(
                    "--park-ram names a budget of parked states; the qwen3 seat parks none — \
                     its resident slots keep their sequences on the card — so it reads no budget"
                        .into(),
                );
            }
            f if slots.take(f, v)? => {}
            other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
        }
    }
    (a.ctx, a.parallel) = slots.given(a.ctx)?;
    if let Some(total) = a.ctx {
        slot_ctx(total, a.parallel.unwrap_or(1))?;
    }
    Ok(a)
}

/// The model's open on the engine thread ([`Weights::hand_over`]).
type OpenModel = Box<dyn FnOnce() -> Result<Model, GateError> + Send>;

/// Where the seat's model is read from, decided once by the path: a directory is a dense
/// checkpoint folder ([`qwen3_folder`]), anything else a GGUF file. Each is opened on the main
/// thread for its header, which the plans are made from.
enum Weights {
    /// A checkpoint folder, boxed: its part source outweighs a header.
    Folder(Box<Stored>),
    /// A GGUF file's header; the engine thread opens the file again.
    File(Split),
}

/// A checkpoint folder's located header and part source, which the engine thread's whole-card
/// load reads every card tensor through.
struct Stored {
    header: Split,
    source: folder::Source,
}

impl Weights {
    /// The weights at `path`, and the folder's own tokenizer (`None` for a file, whose
    /// vocabulary is the file's).
    fn open(path: &Path) -> Result<(Weights, Option<Tokenizer>), GateError> {
        if path.is_dir() {
            let folder::Opened {
                header,
                tokenizer,
                source,
            } = qwen3_folder::open(path)?;
            let stored = Box::new(Stored { header, source });
            return Ok((Weights::Folder(stored), Some(tokenizer)));
        }
        let split = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        Ok((Weights::File(split), None))
    }

    /// The header the plans are made from: a split's first shard's, a folder's located header.
    fn header(&self) -> &Split {
        match self {
            Weights::Folder(stored) => &stored.header,
            Weights::File(header) => header,
        }
    }

    fn is_folder(&self) -> bool {
        matches!(self, Weights::Folder(_))
    }

    /// The model's open on the engine thread: a folder's header and part source go with it, a
    /// file is opened again there.
    fn hand_over(self, path: &Path) -> OpenModel {
        match self {
            Weights::Folder(stored) => {
                let Stored { header, source } = *stored;
                Box::new(move || folder::model(header, source))
            }
            Weights::File(header) => {
                drop(header);
                let file = path.to_path_buf();
                Box::new(move || -> Result<Model, GateError> {
                    Ok(Split::open(&file)
                        .map_err(|e| format!("open {}: {e}", file.display()))?
                        .into())
                })
            }
        }
    }
}

/// One of the two bodies the seat serves, opened as `generate_qwen3moe`
/// opens it, its prompt schedule and its keep rule. Both hold resident
/// sequences ([`Slots`]): the session adds and selects them alike.
trait Body3: SlotRows<Seq: 'static> + app::Prompt + app::Keep + Sized + 'static {
    /// The body name in the default media refusal.
    const ARCH: &'static str;
    /// Whether an undrafted in-file MTP decision prints `load draft=off`.
    /// The qwen3moe body prints nothing of a layer it does not carry.
    const PRINTS_MTP_OFF: bool = false;
    /// Whether the body runs several slots' rows as one pass (`SlotRows`),
    /// through [`Body3::one_pass_round`]: `Body` (a qwen3moe file) and
    /// `Body35` (a qwen35moe file) both do, on a whole-card load and on a
    /// placed one with no expert tier ([`Body3::serves_one_pass`]).
    const ONE_PASS: bool = false;
    /// Whether a load that hangs an expert tier card holding an expert
    /// ([`Body3::tiered`]) still runs several slots' rows as one pass. The
    /// default is no tier: the placed body refuses a pass of several slots
    /// beside one (the tier's page is one column), so its rounds run a slot
    /// at a time. The seat reads this and no other answer.
    fn serves_one_pass(tiered: bool) -> bool {
        Self::ONE_PASS && !tiered
    }
    /// The model of `file`, a GGUF split or a checkpoint folder ([`Model`]: only a qwen35moe
    /// body's whole-card load reads a folder), with a `ctx`-row cache by `load`, decided before
    /// the open at the total its `slots` sequences of that shape split
    /// ([`decide`]), the `plan` record on stderr first when a plan loads it, then the encoder's
    /// `vision_seat` record; `planes` is the cache planes' format and the encoder `--mmproj`
    /// seated, which gives the load an image input ([`open35`]'s `media`) and which only a
    /// qwen35moe body takes.
    fn open(
        file: Model,
        ctx: usize,
        slots: usize,
        load: Q3Load,
        levers: &Levers,
        planes: (KvQ8, Option<&Opened>),
        verifying: bool,
    ) -> Result<GpuModel<Self>, GateError>;
    /// The DSpark draft beside this body's model at `path`, `dir` its HF directory, for contexts
    /// of up to `ctx` positions, behind the width chooser of `mode`. Refused by the default: only
    /// a body whose session verifies rows (a qwen35moe body's) takes one.
    fn open_dspark(
        _s: &mut app::Session<Self>,
        _path: &Path,
        _dir: &Path,
        _ctx: usize,
        _card: &'static str,
        _mode: WidthMode,
    ) -> Result<Option<ServedDraft<Self>>, GateError> {
        Ok(None)
    }
    /// The lookup draft beside this body's model, one context a resident slot
    /// of `s`, each behind the width chooser of `mode`, verified by the
    /// session's pass of several rows. Refused by the default: only a body
    /// whose session verifies rows (a qwen35moe body's) takes one.
    fn open_lookup(
        _s: &mut app::Session<Self>,
        _slots: usize,
        _mode: WidthMode,
    ) -> Result<Option<ServedDraft<Self>>, GateError> {
        Ok(None)
    }
    /// The load runs its routed experts on the host tier too.
    fn placed(m: &GpuModel<Self>) -> Result<bool, GateError>;
    /// The load hangs an expert tier card that holds an expert (the placed
    /// side's `tier_any`): a load that does runs its rounds of several
    /// slots in turns ([`Body3::serves_one_pass`]). The pre-open twin, read
    /// off the plan, is [`Q3Load::tiered`].
    fn tiered(m: &GpuModel<Self>) -> Result<bool, GateError>;
    /// Every pass the prompt call replays, captured after the step.
    fn capture_passes(m: &mut GpuModel<Self>) -> Result<(), GateError>;
    /// The prompt call's ubatch size.
    fn ubatch(m: &GpuModel<Self>) -> Result<usize, GateError>;
    /// `ids` from where the model stands by the body's schedule; the argmax
    /// after the last.
    fn prompt(m: &mut GpuModel<Self>, ids: &[u32]) -> Result<u32, GateError>;
    /// Whether the body's prompt calls take checkpoints at their marks, set
    /// to `on`; what it was. The default holds none to take: a body whose
    /// caches are per-position keeps every position as it stands.
    fn set_marks(_m: &mut GpuModel<Self>, _on: bool) -> Result<bool, GateError> {
        Ok(false)
    }
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
    /// from [`Body3::serves_one_pass`]).
    fn one_pass_round(
        s: &mut app::Session<Self>,
        rows: &mut [SlotStep],
    ) -> Result<usize, GateError> {
        let _ = (s, rows);
        Err("a round of several slots as one pass: this body runs its slots in turns".into())
    }
    /// One prompt call of `ids` with the `encoded` rows of the call's images spliced in, each
    /// span carrying its image's token grid ([`vision_seat::SpanRows::image_spans`]); the greedy
    /// token after the last id. Refused by the default: only a qwen35moe file's body takes
    /// images.
    #[cfg(feature = "vision")]
    fn prompt_media(
        s: &mut app::Session<Self>,
        ids: &[u32],
        encoded: vision_seat::Encoded,
    ) -> Result<u32, GateError> {
        let _ = (s, ids);
        Err(format!(
            "a prompt with {} images: a {} file takes no image input",
            encoded.rows().ats.len(),
            Self::ARCH
        )
        .into())
    }

    /// One prompt call of `ids` whose images are tile groups: each entry is the span's first
    /// position, the group's rows (one bf16 row of the text width a position) and the canvas that
    /// places them ([`vision_seat::Resident::group_rows`]). Refused by the default: only a
    /// qwen35moe file's body takes a canvas group.
    #[cfg(feature = "vision")]
    fn prompt_groups(
        s: &mut app::Session<Self>,
        ids: &[u32],
        groups: &[(usize, Vec<u16>, vision::mrope::Canvas)],
    ) -> Result<u32, GateError> {
        let _ = (s, ids);
        Err(format!(
            "a prompt with {} image groups: a {} file takes no image group",
            groups.len(),
            Self::ARCH
        )
        .into())
    }
}

impl Body3 for Body {
    const ARCH: &'static str = "qwen3moe";
    const ONE_PASS: bool = true;

    /// As `load` decided before the open, making no fit call of its own: the
    /// whole model on its card, or the placed plan made at the total, its
    /// load holding a sequence of `ctx` rows for each of `slots`
    /// (`q3place::open_qwen3_slots`).
    fn open(
        file: Model,
        ctx: usize,
        slots: usize,
        load: Q3Load,
        levers: &Levers,
        (kv, vision): (KvQ8, Option<&Opened>),
        verifying: bool,
    ) -> Result<Qwen3moeModel, GateError> {
        let Model::Split(file) = file else {
            return Err(
                "a qwen3moe load reads a GGUF file: a checkpoint folder is a dense qwen35 \
                        model's"
                    .into(),
            );
        };
        if verifying {
            return Err("a qwen3moe load verifies no rows: a draft needs a qwen35moe file".into());
        }
        // `vision_of` refuses `--mmproj` on this file before anything is planned.
        if vision.is_some() {
            return Err("a qwen3moe load takes no image input".into());
        }
        let opts = Qwen3moeModel::lever_opts(ctx, kv)?;
        match load {
            Q3Load::Whole(on) => Ok(Qwen3moeModel::open(
                q3place::open_unplanned_card(on.card)?,
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
                for line in q.records(&plan, why)? {
                    line.eprint();
                }
                match whole {
                    Some(card) => Ok(Qwen3moeModel::open(
                        Gpu::open_card_after_hook(card.name, card.device)?,
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

    fn tiered(m: &Qwen3moeModel) -> Result<bool, GateError> {
        Ok(m.body(NAME)?.placed().is_some_and(|p| p.tier_any()))
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
    const PRINTS_MTP_OFF: bool = true;
    const ONE_PASS: bool = true;

    /// As `load` decided before the open, making no fit call of its own: the
    /// whole model on its card, or the placed plan made at the total, its
    /// machine carrying the sequences' rows, its load holding a sequence of
    /// `ctx` rows for each of `slots` (`q3place::open_qwen35_slots`).
    fn open(
        file: Model,
        ctx: usize,
        slots: usize,
        load: Q3Load,
        levers: &Levers,
        (kv, vision): (KvQ8, Option<&Opened>),
        verifying: bool,
    ) -> Result<Qwen35moeModel, GateError> {
        let o = load.open35(q3place::serve_open35(
            file.header(),
            ctx,
            kv,
            vision.is_some(),
        )?);
        let mut m = match load {
            Q3Load::Whole(on) => {
                vision_seat::print_seat(vision);
                let gpu = q3place::open_unplanned_card(on.card)?;
                if verifying {
                    Qwen35moeModel::open_verifying(gpu, file, o)?
                } else {
                    Qwen35moeModel::open(gpu, file, o)?
                }
            }
            Q3Load::Placed {
                q,
                total,
                why,
                whole,
            } => {
                let plan = q.plan(total, &PlanLevers::from_levers(levers)?)?;
                for line in q.records(&plan, why)? {
                    line.eprint();
                }
                vision_seat::print_seat(vision);
                match whole {
                    Some(card) => {
                        let gpu = Gpu::open_card_after_hook(card.name, card.device)?;
                        if verifying {
                            Qwen35moeModel::open_verifying(gpu, file, o)?
                        } else {
                            Qwen35moeModel::open(gpu, file, o)?
                        }
                    }
                    None if verifying => {
                        return Err(
                            "BLOOMERY_DRAFT=dspark needs the whole-card load: the plan puts routed \
                             experts on the host tier, and the draft's verifying load is whole on \
                             the card; serve a load that holds the whole model there, or no draft"
                                .into(),
                        );
                    }
                    None => {
                        let Model::Split(file) = file else {
                            return Err("a checkpoint folder loads whole on the card: a placed \
                                        load reads a GGUF file"
                                .into());
                        };
                        q3place::open_qwen35_slots(file, &plan, slots, o, levers.host())?
                    }
                }
            }
        };
        // The seat keeps prefixes, so its prompt calls take the checkpoints
        // of their marks.
        let (_, _, body) = m.body_parts(NAME)?;
        body.set_checkpoints(true);
        Ok(m)
    }

    /// The DSpark draft beside the whole-card verifying target: the target's Q6_K `output.weight`
    /// borrowed in place and its `token_embd.weight` rows (a full-vocabulary draft), or the draft's
    /// own head and table (a reduced one, which borrows nothing and so asks the target for no head
    /// at all). The draft's bytes are checked against the card's free room
    /// after the target's load, its own `Gpu` on the target's card and context.
    fn open_dspark(
        s: &mut app::Session<Body35>,
        path: &Path,
        dir: &Path,
        ctx: usize,
        card: &'static str,
        mode: WidthMode,
    ) -> Result<Option<ServedDraft<Body35>>, GateError> {
        let model = s.model();
        let device = model.gpu().device_id()?;
        let target_ctx = Arc::clone(model.gpu().context());
        let draft_gpu = Gpu::open_card_after_hook(card, Some(device))?;
        if draft_gpu.context().cu_ctx() != target_ctx.cu_ctx() {
            return Err(format!(
                "BLOOMERY_DRAFT=dspark: the draft's card {card} opened another CUDA context than \
                 the target's; the borrowed head runs only on the target's own device"
            )
            .into());
        }
        let t = Instant::now();
        let (free0, total0) = draft_gpu.mem_info()?;
        let need = dspark_hf::card_bytes_for_target(dir, ctx)?;
        if need > free0 as u64 {
            return Err(format!(
                "BLOOMERY_DRAFT=dspark: the draft needs {need} B on {card}, and {free0} B are free \
                 after the target's load; serve a shorter context, or no draft"
            )
            .into());
        }
        let n_vocab = model.n_vocab()?;
        let n_layers = model.layers().len();
        let target_split =
            Arc::new(Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?);
        // The target's hidden width, from its own description: the draft's tap rows are held to
        // it, whatever the target's head is quantized as.
        let hp = Hparams::read(&target_split)?;
        let target = dspark_hf::Target {
            n_vocab,
            n_layers,
            hidden: hp.n_embd,
        };
        // The draft's own shape first: a draft over the target's own ids borrows the target's
        // head in place, a reduced one carries its own head and reads the target's not at all.
        let borrows_head = matches!(
            dspark_hf::shape(dir, &dspark_hf::HfFile::open(dir)?)?.vocab,
            runtime::dspark::DraftVocab::Full { .. }
        );
        let borrowed = if borrows_head {
            let head_name = head_tensor(&target_split);
            let Some(DevWeight::KQuant {
                ty: GgmlType::Q6_K,
                w,
                ..
            }) = model.weights().get(head_name)
            else {
                return Err(format!(
                    "BLOOMERY_DRAFT=dspark: the target's {head_name} is not a resident Q6_K \
                     projection; the draft borrows that matrix in place"
                )
                .into());
            };
            Some(dspark_hf::Borrowed {
                split: target_split,
                head: w,
                ctx: &target_ctx,
            })
        } else {
            None
        };
        let dspark_hf::Parts {
            shape,
            layers,
            fc,
            head,
            embed,
        } = dspark_hf::open_for_target(&draft_gpu, dir, ctx, &target, borrowed.as_ref())?;
        let body = DsparkBody::new(&draft_gpu, &shape, layers, fc, head, embed)?;
        let (free1, _) = draft_gpu.mem_info()?;
        let taken = free0.saturating_sub(free1);
        let own = body.own_held();
        let held = body.layers().held(&own);
        let width = shape.width(<app::Session<Body35> as runtime::Verify>::MAX_ROWS);
        let taps = shape.taps.clone();
        let feature_width = shape.hidden.saturating_mul(taps.len());
        let working = held.working;
        let draft =
            CardDspark::new::<Body35>(body, draft_gpu, target_ctx, card, shape, held, taken)?;
        Record::new(&record::LOAD_DRAFT)
            .w("draft", "dspark")
            .w("card", draft.card())
            .u("width", width)
            .list("target_layers", &taps)
            .u("feature_width", feature_width)
            .u("resident", draft.resident_bytes())
            .u("working", working)
            .u("taken", draft.taken_bytes())
            .u("draft_card_free", free1)
            .u("draft_card_total", total0)
            .f("load_s", t.elapsed().as_secs_f64())
            .eprint();
        // The draft's load left its context current; the session's verify capture runs on the
        // target's.
        s.model().gpu().context().bind_to_thread()?;
        let mut log = Quiet;
        Ok(Some(ServedDraft::dspark::<_, DSPARK_ROWS>(
            s,
            draft,
            mode,
            &mut log,
            Some(dir.to_path_buf()),
        )?))
    }

    /// The lookup draft beside the whole-card verifying target: one n-gram
    /// context a resident slot, each behind the width chooser of `mode`,
    /// verified by the session's two-row pass. The body keeps no token
    /// history, so a context that falls behind the target is not rebuilt from
    /// one.
    fn open_lookup(
        s: &mut app::Session<Body35>,
        slots: usize,
        mode: WidthMode,
    ) -> Result<Option<ServedDraft<Body35>>, GateError> {
        let mut log = Quiet;
        Ok(Some(ServedDraft::lookup(s, slots, mode, &mut log, None)?))
    }

    fn placed(m: &Qwen35moeModel) -> Result<bool, GateError> {
        Ok(m.body(NAME)?.placed().is_some())
    }

    fn tiered(m: &Qwen35moeModel) -> Result<bool, GateError> {
        Ok(m.body(NAME)?.placed().is_some_and(|p| p.tier_any()))
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

    /// [`Body35::set_checkpoints`]: the recurrent stores' checkpoints.
    fn set_marks(m: &mut Qwen35moeModel, on: bool) -> Result<bool, GateError> {
        let (_, _, body) = m.body_parts(NAME)?;
        let was = body.takes_checkpoints();
        body.set_checkpoints(on);
        Ok(was)
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

    /// The session's media call ([`vision_seat::Encoded::prompt`]): each image's span of the call
    /// takes its rows in place of its positions' embeddings and shifts the sequence's rotation
    /// table by its grid, an `(nx, ny)` of positions in raster order. The grid is the span's own
    /// ([`vision_seat::SpanAt::grid`]); the rows are the tower's.
    #[cfg(feature = "vision")]
    fn prompt_media(
        s: &mut app::Session<Body35>,
        ids: &[u32],
        encoded: vision_seat::Encoded,
    ) -> Result<u32, GateError> {
        encoded.prompt(s, ids, |rows| rows.image_spans())
    }

    /// The session's media call of canvas groups ([`Body3::prompt_groups`]): each span's rows
    /// stand in for its positions, and the sequence's rotation table takes the canvas
    /// ([`bloomery_gpu::media::ImageShape::Canvas`]).
    #[cfg(feature = "vision")]
    fn prompt_groups(
        s: &mut app::Session<Body35>,
        ids: &[u32],
        groups: &[(usize, Vec<u16>, vision::mrope::Canvas)],
    ) -> Result<u32, GateError> {
        use bloomery_gpu::media::{ImageShape, ImageSpan, MediaSpan};
        let spans: Vec<ImageSpan<'_>> = groups
            .iter()
            .map(|(at, rows, canvas)| ImageSpan {
                span: MediaSpan {
                    at: *at..*at + canvas.rows(),
                    rows,
                },
                shape: ImageShape::Canvas(*canvas),
            })
            .collect();
        Ok(s.prompt_media(ids, &spans, runtime::Want::Argmax)?.argmax())
    }
}

/// The session on the engine thread, the positions its cache was sized
/// for, and what the seat decided once at its open: whether it runs every
/// round of several slots as one pass, and whether it counts those rounds.
struct Q3<B: Body3> {
    /// The DSpark draft the seat serves ([`ServedDraft`], the serve seats'
    /// one owner of it): off on the plain path. Declared before the session so
    /// its borrowed target head is dropped before the target's model.
    draft: ServedDraft<B>,
    s: app::Session<B>,
    ctx: usize,
    /// Whether a round of several slots runs as one pass of the busy rows
    /// ([`Q3::step_slots`]): the shape [`pass_of`] decided at the open —
    /// [`Body3::serves_one_pass`] on the load (a whole-card load, and a
    /// placed load of `Body` or `Body35` with no expert tier) and, under the
    /// lookup draft, a body that verifies several slots' windows together,
    /// which neither does. Any other load keeps the fallback loop, and a
    /// load of one slot never runs a round of several.
    one_pass: bool,
    /// [`Seat::step_stats`]: the `BLOOMERY_STEP_STATS` the binary parsed.
    stats: bool,
    /// The resident image encoder under `--mmproj` ([`vision_seat::open_resident`]).
    #[cfg(feature = "vision")]
    vision: Option<vision_seat::Resident>,
}

impl<B: Body3> Q3<B> {
    /// The model `at` names, the GGUF file at `path` or the checkpoint folder the main thread
    /// opened ([`Weights::hand_over`]), in graph mode by `load`
    /// ([`Body3::open`]), its step captured and, unless placed, its passes,
    /// `slots` resident sequences parked after them (each capturing on its
    /// first use), the session over it; the `load` record on stderr. The
    /// opened body is held to `pass`, the shape the `parallel` line named.
    #[allow(
        clippy::too_many_arguments,
        reason = "the seat's one engine-thread call takes its plan, cards, planes and optional draft"
    )]
    fn open(
        (path, at): (&Path, OpenModel),
        ctx: usize,
        slots: usize,
        pass: Pass,
        load: Q3Load,
        levers: &Levers,
        (kv, vision): (KvQ8, Option<Opened>),
        draft: Draft,
        dspark: Option<PathBuf>,
        width: WidthMode,
        draft_off: Option<String>,
    ) -> Result<Q3<B>, GateError> {
        let t = Instant::now();
        let file = at()?;
        let arch = file
            .header()
            .architecture()
            .unwrap_or("<missing>")
            .to_owned();
        let opened = t.elapsed();
        let card = match &load {
            Q3Load::Whole(on) => on.card.name,
            Q3Load::Placed {
                whole: Some(card), ..
            } => card.name,
            Q3Load::Placed { q, .. } => q.name(),
        };
        let mut m = B::open(
            file,
            ctx,
            slots,
            load,
            levers,
            (kv, vision.as_ref()),
            draft != Draft::Off,
        )?;
        m.set_mode(StepMode::Graph);
        let nodes = m.capture_step()?;
        let placed = B::placed(&m)?;
        let tiered = B::tiered(&m)?;
        let ran = pass_of::<B>(slots, tiered, draft);
        if ran != pass {
            return Err(format!(
                "the `parallel` line named pass={} and the load runs pass={}: the plan's tier \
                 experts and the opened body's placement disagree",
                pass.word(),
                ran.word()
            )
            .into());
        }
        // The tier's rounds, named as the qwen38 seat names them: the placed
        // body refuses a pass of several slots beside an expert tier card.
        if tiered && slots > 1 {
            eprintln!(
                "{NAME}: --place {card}: the rounds of several slots run a slot at a time: the \
                 body runs no pass of several slots beside an expert tier card"
            );
        }
        if !placed {
            B::capture_passes(&mut m)?;
        }
        let at = u32::try_from(ctx).map_err(|_| format!("--ctx {ctx} passes u32"))?;
        let mut s = app::Session::from_model(m, at);
        if slots > 1 {
            s.add_slots(slots)?;
        }
        let draft = match draft {
            Draft::Off => ServedDraft::off(),
            Draft::Dspark => {
                let Some(dir) = dspark else {
                    return Err("BLOOMERY_DRAFT=dspark: no DSpark draft directory".into());
                };
                B::open_dspark(&mut s, path, &dir, ctx, card, width)?.ok_or_else(|| {
                    format!("BLOOMERY_DRAFT=dspark: a {} body takes no draft", B::ARCH)
                })?
            }
            Draft::Lookup => B::open_lookup(&mut s, slots, width)?.ok_or_else(|| {
                format!(
                    "BLOOMERY_DRAFT=lookup: a {} body takes no lookup draft",
                    B::ARCH
                )
            })?,
        };
        let total = t.elapsed();
        Record::new(&record::LOAD_QWEN3)
            .w("arch", arch)
            .u("resident_bytes", s.model().resident_bytes())
            .w("cache", kv.name())
            .u("ctx", ctx)
            .u("slots", slots)
            .u("slot_ctx", ctx)
            .u("layers", s.model().layers().len())
            .u("ubatch", B::ubatch(s.model())?)
            .u("graph_nodes", nodes)
            .f("load_s", total.as_secs_f64())
            .eprint();
        if B::PRINTS_MTP_OFF
            && let Some(why) = &draft_off
        {
            Record::new(&record::LOAD_DRAFT_OFF_Q3)
                .w("why", why)
                .eprint();
        }
        record::eprint_load_phases(s.model(), total, opened);
        // The tower after the model's captures, as the V4.1 seat opens its encoder.
        #[cfg(feature = "vision")]
        let vision = vision_seat::open_resident(s.model().gpu().context(), vision)?;
        #[cfg(not(feature = "vision"))]
        let _ = vision;
        Ok(Q3 {
            draft,
            s,
            ctx,
            one_pass: ran.together(slots),
            stats: levers.step_stats(),
            #[cfg(feature = "vision")]
            vision,
        })
    }
}

impl<B: Body3> Seat for Q3<B> {
    fn pos(&self) -> usize {
        self.s.model().pos() as usize
    }

    fn ctx_max(&self) -> usize {
        self.ctx
    }

    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        self.prompt_call(ids, CallEnd::Argmax)
    }

    /// The body's prompt call ending as `end` asks
    /// ([`app::Prompt::prompt_call`]: a Qwen3.5/3.6 call runs no head when
    /// unread), through the draft as [`Q3::prefill`] runs it: the DSpark
    /// draft's own call runs its head whatever the end.
    fn prompt_call(&mut self, ids: &[u32], end: CallEnd) -> Result<u32, GateError> {
        self.draft.prefill(&mut self.s, ids, |s| {
            call_token(
                end,
                <B as app::Prompt>::prompt_call(s.model_mut(), ids, end)?,
            )
        })
    }

    /// The resident encoder's media description ([`vision_seat::Resident::media`]); `None`
    /// without `--mmproj`, a seat that takes no image.
    #[cfg(feature = "vision")]
    fn media_model(&self) -> Option<SharedMediaModel> {
        self.vision.as_ref().map(vision_seat::Resident::media)
    }

    /// Each feed's image encoded on the resident encoder and read back as bf16, then one prompt
    /// call with the images spliced in. A patch grid is one row a cell and nothing around it
    /// ([`vision_seat::merged_cells`]: the tokens carry the delimiters;
    /// [`vision_seat::Resident::encode`], [`Body3::prompt_media`]). A tile group is each tile's
    /// cells then its separator rows ([`vision_seat::Resident::group_rows`]), and the span's
    /// shape is that group's canvas ([`Body3::prompt_groups`]). A call is one shape or the other.
    /// The draft's media refusal comes before any encode ([`ServedDraft::prefill_media`]). The
    /// encode binds the model's card back before the prompt call.
    #[cfg(feature = "vision")]
    fn prefill_media(&mut self, ids: &[u32], feeds: &[MediaFeed]) -> Result<u32, GateError> {
        let Q3 {
            s, vision, draft, ..
        } = self;
        let Some(v) = vision.as_mut() else {
            return Err("no --mmproj: this seat serves no image".into());
        };
        draft.prefill_media(s, ids, |s| {
            let inputs: Vec<&vision::media::EncoderInput> =
                feeds.iter().map(|f| &f.prepared.input).collect();
            let tiles = vision_seat::tile_group_count(&inputs)?;
            if tiles != 0 {
                let ctx = Arc::clone(s.model().gpu().context());
                let mut groups = Vec::with_capacity(feeds.len());
                for f in feeds {
                    let vision::media::EncoderInput::Tiles(group) = &f.prepared.input else {
                        return Err("a tile group call holds a patch grid".into());
                    };
                    let (rows, canvas) = v.group_rows(&ctx, group)?;
                    if canvas.rows() != f.prepared.span_len {
                        return Err(format!(
                            "image at {}: the canvas holds {} rows, the prompt's span is {}",
                            f.at,
                            canvas.rows(),
                            f.prepared.span_len
                        )
                        .into());
                    }
                    groups.push((f.at, rows, canvas));
                }
                return B::prompt_groups(s, ids, &groups);
            }
            let encoded = v.encode(
                s.model().gpu().context(),
                feeds
                    .iter()
                    .map(|f| vision_seat::Image::of(f.at, &f.prepared)),
            )?;
            B::prompt_media(s, ids, encoded)
        })
    }

    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        let next = self.s.model_mut().step(&[last])?;
        self.draft.stepped(&mut self.s, last, next)?;
        Ok(next)
    }

    /// An answer's run and the step of its last id. Answer positions are
    /// decode positions, and decode positions take no checkpoints: the steps
    /// the run replaces take none, so its prompt call is fed with the body's
    /// marks off ([`Body3::set_marks`]) and the prompt calls keep theirs. A
    /// run below [`GEMM_FROM`] ids is one prompt call whose head's row is
    /// read: the call runs passes, bit for bit one step a row, so the row is
    /// the one the step after a prefill gives, and the step is a call the row
    /// does not need. A longer run, one id, or a run the DSpark draft takes
    /// (its prompt call feeds the draft's rows besides the target's) is a
    /// prefill and a step ([`bind::prefill_then_step`]): the wide walk's row
    /// is not the step's.
    fn prefill_row(&mut self, ids: &[u32], row: Option<&mut [f32]>) -> Result<u32, GateError> {
        let was = <B as Body3>::set_marks(self.s.model_mut(), false)?;
        let r = if ids.len() < 2 || ids.len() >= GEMM_FROM || self.draft.rows() > 1 {
            bind::prefill_then_step(self, ids, row)
        } else {
            <B as Body3>::prompt(self.s.model_mut(), ids).and_then(|next| {
                if let Some(row) = row {
                    self.logits_into(row)?;
                }
                Ok(next)
            })
        };
        <B as Body3>::set_marks(self.s.model_mut(), was)?;
        r
    }

    /// The draft's pass ([`ServedDraft::pass`]), else one plain step.
    fn pass(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, GateError> {
        match self.draft.pass(&mut self.s, last, out)? {
            Some(drafted) => Ok(drafted),
            None => {
                out.push(self.step(last)?);
                Ok(Drafted::default())
            }
        }
    }

    fn pass_rows(&self) -> usize {
        self.draft.rows()
    }

    fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError> {
        Ok(self.s.model().logits_into(row)?)
    }

    /// The resident sequences the seat made at its open
    /// (`app::Session::slots`): `--parallel`.
    fn slots(&self) -> usize {
        self.s.slots()
    }

    /// Whether the seat keeps its draft's state per slot: the lookup draft's
    /// context is a slot's own ([`ServedDraft::slot_drafts`]), the DSpark
    /// draft's is one sequence's.
    fn slot_drafts(&self) -> bool {
        self.draft.slot_drafts()
    }

    /// The session's slot (`app::Session::select_slot`): the model exchanges
    /// its live sequence with the slot's parked state — pointer moves. The
    /// target then stands on another sequence than the draft was fed.
    fn select(&mut self, slot: usize) -> Result<(), GateError> {
        let changed = slot != self.s.selected();
        self.s.select_slot(slot)?;
        if changed {
            self.draft.moved();
        }
        Ok(())
    }

    /// The lever the binary parsed ([`Q3::stats`]).
    fn step_stats(&self) -> bool {
        self.stats
    }

    /// One round of several slots ([`super::rounds::step_round`]): one pass
    /// of the busy rows while the open decided so ([`Q3::one_pass`], through
    /// [`Body3::one_pass_round`] — the rows' answers and lent logits rows
    /// from the pass's own per-row heads, slot 0 left selected), else the
    /// seat's fallback — a select and a step a row — the lookup draft's
    /// round and every round of a body with no one pass of its slots, and
    /// the round a load of one slot never runs.
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

    /// The model's reset ([`GpuModel::reset`]; no pass is open between
    /// requests, so the session's own reset adds nothing to it), the draft
    /// starting over with it.
    fn reset(&mut self) -> Result<(), GateError> {
        self.s.model_mut().reset()?;
        self.draft.reset(self.s.selected())
    }

    /// The draft's record printed ([`ServedDraft::close`]).
    fn close(&mut self) -> Result<(), GateError> {
        self.draft.close();
        Ok(())
    }

    /// The session's cut (`app::Keep`), which takes only a position
    /// [`Seat::keep`] granted; any other is refused by name.
    fn rollback(&mut self, pos: u32) -> Result<(), GateError> {
        self.draft.moved();
        Ok(self.s.cut(pos)?)
    }

    /// The body's rule: every held position, or for a body with checkpoints
    /// the nearest at or below `n`, with the rule's sentence when it keeps
    /// less; under the DSpark draft, also the window the draft's start over
    /// leaves the prompt call ([`ServedDraft::keep`]).
    fn keep(&self, n: usize) -> (usize, Option<String>) {
        self.draft.keep(n, self.s.pos(), |m| self.s.keep_query(m))
    }

    /// `engine.draft` and the draft's bytes, when one is served
    /// ([`ServedDraft::props`]).
    fn props(&self, p: EngineProps) -> EngineProps {
        self.draft.props(p)
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

/// The draft's store is not cut with the target: the layer attends 0..=pos,
/// and a cut does not leave the step arena holding the hidden row before the
/// prefix.
const CUT_OFF: &str = "the draft's store is not cut with the target: the layer attends 0..=pos, \
     and a cut does not leave the step arena holding the hidden row before the prefix";

/// A verify of the in-file draft: one proposal and the token before it.
const MTP_ROWS: usize = <Body35 as MtpBody>::VERIFY_ROWS;

/// The qwen35moe session and its in-file next-token draft ([`MtpDraft`]),
/// one resident sequence. A cut that moves the position turns the draft off:
/// the layer's store is not the checkpoint the body's cut restores.
struct Q35Mtp {
    drafted: SlotDrafts<Body35, MTP_ROWS>,
    s: app::Session<Body35>,
    ctx: usize,
    stats: bool,
    path: PathBuf,
    /// The draft's resident bytes, read once at the open.
    resident: u64,
}

impl Q35Mtp {
    /// The whole-card load with the file's next-token layer ([`GpuModel::open_mtp`]),
    /// the step captured, one slot, and the draft opened eager over [`Prompt35`].
    /// The `load` record, then `load draft=mtp`. A host-expert placed load and a
    /// media load are refused by name: they do not reach this open.
    #[allow(
        clippy::too_many_arguments,
        reason = "the seat's one engine-thread call takes its plan, cards and planes"
    )]
    fn open(
        path: &Path,
        ctx: usize,
        slots: usize,
        load: Q3Load,
        levers: &Levers,
        (kv, vision): (KvQ8, Option<Opened>),
        width: WidthMode,
    ) -> Result<Q35Mtp, GateError> {
        check_mtp_inputs(slots, vision.is_some())?;
        if matches!(load, Q3Load::Placed { whole: None, .. }) {
            return Err(format!("BLOOMERY_DRAFT=mtp: {}", placed_no_verify()).into());
        }
        let t = Instant::now();
        let file = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let arch = file.architecture().unwrap_or("<missing>").to_owned();
        let opened = t.elapsed();
        let o = load.open35(q3place::serve_open35(&file, ctx, kv, false)?);
        let mut m = match load {
            Q3Load::Whole(on) => {
                vision_seat::print_seat(None);
                let gpu = q3place::open_unplanned_card(on.card)?;
                Qwen35moeModel::open_mtp(gpu, file, o)?
            }
            Q3Load::Placed {
                q,
                total,
                why,
                whole: Some(card),
            } => {
                let plan = q.plan(total, &PlanLevers::from_levers(levers)?)?;
                for line in q.records(&plan, why)? {
                    line.eprint();
                }
                vision_seat::print_seat(None);
                let gpu = Gpu::open_card_after_hook(card.name, card.device)?;
                Qwen35moeModel::open_mtp(gpu, file, o)?
            }
            Q3Load::Placed { whole: None, .. } => {
                return Err(format!("BLOOMERY_DRAFT=mtp: {}", placed_no_verify()).into());
            }
        };
        {
            let (_, _, body) = m.body_parts(NAME)?;
            body.set_checkpoints(true);
        }
        m.set_mode(StepMode::Graph);
        let nodes = m.capture_step()?;
        let at = u32::try_from(ctx).map_err(|_| format!("--ctx {ctx} passes u32"))?;
        let mut s = app::Session::from_model(m, at);
        let mut log = Quiet;
        let drafted = SlotDrafts::<Body35, MTP_ROWS>::open(&mut s, 1, width, &mut log, |m| {
            MtpDraft::open(m, Prompt35, StepMode::Eager)
        })?;
        let (layer, resident, scratch) = {
            let d = s
                .model()
                .body(NAME)?
                .mtp()
                .ok_or_else(|| format!("{NAME}: open_mtp loaded no next-token layer"))?;
            (
                d.index() as u64,
                d.resident_bytes() as u64,
                d.scratch_bytes() as u64,
            )
        };
        let total = t.elapsed();
        Record::new(&record::LOAD_QWEN3)
            .w("arch", arch)
            .u("resident_bytes", s.model().resident_bytes())
            .w("cache", kv.name())
            .u("ctx", ctx)
            .u("slots", slots)
            .u("slot_ctx", ctx)
            .u("layers", s.model().layers().len())
            .u("ubatch", s.model().ubatch()?)
            .u("graph_nodes", nodes)
            .f("load_s", total.as_secs_f64())
            .eprint();
        Record::new(&record::LOAD_DRAFT_Q3)
            .u("layer", layer)
            .u("resident", resident)
            .u("scratch", scratch)
            .w("head", "full")
            .f("load_s", total.as_secs_f64())
            .eprint();
        record::eprint_load_phases(s.model(), total, opened);
        Ok(Q35Mtp {
            drafted,
            s,
            ctx,
            stats: levers.step_stats(),
            path: path.to_path_buf(),
            resident,
        })
    }
}

impl Seat for Q35Mtp {
    fn pos(&self) -> usize {
        self.s.pos() as usize
    }

    fn ctx_max(&self) -> usize {
        self.ctx
    }

    fn slots(&self) -> usize {
        self.s.slots()
    }

    fn slot_drafts(&self) -> bool {
        self.drafted.drafts()
    }

    fn step_stats(&self) -> bool {
        self.stats
    }

    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        self.prompt_call(ids, CallEnd::Argmax)
    }

    /// The draft's prompt call ([`SlotDrafts::prompt_call`]), the target's
    /// head run only when `end` reads it. A delta layer takes no checkpoint
    /// inside a unit, so the seat folds no last id
    /// ([`Seat::prefill_next`]'s default).
    fn prompt_call(&mut self, ids: &[u32], end: CallEnd) -> Result<u32, GateError> {
        let sel = self.s.selected();
        self.drafted.prompt_call(&mut self.s, sel, ids, end)
    }

    /// Answer positions take no checkpoints. The prompt calls keep theirs.
    fn prefill_row(&mut self, ids: &[u32], row: Option<&mut [f32]>) -> Result<u32, GateError> {
        let was = {
            let (_, _, body) = self.s.model_mut().body_parts(NAME)?;
            let was = body.takes_checkpoints();
            body.set_checkpoints(false);
            was
        };
        let r = bind::prefill_then_step(self, ids, row);
        let (_, _, body) = self.s.model_mut().body_parts(NAME)?;
        body.set_checkpoints(was);
        r
    }

    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        let sel = self.s.selected();
        self.drafted.step(&mut self.s, sel, last)
    }

    fn step_row(&mut self, last: u32, row: &mut [f32]) -> Result<u32, GateError> {
        let sel = self.s.selected();
        self.drafted.step_with_row(&mut self.s, sel, last, row)
    }

    fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError> {
        Ok(self.s.model().logits_into(row)?)
    }

    fn pass(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, GateError> {
        let sel = self.s.selected();
        self.drafted.pass(&mut self.s, sel, last, out)
    }

    fn pass_rows(&self) -> usize {
        self.drafted.pass_rows(self.s.selected())
    }

    fn pass_sampled(
        &mut self,
        last: u32,
        history: &mut Vec<u32>,
        sampler: &mut serve::Sampler,
        best: Option<usize>,
        out: &mut Vec<u32>,
    ) -> Result<Drafted, GateError> {
        let sel = self.s.selected();
        self.drafted
            .pass_sampled(&mut self.s, sel, last, history, sampler, best, out)
    }

    fn drafts_sampled(&self) -> bool {
        self.drafted.drafts()
    }

    fn reset(&mut self) -> Result<(), GateError> {
        let sel = self.s.selected();
        self.drafted.reset(&mut self.s, sel)
    }

    /// The draft's record printed ([`SlotDrafts::close`]).
    fn close(&mut self) -> Result<(), GateError> {
        self.drafted.close();
        Ok(())
    }

    /// The session's cut. A cut that moves the position turns the draft off
    /// ([`CUT_OFF`]): the body's cut does not leave the step arena holding
    /// the hidden row before the prefix, and the layer attends `0..=pos`.
    fn rollback(&mut self, pos: u32) -> Result<(), GateError> {
        let held = self.s.pos();
        self.s.cut(pos)?;
        if pos < held {
            self.drafted.turn_off(0, CUT_OFF)?;
        }
        Ok(())
    }

    /// The body's checkpoint rule ([`Session::kept`]).
    fn keep(&self, n: usize) -> (usize, Option<String>) {
        let asked = u32::try_from(n).unwrap_or(u32::MAX);
        let k = self.s.kept(asked);
        let why = match k.why {
            Why::Current | Why::Empty => None,
            _ => Some(k.to_string()),
        };
        (k.at as usize, why)
    }

    /// `/props`' `engine.draft`: the target file, whose next-token layer the
    /// draft is. A whole-card load reports no placement, so the `draft` class
    /// is set only where a device row exists.
    fn props(&self, mut p: EngineProps) -> EngineProps {
        if let Some(place) = p.placement.as_mut() {
            for d in &mut place.devices {
                if d.device.starts_with("GPU") {
                    *d.class_bytes.entry("draft".to_owned()).or_default() += self.resident;
                }
            }
        }
        p.draft = Some(DraftProps {
            model: self.path.file_name().map_or_else(
                || self.path.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            ),
            n_max: Some(<Body35 as MtpBody>::WIDTH as u64),
            kind: Some("mtp".to_owned()),
            path: Some(self.path.display().to_string()),
            device: None,
        });
        p
    }

    fn splits(&self, first: usize, end: usize, marks: &[usize]) -> Vec<usize> {
        <Body35 as Body3>::splits(first, end, marks)
    }

    fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
        Err(format!("a snapshot: {NO_CACHE}").into())
    }

    fn resume(&mut self, _state: &dyn Saved) -> Result<(), GateError> {
        Err(format!("a resume: {NO_CACHE}").into())
    }

    fn note(note: &CacheNote) {
        note_line(NAME, note);
    }
}

/// What the file loads as, decided before the open ([`q3place::serve_load`]).
use q3place::ServeLoad as Q3Load;

impl Q3Load {
    /// The `plan` record a placed load prints before its open
    /// ([`Body3::open`]); a whole load has none. The encoder's `vision_seat` record follows it
    /// (a whole load's stands alone).
    fn print_plan(&self, levers: &Levers, vision: Option<&Opened>) -> Result<(), GateError> {
        if let Q3Load::Placed { q, total, why, .. } = self {
            let plan = q.plan(*total, &PlanLevers::from_levers(levers)?)?;
            for line in q.records(&plan, *why)? {
                line.eprint();
            }
        }
        vision_seat::print_seat(vision);
        Ok(())
    }

    /// Whether the open is the placed body ([`Body3::placed`] reads the same
    /// fact off the opened body): a placed load, whether its plan holds host
    /// experts or none. A load that opens the whole model is not.
    fn placed_body(&self) -> bool {
        matches!(self, Q3Load::Placed { whole: None, .. })
    }

    /// Whether the load hangs an expert tier that holds an expert
    /// ([`Body3::tiered`] reads the same fact off the opened body): the
    /// placed plan's tier experts, none on a load that opens the whole
    /// model. The plan is the open's own ([`Body3::open`] plans at the same
    /// total under the same levers).
    fn tiered(&self, levers: &PlanLevers) -> Result<bool, GateError> {
        let Q3Load::Placed {
            q,
            total,
            whole: None,
            ..
        } = self
        else {
            return Ok(false);
        };
        Ok(q.plan(*total, levers)?
            .tier_n_l
            .iter()
            .flatten()
            .any(|&n| n > 0))
    }
}

/// The shape of a round of `slots` slots on body `B`, by what the body
/// runs ([`Pass::of_body`]): one pass of the busy rows when
/// [`Body3::serves_one_pass`] says the load can (a whole-card load, and a
/// placed load of `Body` or `Body35` whose plan hangs no expert tier holding
/// an expert) and, under a draft, the body verifies several slots' windows
/// of its rows together (`bind::verifies_slots`); else the fallback loop (a
/// select and a step a slot). The one owner of the rule, which the
/// `parallel` line names ahead of the load and the open holds the load to.
fn pass_of<B: Body3>(slots: usize, tiered: bool, draft: Draft) -> Pass {
    let window = match draft {
        Draft::Off => None,
        Draft::Lookup => Some(LOOKUP_ROWS),
        Draft::Dspark => Some(DSPARK_ROWS),
    };
    Pass::of_body::<B>(slots, B::serves_one_pass(tiered), window)
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
/// The line a `--ctx` choice (`word`, set or default) prints once it
/// answers: the probes it ran and the census readings it took
/// ([`q3place::serve_choice`]) — one reading serves every probe, so a choice
/// that read the census per probe shows here.
fn choice_line(word: &str, r: q3place::Reads) {
    eprintln!(
        "{NAME}: the {word} --ctx choice ran {} probes on {} census reading(s)",
        r.probes, r.censuses
    );
}

/// The context `slots` slots serve and the load made at it
/// ([`q3place::serve_choice_lanes`], at `lanes` delta lanes — the seat's,
/// [`verdict_lanes`]), its choice line printed.
fn ctx_and_load(
    split: &Split,
    (set, slots): (Option<usize>, usize),
    o: q3place::ServeOpts<'_>,
    lanes: usize,
) -> Result<Chosen<Q3Load>, GateError> {
    let (chosen, reads) = q3place::serve_choice_lanes(split, (set, slots), o, lanes)?;
    choice_line(if set.is_some() { "set" } else { "default" }, reads);
    Ok(chosen)
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
/// ([`vision_seat::open_seat`]); the file is `--mmproj`'s, else the `--hf` repo's projector
/// ([`vision_seat::Flags::encoder`]), and `None` without either. A qwen3moe file (Qwen3-30B-A3B)
/// has no image input, so `--mmproj` there is refused by name, and so is a build without the
/// `vision` feature, whose tower is not built. A `minicpmv4_7` file is read here
/// ([`open_minicpm`]): its image ids come from the text tokenizer, which the encoder file does
/// not carry. The tower is sized for [`vision_seat::MINICPM_TILE_TOKENS`] and seated.
fn vision_of(
    a: &Args,
    arch: Family,
    (path, split): (&Path, &Split),
    token: &dyn Fn(u32) -> Option<String>,
    one: &dyn Fn(&str) -> Option<u32>,
) -> Result<Option<Opened>, GateError> {
    if matches!(arch, Family::Qwen3moe) {
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
    // Images enter the Qwen3.5 body through its IMROPE rotation; a trunk that rotates otherwise
    // has no image input.
    let imrope = model::arch::spec(split)?.spec.layers.iter().any(
        |l| matches!(&l.mixer, Mixer::Gqa(g) if matches!(g.rope.mode, RopeMode::Imrope { .. })),
    );
    if !imrope {
        return match a.vision.wanted()? {
            Some(file) => Err(format!(
                "--mmproj {}: {} is a {} file whose attention does not rotate by IMROPE, which \
                 has no image input; this seat serves images on a qwen35moe file \
                 (Qwen3.6-35B-A3B)",
                file.display(),
                path.display(),
                split.architecture().unwrap_or("<missing>")
            )
            .into()),
            None => Ok(None),
        };
    }
    a.vision
        .check_before_model(bloomery_gpu_gates::model_file::hf_repo().is_some())?;
    // Every attention layer rotates by IMROPE over the file's sections; a PLE site names the
    // image id its hash reads.
    let spec = model::arch::spec(split)?.spec;
    let facts = TextFacts {
        ple: PleImage::of_spec(&spec),
        ..TextFacts::of_imrope(&spec)?
    };
    let file = a.vision.encoder(true)?;
    if let Some(enc) = file.as_deref()
        && is_minicpm_projector(enc)
    {
        return open_minicpm(
            enc,
            &a.vision,
            a.place.unwrap_or(Place::A),
            &facts,
            token,
            one,
        )
        .map(Some);
    }
    vision_seat::open_seat(
        &a.vision,
        file.as_deref(),
        a.place.unwrap_or(Place::A),
        &[vision::arch::qwen3vl::PROJECTOR_TYPE],
        &facts,
        token,
    )
}

/// Whether `path` opens as a `minicpmv4_7` encoder. A file that does not open, or whose projector
/// type is absent, is left to [`vision_seat::open_seat`], which reports that failure.
fn is_minicpm_projector(path: &Path) -> bool {
    let Ok(file) = gguf::Gguf::open(path) else {
        return false;
    };
    file.value("clip.projector_type")
        .and_then(gguf::Value::as_str)
        == Some(vision::arch::minicpmv::PROJECTOR_TYPE)
}

/// A `minicpmv4_7` encoder: the file's hyperparameters, the text tokenizer's marks and the `<unk>`
/// pad ([`vision::arch::minicpmv::Media::new`]), held against the text model. The tower is
/// sized for [`vision_seat::MINICPM_TILE_TOKENS`] (1120 patches) and seated on `place`.
/// `one` is a string that tokenizes to exactly one id.
#[cfg(feature = "vision")]
fn open_minicpm(
    path: &Path,
    flags: &vision_seat::Flags,
    place: Place,
    text: &TextFacts,
    token: &dyn Fn(u32) -> Option<String>,
    one: &dyn Fn(&str) -> Option<u32>,
) -> Result<Opened, GateError> {
    use vision::arch::minicpmv::{self, Hparams, Marks, Media};
    use vision::arch::qwen3vl::TokenLimits;

    let at =
        |detail: &str| -> GateError { format!("--mmproj {}: {detail}", path.display()).into() };
    let file = gguf::Gguf::open(path).map_err(|e| at(&e.to_string()))?;
    let rule = vision_seat::rules()
        .iter()
        .find(|r| r.projector == minicpmv::PROJECTOR_TYPE)
        .ok_or_else(|| {
            at(&format!(
                "projector type {:?} has no rule in the vision seat's table",
                minicpmv::PROJECTOR_TYPE
            ))
        })?;
    if flags.sets_limits() && !rule.takes_limits {
        return Err(at(&format!(
            "the {} projector plans an image from its file's own grid; \
             --image-min-tokens and --image-max-tokens steer only the projectors whose \
             size rule they bound",
            minicpmv::PROJECTOR_TYPE
        )));
    }
    let hp = Hparams::read(&file).map_err(|e| at(&e.to_string()))?;
    hp.check_text_width(text.n_embd)
        .map_err(|e| at(&e.to_string()))?;
    // The card borrows `hp`. `Media::new` takes `hp` after that.
    let limits =
        TokenLimits::new(1, vision_seat::MINICPM_TILE_TOKENS).map_err(|e| at(&e.to_string()))?;
    let patches = vision_seat::MINICPM_TILE_TOKENS * vision_seat::MINICPM_PATCHES_PER_TOKEN;
    let card = minicpmv::card::of(&hp, patches);
    let marks = Marks::lookup(one).map_err(|e| at(&e.to_string()))?;
    // The pad the group's rows expand to: the vocabulary's `<unk>`, one token.
    const PAD_TEXT: &str = "<unk>";
    let pad = one(PAD_TEXT).ok_or_else(|| {
        at(&format!(
            "the pad {PAD_TEXT:?} does not tokenize to exactly one token"
        ))
    })?;
    let projector = vision_seat::Projector {
        path: path.to_owned(),
        rule,
        card,
        out_dim: hp.out_dim,
        media: Arc::new(Media::new(hp, marks, PAD_TEXT, pad)),
        cap: limits.max(),
        limits,
    };
    projector.check_text(text, token)?;
    let seat = vision_seat::encoder_seat(place, projector.card.total())?;
    Ok(vision_seat::Opened {
        path: projector.path,
        seat,
        media: projector.media,
        cap: projector.cap,
        card: projector.card,
        limits: projector.limits,
    })
}

/// The same open on a build whose tower is not compiled. The file is refused by name. No card
/// figure and no seat: [`vision_seat::encoder_seat`] reads the census, and this build has no tower
/// to put there.
#[cfg(not(feature = "vision"))]
fn open_minicpm(
    path: &Path,
    flags: &vision_seat::Flags,
    place: Place,
    text: &TextFacts,
    token: &dyn Fn(u32) -> Option<String>,
    one: &dyn Fn(&str) -> Option<u32>,
) -> Result<Opened, GateError> {
    let _ = (path, flags, place, text, token, one);
    Err(vision_seat::NO_TOWER.into())
}

/// The refusal of a draft (`word`, the `BLOOMERY_DRAFT` value; `name`, how
/// the sentence names it) on a qwen35moe load placed on the host tier
/// (`placement` and `why` are what it resolved to): the draft's verify pass
/// needs the whole model on the card. The leading phrase is the release
/// pass's key.
fn whole_card_refusal(word: &str, name: &str, (placement, why): (String, String)) -> GateError {
    format!(
        "BLOOMERY_DRAFT={word} needs the whole-card load: {placement} resolved to a placed \
         load, {why}; the {name} draft verifies its rows on a model held whole on the card - \
         serve a load that holds the whole model there, or no draft"
    )
    .into()
}

/// The body's words for a placed load ([`Body35::build`]'s doc and its lane
/// refusal): a placed load runs no verify.
fn placed_no_verify() -> String {
    let lanes = Body35::VERIFY_LANES;
    format!(
        "a placed load runs no verify; {lanes} lanes a delta store: a load keeps 1, or {lanes} \
         on a whole-card load that verifies drafted rows"
    )
}

/// [`Body35::new_seq`]: the draft's K/V is one cache on the body.
const ONE_SEQUENCE: &str = "an in-file MTP load supports one resident sequence";

/// [`Body35`]'s media refusal of the next-token layer.
const MEDIA_DRAFT: &str = "a media load: its sequences' rotation tables are their own, and the \
     draft reads the plain table";

/// What a qwen35moe or qwen35 file does with `BLOOMERY_DRAFT` unset, `off`
/// or `mtp`, after the load is known.
enum Choice {
    /// The file carries a shaped next-token layer and the seat drafts it.
    InFile,
    /// The plain path, and why, printed on `load draft=off`.
    Plain(String),
}

/// The positions the in-file window needs over `slots` slots: the verify's
/// two rows and one more, the same count the Qwen3.8 seat's window uses.
fn window_need(slots: usize) -> usize {
    slots.saturating_mul(<Body35 as MtpBody>::VERIFY_ROWS + 1)
}

/// The file's next-token layer, or why it is not a draft. A missing layer
/// is the plain path: the why is the enum's own words, the stripped block's
/// reason, or the error's.
fn layer_choice(file: &Split) -> Choice {
    match in_file(file) {
        Ok(InFile::Draft(_)) => Choice::InFile,
        Ok(InFile::Absent) => Choice::Plain(
            "The key is absent, or the blocks past the trunk are trunk layers.".to_owned(),
        ),
        Ok(InFile::Unshaped) => Choice::Plain(
            "The key is 1 and every tail tensor is a one-element vector: the role table's stub, \
             not a draft's shapes."
                .to_owned(),
        ),
        Ok(InFile::Stripped { reason }) => Choice::Plain(reason),
        Err(e) => Choice::Plain(e.to_string()),
    }
}

/// Why a checkpoint folder drafts no next-token layer: its map leaves the folder's `mtp.*` tensors
/// out of the load.
const FOLDER_NO_MTP: &str = "a checkpoint folder's mtp.* tensors are not read";

/// `BLOOMERY_DRAFT` `mtp`, `off` or unset on a checkpoint folder, whose next-token layer the load
/// does not read: set `mtp` is refused by name, `off` and unset take the plain path and print why
/// on `load draft=off`.
fn folder_choice(draft: Option<&str>) -> Result<Choice, GateError> {
    match draft {
        Some("mtp") => Err(format!(
            "BLOOMERY_DRAFT=mtp: {FOLDER_NO_MTP}; serve the folder with BLOOMERY_DRAFT unset or \
             off, or a GGUF file that carries the layer"
        )
        .into()),
        Some("off") => Ok(Choice::Plain(Draft38Off::Set.to_string())),
        None => Ok(Choice::Plain(FOLDER_NO_MTP.to_owned())),
        Some(other) => {
            Err(format!("BLOOMERY_DRAFT {other:?} is not an in-file draft decision").into())
        }
    }
}

/// `BLOOMERY_DRAFT` `mtp`, `off` or unset on a qwen35moe or qwen35 file, once
/// the load is known. `whole` is a load that opens the model on the card.
/// Set `mtp` is refused by name where the load cannot draft. Unset takes
/// [`draft38_unset`] and then the seat's own facts; a fact that forbids the
/// draft leaves the plain path and names why.
#[allow(
    clippy::too_many_arguments,
    reason = "the load's levers, file, slots, context, card, place and media decide the draft (rust-quality R8)"
)]
fn decide_q35(
    levers: &Levers,
    path: &Path,
    file: &Split,
    slots: usize,
    slot_ctx: usize,
    whole: bool,
    place_name: &str,
    media: bool,
) -> Result<Choice, GateError> {
    let need = window_need(slots);
    let forbid_set = |why: String| -> Result<Choice, GateError> {
        Err(format!("BLOOMERY_DRAFT=mtp: {why}").into())
    };
    match levers.draft() {
        Some("mtp") => {
            if media {
                return forbid_set(MEDIA_DRAFT.to_owned());
            }
            if !whole {
                return forbid_set(placed_no_verify());
            }
            if slots > 1 {
                return forbid_set(ONE_SEQUENCE.to_owned());
            }
            if slots.saturating_mul(slot_ctx) < need {
                return Err(format!(
                    "a slot's context of {slot_ctx} leaves no positions for the MTP draft's \
                     window; --ctx-size names a total of at least {need}"
                )
                .into());
            }
            Ok(layer_choice(file))
        }
        Some("off") => Ok(Choice::Plain(Draft38Off::Set.to_string())),
        None => {
            let at = Draft38At {
                place_a: whole && place_name == "a",
                logits: false,
                route_trace: levers.route_trace().is_some(),
                file: path,
                file_is_there: true,
                looked: &[],
                need,
                ctx: slot_ctx.saturating_mul(slots),
            };
            let mut off = draft38_unset(&at);
            if matches!(off, Some(Draft38Off::Gate)) && !whole && place_name == "a" {
                off = None;
            }
            if let Some(why) = off {
                return Ok(Choice::Plain(why.to_string()));
            }
            if media {
                return Ok(Choice::Plain(Draft38Off::Media.to_string()));
            }
            if !whole {
                return Ok(Choice::Plain(placed_no_verify()));
            }
            if slots > 1 {
                return Ok(Choice::Plain(ONE_SEQUENCE.to_owned()));
            }
            Ok(layer_choice(file))
        }
        Some(other) => {
            Err(format!("BLOOMERY_DRAFT {other:?} is not an in-file draft decision").into())
        }
    }
}

/// The delta lanes the seat's whole-card load of a qwen35 family file keeps:
/// [`Body35::VERIFY_LANES`] on a load that verifies drafted rows
/// (`GpuModel::open_verifying`, `GpuModel::open_mtp`), one on a plain load —
/// the count its whole-fit verdict prices the load at
/// ([`q3place::serve_choice_lanes`]), decided before the load is. The draft
/// word decides ahead of it: `dspark` and `lookup` verify a whole-card load
/// whatever the file; `mtp` and an unset word draft the file's own
/// next-token layer ([`layer_choice`]), which needs one text sequence on
/// place `a` with no route trace. [`decide_q35`] keeps the load's own half —
/// a placed load drafts nothing, and its refusals stand — so where this pick
/// cannot know the load it takes the wider side: the verdict it feeds never
/// under-counts the lanes of the load the seat opens.
fn verdict_lanes(
    levers: &Levers,
    word: Word,
    file: &Split,
    place: &str,
    media: bool,
    slots: usize,
) -> usize {
    let in_file = matches!(layer_choice(file), Choice::InFile);
    let drafted = match word {
        Word::Dspark | Word::Lookup => true,
        Word::Mtp => !media && slots == 1 && in_file,
        Word::Off => {
            levers.draft().is_none()
                && place == "a"
                && !media
                && slots == 1
                && levers.route_trace().is_none()
                && in_file
        }
    };
    if drafted { Body35::VERIFY_LANES } else { 1 }
}

/// [`Word`] from `BLOOMERY_DRAFT` for the `arch` file at `path`, `name` its
/// architecture. `lookup` is refused by name on a qwen3moe file
/// ([`lookup_refusal`]). `dspark` is the draft a qwen35moe file serves.
/// `mtp` on a qwen3moe file is refused by name before any plan: that body
/// has no in-file next-token layer and runs no verify pass.
fn draft_of(arch: Family, name: &str, levers: &Levers, path: &Path) -> Result<Word, GateError> {
    let word = served::word(levers)?;
    match (word, arch) {
        (Word::Lookup, Family::Qwen3moe) => Err(lookup_refusal(arch).into()),
        (Word::Dspark, Family::Qwen3moe | Family::Qwen35) => Err(format!(
            "BLOOMERY_DRAFT=dspark: the DSpark draft serves a qwen35moe file (Qwen3.6-35B-A3B \
             or MiniCPM-V 4.7-35B-A3B); {} is a {name} file",
            path.display()
        )
        .into()),
        (Word::Mtp, Family::Qwen3moe) => Err(
            "BLOOMERY_DRAFT=mtp: a qwen3moe file has no in-file next-token layer and its body \
             runs no verify pass"
                .into(),
        ),
        (word, _) => Ok(word),
    }
}

/// `word` as the served draft the lookup and DSpark paths take. `mtp` and
/// the plain word are [`Draft::Off`]: the in-file decision is [`decide_q35`]'s,
/// and [`served::draft_slots`] decides a count only for DSpark.
fn served_draft(word: Word) -> Draft {
    match word {
        Word::Lookup => Draft::Lookup,
        Word::Dspark => Draft::Dspark,
        Word::Off | Word::Mtp => Draft::Off,
    }
}

/// Why `BLOOMERY_DRAFT=lookup` is refused on a qwen3moe file: the lookup
/// draft needs a load whose session verifies several rows, and the qwen3moe
/// body runs no pass of several rows.
///
/// PIN(2026-10-22): the qwen35moe and qwen35 arms of this refusal are gone —
/// the seat now opens the whole-card verifying load (`GpuModel::<Body35>::open_verifying`)
/// that the lookup draft needs, so a qwen35moe file serves the lookup and this
/// sentence is the qwen3moe body's own reason alone. The gate that proves it
/// (`gate_qwen3_dspark`) gained the clause that the lookup's ids are the plain
/// run's and the clause that `/props` names the lookup as the served draft.
fn lookup_refusal(arch: Family) -> String {
    format!(
        "BLOOMERY_DRAFT=lookup on a {} file: the lookup draft needs a load that verifies \
         several rows, and the qwen3moe body runs no pass of several rows; serve the plain path",
        arch.name()
    )
}

/// The input restrictions of the in-file draft independent of model headers.
fn check_mtp_inputs(slots: usize, media: bool) -> Result<(), GateError> {
    if slots != 1 {
        return Err(format!("BLOOMERY_DRAFT=mtp: {ONE_SEQUENCE}").into());
    }
    if media {
        return Err(format!("BLOOMERY_DRAFT=mtp: {MEDIA_DRAFT}").into());
    }
    Ok(())
}

/// The levers the seat acts on for `args`: a placed load (`--place`) acts on the placement's levers
/// too; set beside an unplaced one, they are refused by name. Every load reads its ubatch size in
/// place.
fn acts_on(args: &[String]) -> Vec<&'static str> {
    let mut acts_on = vec![
        bloomery_levers::DRAFT,
        bloomery_levers::MTP_WIDTH,
        bloomery_levers::QWEN3_KV,
        bloomery_levers::STEP_STATS,
        bloomery_levers::SCREEN,
        bloomery_levers::SCHEMAJUMP,
        UBATCH_ENV,
    ];
    if bloomery_gpu_gates::llama_launch::serve_option(args, "--place") {
        acts_on.extend(q3place::PLACED_LEVERS);
    }
    acts_on
}

/// The levers (unless `read_levers` is false) and flags [`run`] reads first, checked before the
/// model is fetched ([`bloomery_gpu_gates::bind::check_seat`]).
pub(crate) fn check(
    args: &[String],
    read_levers: bool,
    from_repo: bool,
) -> Result<bloomery_gpu_gates::bind::CheckedSeat<Args>, GateError> {
    bloomery_gpu_gates::bind::check_seat(
        read_levers.then(|| acts_on(args)).as_deref(),
        args,
        |args, levers| {
            let a = parse_args(args)?;
            a.template_file.check()?;
            a.vision.check_before_model(from_repo)?;
            KvQ8::choose(
                a.cache_type_k.as_deref(),
                levers.and_then(Levers::qwen3_kv_set),
                KvQ8::F16,
            )?;
            if let Some(levers) = levers {
                let word = served::word(levers)?;
                WidthMode::of(levers.mtp_width())?;
                bind::refuse_plain_width(
                    levers,
                    (levers.draft() == Some("off")).then_some("BLOOMERY_DRAFT=off"),
                )?;
                served::draft_slots(a.parallel, served_draft(word), true)?;
                if word == Word::Mtp {
                    let (slots, _) = slots_given(a.parallel, a.ctx, DEFAULT_SLOTS)?;
                    check_mtp_inputs(slots, a.vision.mmproj()?.is_some())?;
                }
            }
            Ok(a)
        },
    )
}

/// Loads the model, and the embedding model `side` names beside it, and serves until the listener
/// or the engine fails; `Ok` carries why the server ended.
pub fn run(args: &[String], side: &side_seat::Flags) -> Result<ServeError, GateError> {
    record::at_main_with_flags(NAME, record::BLOOMERY_SERVE_QWEN3, |flag| {
        bloomery_gpu_gates::llama_launch::serve_option(args, flag)
    });
    let (a, levers) = check(
        args,
        true,
        bloomery_gpu_gates::model_file::hf_repo().is_some(),
    )?
    .run();

    arm_screen(levers.screen());
    let kv = KvQ8::choose(a.cache_type_k.as_deref(), levers.qwen3_kv_set(), KvQ8::F16)?;
    let path = ref_model_path()?;
    // A directory is a checkpoint folder, opened once here: its located header plans as a
    // split's first shard does, and the header and part source go to the engine thread.
    let (weights, folder_tokenizer) = Weights::open(&path)?;
    let split = weights.header();
    let name = split.architecture().unwrap_or("<missing>").to_owned();
    let arch = Family::of(split).map_err(|e| format!("{}: {e}", path.display()))?;
    // The draft word, before anything is planned or loaded.
    let word = draft_of(arch, &name, &levers, &path)?;
    let served = served_draft(word);
    // The width a drafted window verifies: the chooser's (`cost`, unset) or the
    // draft's own (`fixed`). An explicit off, and an unset qwen3moe file, refuse
    // it here. A qwen35 file whose default may still draft does not.
    let width = WidthMode::of(levers.mtp_width())?;
    bind::refuse_plain_width(
        &levers,
        (levers.draft() == Some("off")
            || (matches!(arch, Family::Qwen3moe) && levers.draft().is_none()))
        .then_some("no draft selected"),
    )?;
    // The width chooser's record is printed on every request that may draft.
    let may_draft = matches!(word, Word::Lookup | Word::Dspark | Word::Mtp)
        || (levers.draft().is_none() && matches!(arch, Family::Qwen35moe | Family::Qwen35));
    if may_draft {
        super::drafted::serves_widths();
    }
    // The vocabulary before the plans: `--mmproj`'s encoder is held against it, and its bytes
    // are a reserve of every plan that follows.
    let ChatSurface {
        template,
        name: alias,
        ..
    } = ChatSurface::read(
        &path,
        split.value("tokenizer.chat_template"),
        split.value("general.name"),
        a.template_file.path(),
        "qwen3",
    )?;
    let tokenizer = match folder_tokenizer {
        Some(tokenizer) => tokenizer,
        None => Tokenizer::from_gguf(&path)?,
    };
    let vocab = Arc::new(Vocab::new(tokenizer)?.with_chat_template(&template)?);
    let tok = vocab.tokenizer();
    let token = |id| tok.text(id).map(str::to_owned);
    // A string that is exactly one token: MiniCPM's marks and its `<unk>` pad.
    let one = |text: &str| {
        let ids = tok.encode(text, false, true);
        (ids.len() == 1).then(|| ids[0])
    };
    let vision = vision_of(&a, arch, (&path, split), &token, &one)?;
    // Keep the stage context and bundle resident before planning, as the CLI does.
    let stage = a
        .place
        .unwrap_or(Place::A)
        .on(&gpu_census::census()?)?
        .card_specs()?[0];
    Gpu::hold_card(stage.name, stage.device)?;
    // The side model is read and seated before the plans, each of which holds its bytes when it
    // sits on the stage card; it takes the card the encoder's seat takes.
    let side = side.read()?;
    let side_seat = side
        .as_ref()
        .map(|s| s.seat(a.place.unwrap_or(Place::A)))
        .transpose()?;
    let embedding = side_seat.as_ref().and_then(|s| s.reserve);
    // The slot count before the context it sizes: a set `--ctx` with no
    // `--parallel` is one request's context — one slot at the whole of it
    // (`bind::slots_of`).
    let (slots, from) = slots_given(a.parallel, a.ctx, DEFAULT_SLOTS)?;
    // The DSpark draft's window is one sequence's: several slots are refused by name.
    // `mtp` decides no count here.
    let (slots, from) = served::draft_slots(a.parallel, served, true)?.unwrap_or((slots, from));
    // The delta lanes the seat's whole-card qwen35 family load keeps
    // ([`verdict_lanes`]): every whole-fit verdict below prices the load at
    // them, before the load is known.
    let lanes = match arch {
        Family::Qwen35moe | Family::Qwen35 => verdict_lanes(
            &levers,
            word,
            split,
            a.place.unwrap_or(Place::A).name(),
            vision.is_some(),
            slots,
        ),
        Family::Qwen3moe => 1,
    };
    // A projector the `--hf` repo picked stays only where the load gives the text model the same
    // experts and context with the encoder as without ([`vision_seat::kept`]).
    let vision = vision_seat::kept(vision, &a.vision, |v| {
        let plan_levers = PlanLevers::from_levers(&levers)?;
        let asked = ctx_and_load(
            split,
            (a.ctx, slots),
            serve_opts((a.place, &plan_levers), (kv, a.ubatch), (v, embedding)),
            lanes,
        )?;
        let cost_ctx = asked.total;
        Ok(match asked.plan {
            Q3Load::Whole(_) | Q3Load::Placed { whole: Some(_), .. } => {
                vision_seat::Cost::whole(cost_ctx)
            }
            Q3Load::Placed { q, total, .. } => {
                vision_seat::Cost::of(&q.plan(total, &plan_levers)?, cost_ctx)
            }
        })
    })?;
    // The common unset rule, once, on one census reading: the load below
    // still takes `a`'s card (`PlaceQ3::unplaced_on`), the rule's own pick
    // for a body with no tier card; its `place unset` record is printed here.
    let census = gpu_census::census()?;
    let chosen = q3place::choose(a.place, &census)?;
    chosen.record().eprint();
    // The context and the load of the slots the flags or the default name,
    // one choice: the ask first, the plan around it ([`ctx_and_load`]). A
    // count is never lowered: the default is one slot.
    let plan_levers = PlanLevers::from_levers(&levers)?;
    let asked = ctx_and_load(
        split,
        (a.ctx, slots),
        serve_opts(
            (a.place, &plan_levers),
            (kv, a.ubatch),
            (vision.as_ref(), embedding),
        ),
        lanes,
    )?;
    let (_, image) = vision_seat::images_of(vision.as_ref());
    eprintln!(
        "{}{}",
        asked.line(),
        beside_words(Beside { image, embedding })
    );
    let slot_ctx = slot_ctx(usize::try_from(asked.total)?, slots)?;
    let load = asked.plan;
    if matches!(arch, Family::Qwen35moe | Family::Qwen35) {
        q3place::checkpoints_fit(seqs35(split, &load, slots)?, slot_ctx, load.placed_body())?;
    }
    // The shape of a round of several slots, from the load's body
    // ([`Q3Load::placed_body`], [`Q3Load::tiered`]); the open holds the
    // opened body to it.
    let placed = load.placed_body();
    // A plan that hangs an expert tier holding an expert runs its rounds a
    // slot at a time ([`Body3::serves_one_pass`]: the placed body refuses a
    // pass of several slots beside one).
    let tiered = load.tiered(&plan_levers)?;
    if served != Draft::Off && placed {
        let why = match a.place {
            Some(p) => (
                format!("--place {}", p.name()),
                "its plan puts routed experts on the host tier".to_owned(),
            ),
            None => (
                format!("--place unset on {}", chosen.place.name()),
                "the whole model does not fit the card's free bytes, so the plan puts routed \
                 experts on the host tier"
                    .to_owned(),
            ),
        };
        return Err(match served {
            Draft::Dspark => whole_card_refusal("dspark", "DSpark", why),
            Draft::Lookup => whole_card_refusal("lookup", "lookup", why),
            Draft::Off => unreachable!("the plain path is never placed away"),
        });
    }
    let choice = if matches!(arch, Family::Qwen35moe | Family::Qwen35)
        && matches!(word, Word::Off | Word::Mtp)
    {
        Some(if weights.is_folder() {
            folder_choice(levers.draft())?
        } else {
            decide_q35(
                &levers,
                &path,
                split,
                slots,
                slot_ctx,
                !placed,
                chosen.place.name(),
                vision.is_some(),
            )?
        })
    } else {
        None
    };
    bind::refuse_plain_width(
        &levers,
        match &choice {
            Some(Choice::Plain(why)) => Some(why),
            Some(Choice::InFile) | None => None,
        },
    )?;
    let draft_dir = match word {
        Word::Dspark => Some(dspark_hf::dir_from_env()?),
        Word::Off | Word::Lookup | Word::Mtp => None,
    };
    let pass = match arch {
        Family::Qwen3moe => pass_of::<Body>(slots, tiered, served),
        _ => pass_of::<Body35>(slots, tiered, served),
    };
    // The slots the seat serves, one line before the load (`from` names what
    // set the count).
    parallel_line(slots, slot_ctx, from, pass);
    if a.plan_only {
        // The records the load would print first are out; no model was
        // opened on a card (the stage card was held above).
        load.print_plan(&levers, vision.as_ref())?;
        bloomery_gpu_gates::start::plan_exit();
    }
    // `/props`' placement: a placed load's plan, the one its open makes from the same plan and
    // levers; a whole-card load opens with no plan, so it reports none and says so.
    let placement = match &load {
        Q3Load::Placed { q, total, .. } => {
            let plan = q.plan(*total, &plan_levers)?;
            placement_of(NAME, plan.machine, &plan)
        }
        Q3Load::Whole(_) => {
            eprintln!(
                "{NAME}: /props leaves the placement out: a whole-card load opens with no plan"
            );
            None
        }
    };
    let props = EngineProps {
        placement,
        ..EngineProps::default()
    };
    let at = weights.hand_over(&path);
    let open = path.clone();
    let device = match &load {
        Q3Load::Whole(on) => on.card.name,
        Q3Load::Placed { q, .. } => q.name(),
    }
    .to_owned();
    let schema_jump = levers.schema_jump();
    let engine = match (arch, choice) {
        (Family::Qwen3moe, _) => SeatEngine::spawn(
            move || {
                Q3::<Body>::open(
                    (&open, at),
                    slot_ctx,
                    slots,
                    pass,
                    load,
                    &levers,
                    (kv, vision),
                    served,
                    draft_dir,
                    width,
                    None,
                )
            },
            slot_ctx,
            vocab,
            device,
            props,
            0,
        )?,
        (_, Some(Choice::InFile)) => SeatEngine::spawn(
            move || Q35Mtp::open(&open, slot_ctx, slots, load, &levers, (kv, vision), width),
            slot_ctx,
            vocab,
            device,
            props,
            0,
        )?,
        (_, Some(Choice::Plain(why))) => SeatEngine::spawn(
            move || {
                Q3::<Body35>::open(
                    (&open, at),
                    slot_ctx,
                    slots,
                    pass,
                    load,
                    &levers,
                    (kv, vision),
                    Draft::Off,
                    None,
                    width,
                    Some(why),
                )
            },
            slot_ctx,
            vocab,
            device,
            props,
            0,
        )?,
        (_, None) => SeatEngine::spawn(
            move || {
                Q3::<Body35>::open(
                    (&open, at),
                    slot_ctx,
                    slots,
                    pass,
                    load,
                    &levers,
                    (kv, vision),
                    served,
                    draft_dir,
                    width,
                    None,
                )
            },
            slot_ctx,
            vocab,
            device,
            props,
            0,
        )?,
    };
    let side = side_seat::Side::start(side.zip(side_seat))?;
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
            preparation: None,
            slot_save_path: None,
            api_keys: a.api_keys,
            parallel: slots,
            queue_depth: a.queue_depth,
            schema_jump,
            side,
        },
    )?;
    Record::new(&record::LISTENING_QWEN3)
        .w("arch", &name)
        .u("ctx", slot_ctx)
        .u("slots", slots)
        .u("slot_ctx", slot_ctx)
        .w("addr", server.local_addr()?)
        .eprint();
    Ok(server.run())
}

#[cfg(test)]
mod tests {
    use super::{Choice, folder_choice, parse_args};
    use model::placement::workstation::{A6000, RTX_3090};

    /// The open's round shape is the body's fact: Qwen3.6's body keeps a
    /// pass's rows whole, so a drafted load takes turns while a plain one runs
    /// one pass, and one slot is one pass whatever the body does. Mutant:
    /// drop `SETTLES_PARTIAL_KEEP` from `bind::verifies_slots`.
    #[test]
    fn a_drafted_qwen36_round_takes_turns_by_what_its_body_keeps() {
        use super::{Body35, Draft, Pass, pass_of};
        assert_eq!(pass_of::<Body35>(2, false, Draft::Lookup), Pass::Turns);
        assert_eq!(pass_of::<Body35>(2, false, Draft::Off), Pass::One);
        assert_eq!(pass_of::<Body35>(1, false, Draft::Lookup), Pass::One);
    }

    /// The seat reads `--place` as the shared grammar and leaves a tier card
    /// to its plan: each word it took before parses to the placement it names,
    /// a list of tier cards is not refused at parse, and a word no card or
    /// ordinal spells is.
    #[test]
    fn the_place_word_is_the_shared_grammar_and_a_tier_count_is_the_plans() {
        let args = |word: &str| [String::from("--place"), word.to_owned()];
        for (word, name, cards) in [
            ("a", "a", vec![A6000.name]),
            ("gate", "gate", vec![RTX_3090.name]),
            ("bp", "bp", vec![A6000.name, RTX_3090.name]),
            ("a6000+3090", "bp", vec![A6000.name, RTX_3090.name]),
            ("cuda1+0", "cuda1+0", vec!["cuda1", "cuda0"]),
            ("tp", "tp", vec![A6000.name]),
        ] {
            let a = parse_args(&args(word)).unwrap_or_else(|e| panic!("{word}: {e}"));
            let p = a.place.expect(word);
            assert_eq!((p.name(), p.cards()), (name, cards), "{word}");
        }
        let many = "cuda0+cuda1+cuda2";
        let a = parse_args(&args(many)).unwrap_or_else(|e| panic!("{many}: {e}"));
        assert_eq!(a.place.map(|p| p.tier_cards().len()), Some(2));
        let Err(e) = parse_args(&args("b2")) else {
            panic!("b2 is no card or ordinal");
        };
        assert!(e.to_string().starts_with("--place is a, gate, bp"), "{e}");
    }

    /// A checkpoint folder's next-token layer is not read: `BLOOMERY_DRAFT=mtp` is refused by
    /// name, and `off` and unset take the plain path and say why.
    #[test]
    fn a_checkpoint_folder_serves_no_next_token_layer() {
        let e = folder_choice(Some("mtp")).err().unwrap().to_string();
        assert!(
            e.contains("BLOOMERY_DRAFT=mtp") && e.contains("mtp.* tensors are not read"),
            "{e}"
        );
        let Ok(Choice::Plain(why)) = folder_choice(None) else {
            panic!("unset is the plain path");
        };
        assert!(why.contains("mtp.* tensors are not read"), "{why}");
        let Ok(Choice::Plain(why)) = folder_choice(Some("off")) else {
            panic!("off is the plain path");
        };
        assert_eq!(why, "BLOOMERY_DRAFT=off");
        let e = folder_choice(Some("dspark")).err().unwrap().to_string();
        assert!(e.contains("is not an in-file draft decision"), "{e}");
    }
}
