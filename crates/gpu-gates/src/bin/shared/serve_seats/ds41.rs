//! `bloomery-serve-ds41` — the llama-server-compatible HTTP API on the V4.1 engine.
//!
//! This module is the seat: `bloomery-serve-ds41` and `bloomery-serve --model
//! ds41` are each one call of [`run`], which takes the process's arguments
//! (`--model` already taken out by the one-binary server).
//!
//!     bloomery-serve-ds41 [--host 127.0.0.1] [--port 8080] [--place PLACE]
//!                         [--ctx C] [--alias NAME] [--cache-ram MIB]
//!                         [--parallel N] [--slot-ctx S] [--queue-depth Q] [--slot-save-path DIR]
//!                         [--chat-template-file PATH]
//!                         [--mmproj FILE | --no-mmproj] [--image-min-tokens N]
//!                         [--image-max-tokens N] [--plan]
//!
//! `--plan` prints the lines the decisions before the load make (the `ctx`
//! line, the `parallel` line) and the plan's own (the `plan` and `need`
//! records, the `residency` records beside them) and exits 0 before the model
//! is opened on a card.
//!
//! `--ctx` is also spelled `--ctx-size` and `-c`, as llama-server spells it
//! (`serve::flag::CTX`); a number flag's value that is not a number is
//! refused naming the flag and the value.
//!
//! `PLACE` is `a`, `gate`, `bp` or a card list `<stage>[+<tier>…]`
//! (`generate::Place`, as `generate_ds41` takes it); a list whose stage card
//! is not the A6000 and a list of more tier cards than the V4.1 body serves
//! are refused by name before any plan. Unset it is the common rule's
//! (`generate::Place::choose` by `place::TIER_RULE`): `a`, and a `place unset`
//! record names the placement and why, first of the seat's records.
//!
//! The model is `$BLOOMERY_REF_MODEL`; its first shard gives the vocabulary,
//! the chat template (`tokenizer.chat_template`) and the default alias
//! (`general.name`). The `place unset` record, the `residency lever` line,
//! then the plan line, then the `load`, host and `capture` lines of
//! [`Generator::open`] go to stderr
//! as `generate_ds41` prints them,
//! then `listening on http://<addr>` once the model is loaded and the port is
//! bound (`--port 0` binds a free one). `BLOOMERY_RESIDENCY` unset resolves
//! against the plan the seat prints (`residency::at_plan`): the plan-side
//! detail on the line beside the lever record when the plan moved or refused
//! the word. Sampling is the sampler crate's chain
//! with no repetition penalty; `temperature <= 0` is the engine's argmax, the
//! ids `generate_ds41 --tokens <the prompt's ids>` prints.
//!
//! `/props`' `engine` object carries the file's header facts and the printed
//! plan's resident bytes per device and class, each card named by its
//! nvidia-smi index — under `--place bp` (plan (b′): plan (a) on the A6000,
//! the 3090 an expert tier; see `generate_ds41`) the tier card's row too,
//! with no `layers`; when an index cannot be found the placement is left
//! out, with the reason on stderr. Under `bp` a prompt call is fed as
//! `BLOOMERY_PREFILL` says, the tier serving its experts' slots of each
//! batch, and a lost tier card is an engine error like any other: the
//! request's 500 and `/health`'s 503 carry its message, which names the card.
//!
//! Each slot serves its share of `--ctx` (below): that is `/props`' `n_ctx`,
//! a prompt that long is a 400 before it reaches the engine, and generation
//! stops there with `truncated`. The context is the seats' common choice
//! (`place::ctx_choice`, `generate::choose_ctx`): it comes first, the experts fill
//! what is left. The limit is the file's trained context within
//! `plan::KERNEL_POSITIONS`. Set, `--ctx` is the total the slots split (llama-server's
//! `-np` without `-kvu`); a share past the limit is refused by name, and one
//! the machine cannot hold is refused with the largest share it holds,
//! before the load. Unset, a slot asks the limit and, where that plan is short
//! of the card, takes the largest context under it whose plan holds, down to
//! the floor; one `ctx` line on stderr names the rule. The image encoder stays
//! on, and when its reserve lowers that context the same line names the
//! positions a slot gets with the reserve and without it.
//! `/props`' `engine.ctx_verified` is the
//! deepest context the reference sets hold our numbers to ik's at
//! (`refset::arch::deepseek41::VERIFIED_POSITIONS`), which bounds nothing.
//!
//! The prompt cache (llama-server's `--cache-ram`, in MiB; 0 turns it off)
//! holds the body's saved sequence states in host RAM. Its default is the
//! shared [`CacheRam::of`] budget: the lesser of the cap
//! (`bind::CACHE_RAM_CAP`) and half of what `MemAvailable` leaves past the
//! plan's host need and the residency's churn pool (V4.1's body pins no
//! checkpoints); the `cache` record prints
//! it, and the token a prompt call is cut at so a
//! later request keeps the start of a user message ([`USER_START`], with
//! whether the chat template writes it). Every cache event and every prefix
//! the body keeps less of than a request shares is a record line.
//!
//! `--parallel N` (`-np N`, default one) serves N resident sequences
//! inside the one model (`Session::add_slots` over the body's `Slots`): the
//! server steps the busy slots together, the sequences switched by pointer
//! exchange; nothing parks and no slot waits for another. At a fixed
//! placement — `BLOOMERY_RESIDENCY=off`, as `--place gate` always runs, or
//! before any residency flip lands — each request's tokens are its solo
//! run's; under adaptive residency every slot's passes move the placement,
//! and the placement changes the bits. A load of more than one resident
//! slot with no draft runs each round as one pass of the busy rows
//! ([`V41::step_slots`] through `serve_seats::rounds::step_rows_one_pass` —
//! two slots' rows one layer apart, the body's two-row slot pass); every
//! other load keeps the fallback loop, a select and a step a row: one slot
//! (a round of several never runs), the DSpark draft (which serves one
//! slot), and the lookup draft (whose [`Seat::step`] feeds its tables, and
//! whose drafted rounds are [`Seat::pass_slots`] rounds). Under
//! `BLOOMERY_STEP_STATS=1` each round of several slots prints a
//! `slots round` record naming its command, rows, passes and the slots the
//! seat serves, and the step rows keep their engram fill statistics (clock
//! reads, page-classifying syscalls) — a server run under it is never a
//! timing reading.
//! `--parallel 1` is exactly the one-sequence server, and `--parallel 0` is
//! refused by name. A slot's share under the pair pass's rows is refused by
//! name before the load ([`slot_ctx`]). The plan counts every sequence —
//! each one's caches on the card and ring shadows on the host
//! (`PlanInputs::plan_with_slots`) — so slots the machine cannot hold are
//! refused before the load by the plan's own refusal, as one sequence's are
//! (the card's floor or its usable bytes
//! past its dense trunk, KV, context, scratch, reserves and margin; the
//! host's usable bytes past its tensors, ring shadows and reserves); and
//! `--park-ram` is refused by name: resident slots hold their state in the
//! plan. A `parallel` line on stderr names the rule (`slots`), the slots, a
//! slot's context, the total, what set the count (`flag`, `default`, or
//! one slot under the DSpark draft and under a route trace) and the shape of
//! a round of several slots (`pass=one` with no draft, `pass=turns` under
//! the lookup draft); a `slots` line
//! after the load names each sequence's card and page-locked host bytes.
//! `--queue-depth Q` bounds the requests that wait for a slot. Several slots
//! are refused by name under the DSpark draft (its window is one sequence's,
//! on its own card: a select would start it over) and under
//! `BLOOMERY_ROUTE_TRACE` (the trace records one sequence's steps after its
//! call row); the lookup draft rebuilds its tables from the selected slot's
//! history at that slot's next pass, so its passes are the slot's alone.
//!
//! `--slot-save-path DIR` turns on the slot actions as llama-server's: an
//! erase empties that slot alone; a save writes the slot's sequence state
//! in the body's byte form (`body::save_state`) after the server's header,
//! and a restore reads it back into the slot it names (`body::restore_state`),
//! the drafts then following as after a prompt-cache resume. A file of
//! another model file, card, context or slot count is refused by name: the
//! request's 400, the slot reset, the server serving on.
//!
//! An engine error ends the process: the request gets a 500, `/health` a 503
//! for a moment, then the crash block (card, position, error) goes to stderr
//! and the exit code is 70.
//!
//! `BLOOMERY_ROUTE_TRACE=<dir>` writes a route trace
//! (`bloomery_gpu::host::route_trace`) into `<dir>`, which `main` creates as
//! a new directory before the load: every position the engine runs, each
//! layer's routed ids and the slot each ran in, a `call` row per prompt call.
//! It needs the step feed (`BLOOMERY_PREFILL=steps`) and no draft, and is
//! refused by name otherwise. Its `# build` names the server's version with
//! its commit, `/props`' `engine.version` before the engine's note.
//!
//! The levers it acts on (`serve_levers::ACTS_ON`) are parsed once, at
//! `main` (`bloomery_levers::at_main`), which refuses by name a lever set
//! outside them and a `BLOOMERY_*` name no registry row names; `--levers`
//! prints them with this process's values and exits. A prompt is fed the way
//! the engine holds `BLOOMERY_PREFILL`. The stderr lines named
//! above are records of the kinds `bloomery_gpu_gates::record` declares;
//! `--records-schema` prints those kinds and exits.
//!
//! The model opens as `generate_ds41`'s does (`app::Loaded`, then
//! `Loaded::ready`), and `BLOOMERY_DRAFT=lookup|dspark` serves its draft the
//! same way (`shared/ds41_draft.rs`): the DSpark draft's file is
//! `$BLOOMERY_DSPARK_MODEL` and its card `BLOOMERY_DSPARK_CARD` (a card of
//! the plan, which reserves the draft's bytes there: unset, the tier card
//! under `bp`, else the stage when it is the only card, else the other card as
//! a reserve-only member of the plan),
//! loaded once, its `load draft=dspark` line after the `load`
//! line, the pair pass captured after the step. Every greedy token after a
//! request's first then comes out of a pass (`serve::Engine::advance`), and
//! a request that needs the logits row — `temperature` above 0 (llama-server's
//! default 0.8 included) or `ignore_eos` — takes plain steps, each step's row
//! the target's (the draft writes only its own card's buffers or host tables).
//! A greedy request's `timings` carry `draft_n` and `draft_n_accepted`,
//! `/metrics` their sums,
//! and `/props`' `engine.draft` names the draft (its kind, file and device),
//! whose card's placement row holds its resident bytes as the class `draft`.
//! Both drafts' arms of every one of these calls, the draft word and the
//! slot rule are `served`'s, the serve seats' one owner of the draft; the
//! lookup's context (`drafted::Lookups`, one a slot) is rebuilt from the V4.1
//! body's token history when it falls behind the target.
//!
//! The prompt cache under the DSpark draft: the draft keeps no state of its
//! own worth saving — its window ring is a function of the last
//! `attention.sliding_window` positions' features, which the target's
//! prompt call hands over again. A request that continues exactly where the
//! draft stands keeps every position the target keeps. Any other — a cut, a
//! saved state put back — starts the draft over, so the target keeps no
//! position past the request's shared prefix less a window: the prompt call
//! then carries the window's features to the draft, and the ids are the full
//! prompt's. The lookup draft's context of a slot follows the slot's
//! positions; a saved state put back starts it over from the target's token
//! history at the request's first call.
//!
//! `--mmproj FILE` (llama-server's spelling, with `--no-mmproj` and
//! `--image-min-tokens`/`--image-max-tokens`: [`vision_seat::Flags`]) turns
//! the seat's image input on (behind the `vision` feature — the encoder's
//! device code; a build without it refuses the flag by name, and the seat
//! serves no image): the file is the V4.1 encoder's GGUF, opened at load and
//! refused by name when it is not ([`vision_seat::Projector`]'s checks, and
//! [`vision_seat::Resident::open`]'s), and the encoder is resident from then
//! on — on a card this process holds that the placement's plan leaves off it
//! (the 3090 under `--place a` here, at no cost to the plan), else on the
//! stage card as a named reserve the plan's fit counts
//! ([`vision_seat::EncoderSeat`]); the `vision` record after the load names
//! the card, which of the two it is, and the encoder's bytes. A request's image parts expand
//! to their spans (`serve::media` over the engine's [`Seat::media_model`]),
//! each span encoded on the resident encoder and spliced into the prompt
//! call ([`ds41_media`]'s one assembly with the media gate); the seat's
//! resident slots carry each span's history with their sequences, and the
//! prompt cache keys a kept prefix by its images as well as its ids.
//! Refused by name: `--mmproj` under the step feed (a span's rows enter only
//! through the batched call), and an image under the DSpark draft (its
//! window is fed by the prompt call's features, which a media call hands
//! none).

use std::any::Any;
use std::io::{Read, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use app::arch::deepseek41::Ds41Cfg;
use app::{Loaded, OpenLog, RowsLog, Session, SessionError};
use bloomery_gpu::host::route_trace::{RouteTrace, TraceHeader};
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::model::{Slots, StepMode};
use bloomery_gpu_deepseek41::body::{self, Body, Deepseek41Model, PAIR_ROWS, SeqSnapshot};
use bloomery_gpu_deepseek41::swap;
use bloomery_gpu_gates::beside::{self, Beside};
use bloomery_gpu_gates::bind::{
    CacheRam, ChatSurface, Listen, Pass, Seat, SeatEngine, SlotFlags, SlotStep, SlotsFrom,
    TemplateFile, Vocab, arm_screen, bind_server, model_props, parallel_line, placement_of,
};
use bloomery_gpu_gates::generate::{DraftSeat, Lane, Place, lane_at, mode_name};
use bloomery_gpu_gates::need;
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::residency38::{Seqs, checkpoints_reserved, reserve_checkpoints};
use bloomery_gpu_gates::side_seat;
use bloomery_gpu_gates::vision_seat;
use bloomery_gpu_gates::{GateError, ref_model_path, residency};
use bloomery_levers::{ResidencyAt, ResidencyPick, ResidencyWhy};
use gguf::Split;
use model::arch::deepseek41::hparams::Model as Hp41Model;
use model::arch::deepseek41::place::PlanInputs;
use model::arch::dspark::DraftHparams;
use model::placement::ctx_ask::DEFAULT_SLOTS;
use model::placement::workstation::{HostNeed, TierBatchBytes};
use model::placement::{Machine, Plan, PlanLevers};
use refset::arch::deepseek41::VERIFIED_POSITIONS;
use runtime::width::Mode as WidthMode;
use runtime::{Target, Want};
use serve::dsml::{Spelling, respell};
use serve::flag::{CTX, number};
use serve::{
    CacheNote, Drafted, EngineError, EngineProps, PlacementProps, ResidencyReset, Saved,
    ServeError, StateError,
};
use tokenizer::Tokenizer;

#[cfg(feature = "vision")]
use bloomery_gpu_gates::ds41_media;
#[cfg(feature = "vision")]
use cuda_core::CudaContext;
#[cfg(feature = "vision")]
use serve::media::MediaFeed;
#[cfg(feature = "vision")]
use vision_seat::{Opened, PleImage, TextFacts};

use super::drafted::LOOKUP_ROWS;
use super::served::{self, ServedDraft};
use crate::draft::{Draft, open_dspark};
use crate::{dspark, place};

const USAGE: &str = "usage: bloomery-serve-ds41 [--host H] [--port P] \
                     [--place a|gate|bp|<stage>[+<tier>…]] [--ctx C] [--alias NAME] \
                     [--cache-ram MIB] [--parallel N] [--slot-ctx S] [--queue-depth Q] [--plan] \
                     [--slot-save-path DIR] [--chat-template-file PATH] \
                     [--mmproj FILE | --no-mmproj] [--image-min-tokens N] [--image-max-tokens N] [--api-key KEY] \
                     [--api-key-file FNAME]";

/// What the server says when it serves the chat template (the file's, or
/// `--chat-template-file`'s) with its DSML tags respelled ([`respell`]).
const DSML_RESPELLED: &str = "bloomery-serve: chat template: this V4.1 chat template spells \
    V4's DSML tags, but its model writes V4.1's (a space after the token: `<｜DSML｜ calls>`, \
    `<｜DSML｜ invoke>`, `<｜DSML｜ parameter>`); serving the template respelled that way";

/// What the server says when it serves DeepSeek's official V4.1 prompt
/// encoding, the one the V4.1 release's own encoder writes.
const V41_OFFICIAL: &str = "bloomery-serve: chat template: this V4.1 model file is served with \
    DeepSeek's official V4.1 prompt encoding (the system token, the reasoning effort and the \
    spaced DSML tags the V4.1 release writes), not the file's own template";

/// The token V4.1's chat template opens every user and tool message with.
pub const USER_START: &str = "<｜User｜>";

/// A prompt call is cut at a message start only this far past the call's
/// start: below it the cut's second call costs more than the prefix a
/// later request keeps saves.
const SPLIT_MIN: usize = 64;

pub(crate) struct Args {
    host: String,
    port: u16,
    /// The placement the seat runs by, resolved against this process's
    /// devices by the common unset rule ([`place::choose`]).
    place: Place,
    /// `--ctx`: the total the slots split; `None` takes the seats' default
    /// choice (`place::ctx_choice`).
    ctx: Option<usize>,
    alias: Option<String>,
    /// `--cache-ram` in bytes; `None` takes the default.
    cache_ram: Option<u64>,
    /// `--parallel`: the resident sequences the seat serves; `None` takes
    /// [`slot_count`]'s default.
    parallel: Option<usize>,
    queue_depth: Option<usize>,
    /// `--slot-save-path`: the directory the slot actions answer from;
    /// `None` refuses every one, as llama-server does.
    slot_save_path: Option<PathBuf>,
    /// `--chat-template-file`, replacing the file's own template; the respell
    /// ([`respell`]) reads it as it reads the file's.
    template_file: TemplateFile,
    /// The image-input flags: `--mmproj` is the V4.1 encoder file whose
    /// images the seat serves, none serves no image.
    vision: vision_seat::Flags,
    /// `--api-key`/`--api-key-file`: the keys every request is checked
    /// against ([`serve::flag::ApiKeys`]).
    api_keys: serve::flag::ApiKeys,
    /// `--plan`: the lines before the load (`ctx`, `parallel`), then exit;
    /// nothing is opened on a card.
    plan_only: bool,
}

/// The flags of `args`, and `--place`'s word: the placement is chosen against the devices
/// ([`place::choose`]) by [`run`], since the flags are checked before any card is read
/// ([`check`]).
fn parse_args(args: &[String]) -> Result<(Args, Option<Place>), GateError> {
    let mut a = Args {
        host: "127.0.0.1".to_owned(),
        port: 8080,
        place: Place::A,
        ctx: None,
        alias: None,
        cache_ram: None,
        parallel: None,
        queue_depth: None,
        slot_save_path: None,
        template_file: TemplateFile::default(),
        vision: vision_seat::Flags::default(),
        api_keys: serve::flag::ApiKeys::default(),
        plan_only: false,
    };
    let mut park_ram = false;
    let mut place_flag = None;
    let mut slots = SlotFlags::default();
    let mut it = args.iter().map(|s| s.as_str());
    while let Some(flag) = it.next() {
        if flag == "--help" || flag == "-h" {
            return Err(USAGE.into());
        }
        if flag == "--plan" {
            a.plan_only = true;
            continue;
        }
        if a.vision.take(flag, &mut it)? {
            continue;
        }
        let v = it
            .next()
            .ok_or_else(|| format!("{flag} needs a value, or is unknown: {USAGE}"))?;
        match flag {
            "--host" => a.host = v.to_owned(),
            "--port" => a.port = number(flag, v)?,
            "--place" => place_flag = Some(place::parse(v)?),
            f if CTX.contains(&f) => a.ctx = Some(number(flag, v)?),
            "--alias" => a.alias = Some(v.to_owned()),
            "--cache-ram" => a.cache_ram = Some(mib_bytes(flag, v)?),
            "--queue-depth" => a.queue_depth = Some(number(flag, v)?),
            "--slot-save-path" => a.slot_save_path = Some(PathBuf::from(v)),
            TemplateFile::FLAG => a.template_file.set(v),
            f if serve::flag::KEYS.contains(&f) => a.api_keys.add(f, v)?,
            "--park-ram" => {
                mib_bytes(flag, v)?;
                park_ram = true;
            }
            f if slots.take(f, v)? => {}
            other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
        }
    }
    (a.ctx, a.parallel) = slots.given(a.ctx)?;
    if park_ram {
        return Err(
            "--park-ram holds the states of slots that take the model in turns; this seat's \
             slots are resident sequences, which park nothing — the plan counts their state"
                .into(),
        );
    }
    Ok((a, place_flag))
}

/// The levers (unless `read_levers` is false) and flags [`run`] reads first, checked before the
/// model is fetched ([`bloomery_gpu_gates::bind::check_seat`]), and `--ctx` against the fewest
/// slots the start can serve ([`slot_ctx`]): `--parallel`'s count, else one, the fewest an unset
/// count comes to under any lever [`slot_count`] reads.
pub(crate) fn check(
    args: &[String],
    read_levers: bool,
    from_repo: bool,
) -> Result<bloomery_gpu_gates::bind::CheckedSeat<(Args, Option<Place>)>, GateError> {
    bloomery_gpu_gates::bind::check_seat(
        read_levers.then_some(crate::serve_levers::ACTS_ON),
        args,
        |args, levers| {
            let (a, place) = parse_args(args)?;
            a.template_file.check()?;
            a.vision.check_before_model(from_repo)?;
            let fewest = a
                .parallel
                .and_then(NonZeroUsize::new)
                .unwrap_or(NonZeroUsize::MIN);
            slot_ctx(a.ctx, fewest)?;
            if let Some(levers) = levers {
                let draft = Draft::check_levers(levers)?;
                WidthMode::of(levers.mtp_width())?;
                bloomery_gpu_gates::bind::refuse_plain_width(
                    levers,
                    (draft == Draft::Off).then_some("BLOOMERY_DRAFT is unset"),
                )?;
                let (slots, _) = slot_count(a.parallel, draft, levers.route_trace().is_some())?;
                slot_ctx(a.ctx, slots)?;
                let at = ResidencyAt {
                    serving_place: place != Some(Place::Gate),
                    check_finite: false,
                    route_trace: levers.route_trace().is_some(),
                    prefill_steps: body::PrefillMode::from_name(levers.prefill())
                        == Some(body::PrefillMode::Steps),
                };
                let cfg = body::OpenCfg::from_levers_at(levers, at)?;
                let residency = levers.residency_at(at);
                if cfg.body.residency != Residency::Off && place == Some(Place::Gate) {
                    return Err(format!("BLOOMERY_RESIDENCY={} under --place gate: the residency machine runs under \
                                   --place a and bp only", residency.word).into());
                }
                check_trace(levers, &cfg, residency, (draft, slots))?;
                if a.vision.mmproj()?.is_some() {
                    refuse_media_prefill(cfg.body.prefill)?;
                }
            }
            Ok((a, place))
        },
    )
}

/// A flag's value in MiB, as bytes.
fn mib_bytes(flag: &str, v: &str) -> Result<u64, GateError> {
    let mib: u64 = number(flag, v)?;
    Ok(mib
        .checked_mul(1 << 20)
        .ok_or_else(|| format!("{flag} {mib} MiB passes u64 bytes"))?)
}

/// The resident sequences the seat serves and what set the count:
/// `parallel` (`--parallel`) as given, else `ctx_ask::DEFAULT_SLOTS` — one under the DSpark
/// draft, whose window is one sequence's on its own card
/// ([`served::draft_slots`], which refuses several under it by name), and one
/// under a route trace (`traced`), which records one sequence's steps; under a
/// route trace, [`route_trace`] refuses several.
fn slot_count(
    parallel: Option<usize>,
    draft: Draft,
    traced: bool,
) -> Result<(NonZeroUsize, SlotsFrom), GateError> {
    let (n, from) = match served::draft_slots(parallel, draft, true)? {
        Some(decided) => decided,
        None => match parallel {
            Some(n) => (n, SlotsFrom::Flag),
            None if traced => (1, SlotsFrom::RouteTrace),
            None => (DEFAULT_SLOTS, SlotsFrom::Default),
        },
    };
    let n = NonZeroUsize::new(n).ok_or("--parallel 0: the server serves no slot")?;
    Ok((n, from))
}

/// A set `--ctx` `total`'s share a slot of `slots` (llama-server's `-np`
/// without `-kvu`, `slots::split_ctx`), refused by name below
/// [`PAIR_ROWS`], the pair pass's rows: the most one call of a slot runs on a
/// capture. Unset, nothing to check before the choice.
fn slot_ctx(total: Option<usize>, slots: NonZeroUsize) -> Result<(), GateError> {
    let Some(total) = total else {
        return Ok(());
    };
    let share = total / slots;
    if share < PAIR_ROWS {
        return Err(format!(
            "--ctx {total}: each of the {slots} slots serves {share} positions, below the \
             {PAIR_ROWS} a slot serves at least (the pair pass's rows)"
        )
        .into());
    }
    Ok(())
}

/// An explicit projector feeds rows only through the batched prompt call.
fn refuse_media_prefill(prefill: body::PrefillMode) -> Result<(), GateError> {
    if prefill != body::PrefillMode::Batch {
        return Err(format!(
            "--mmproj under BLOOMERY_PREFILL={}: a media span's rows enter only through the \
             batched prompt call (BLOOMERY_PREFILL=batch)",
            prefill.name()
        )
        .into());
    }
    Ok(())
}

/// What `--mmproj` opened for the plan ([`vision_seat::open_seat`]): the encoder file's card
/// figure — refused by name when the file is not the V4.1 encoder's — and the seat that figure goes
/// on, with the file's projector held against the text model's embedding width and vocabulary.
/// Refused by name under the step feed: a span's rows enter only through the batched call.
#[cfg(feature = "vision")]
fn vision_of(
    a: &Args,
    prefill: body::PrefillMode,
    n_embd: usize,
    token: &dyn Fn(u32) -> Option<String>,
) -> Result<Option<Opened>, GateError> {
    if a.vision.wanted()?.is_some() {
        refuse_media_prefill(prefill)?;
    }
    let facts = TextFacts {
        n_embd,
        imrope: None,
        ple: PleImage::Absent,
    };
    // The `--hf` repo's projector is taken only where a span's rows can enter.
    let file = a.vision.encoder(prefill == body::PrefillMode::Batch)?;
    vision_seat::open_seat(
        &a.vision,
        file.as_deref(),
        a.place,
        &[vision::arch::deepseek41v::PROJECTOR_TYPE],
        &facts,
        token,
    )
}

/// `--mmproj` on a build that takes no images: the flag is refused by name
/// (the encoder's device code sits behind the `vision` feature).
#[cfg(not(feature = "vision"))]
fn vision_of(
    a: &Args,
    prefill: body::PrefillMode,
    n_embd: usize,
    token: &dyn Fn(u32) -> Option<String>,
) -> Result<(), GateError> {
    let _ = (prefill, n_embd, token);
    a.vision
        .check_before_model(bloomery_gpu_gates::model_file::hf_repo().is_some())
}

/// Loads the model, and the embedding model `side` names beside it, and serves until the listener
/// or the engine fails; `Ok` carries why the server ended.
pub fn run(args: &[String], side: &side_seat::Flags) -> Result<ServeError, GateError> {
    record::at_main_with_flags("bloomery-serve-ds41", record::BLOOMERY_SERVE_DS41, |flag| {
        bloomery_gpu_gates::llama_launch::serve_option(args, flag)
    });
    let ((mut a, place_flag), levers) = check(
        args,
        true,
        bloomery_gpu_gates::model_file::hf_repo().is_some(),
    )?
    .run();
    model::ops::set_host_lanes(levers.host_lanes());

    super::drafted::serves_widths();
    arm_screen(levers.screen());
    let placed = place::choose(place_flag)?;
    a.place = placed.place;
    placed.record().eprint();
    // The side model is read and seated before the plan, which holds its bytes when it sits on
    // the stage card.
    let side = side.read()?;
    let side_seat = side.as_ref().map(|s| s.seat(a.place)).transpose()?;
    let embedding = side_seat.as_ref().and_then(|s| s.reserve);
    // The lane's rate, taken before the plan reads the room (`lane_of`).
    // `--plan` returns before any card is opened, so it probes no lane.
    let lane = if a.plan_only {
        None
    } else {
        Some(lane_at(placed.place)?)
    };
    let at = ResidencyAt {
        serving_place: a.place != Place::Gate,
        check_finite: false,
        route_trace: levers.route_trace().is_some(),
        prefill_steps: body::PrefillMode::from_name(levers.prefill())
            == Some(body::PrefillMode::Steps),
    };
    let residency = levers.residency_at(at);
    let cfg = body::OpenCfg::from_levers_at(&levers, at)?;
    let draft = Draft::from_levers(&levers)?;
    let width = WidthMode::of(levers.mtp_width())?;
    let (slots, from) = slot_count(a.parallel, draft, levers.route_trace().is_some())?;
    // The rounds of several slots, decided once (the open takes it through
    // `SeatArgs`) by what the body runs ([`Pass::of_body`]): one pass of the
    // busy rows with no draft; a drafted round needs a body that verifies
    // several slots' windows together, which this one does not (it keeps a
    // pass's rows whole), so the fallback loop ([`V41::one_pass`]).
    let window = match draft {
        Draft::Off => None,
        Draft::Lookup => Some(LOOKUP_ROWS),
        Draft::Dspark => Some(PAIR_ROWS),
    };
    let pass = Pass::of_body::<Body>(slots.get(), true, window);
    // The draft's file is read before the target's load, which takes a minute.
    let draft_file = match draft {
        Draft::Dspark => Some(dspark::draft_hparams()?),
        Draft::Off | Draft::Lookup => None,
    };
    let path = ref_model_path()?;
    let reserve = match &draft_file {
        Some((d, _)) => Some(dspark::draft_seat(a.place, d, &path)?),
        None => None,
    };
    let vocab = Arc::new(Vocab::new(Tokenizer::from_gguf(&path)?)?.with_user_start(USER_START)?);
    let inv = gguf::inventory_of(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut surface = ChatSurface::read(
        &path,
        inv.value("tokenizer.chat_template"),
        inv.value("general.name"),
        a.template_file.path(),
        "deepseek-v4.1",
    )?;
    drop(inv);

    let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let inputs = PlanInputs::read(&split)?;
    // A V4.1 model file is served with DeepSeek's official V4.1 prompt
    // encoding, which spells V4.1's tags, unless the client gave a template
    // of its own. A V4 file, and any file with a client template, keeps the
    // file's own template; a V4.1 one is respelled, because a V4.1 model
    // writes the spaced DSML tags whatever its GGUF's template spells (the
    // V4.1 files carry V4's): the template is served spelled as the model
    // writes.
    let official = match inputs.hp.model {
        Hp41Model::Deepseek41 => Some(serve::v41::TEMPLATE),
        Hp41Model::Deepseek4 => None,
    };
    let template = std::mem::take(&mut surface.template);
    let preparation: Option<serve::ChatPreparation> =
        match official.filter(|_| a.template_file.path().is_none()) {
            Some(official) => {
                eprintln!("{V41_OFFICIAL}");
                surface.template = official.to_owned();
                // The official encoding's own request preparation, with it.
                Some(serve::v41::prepare)
            }
            None => {
                surface.template = match inputs.hp.model {
                    Hp41Model::Deepseek41 => match respell(&template, Spelling::Spaced)? {
                        Some(spaced) => {
                            eprintln!("{DSML_RESPELLED}");
                            spaced
                        }
                        None => template,
                    },
                    Hp41Model::Deepseek4 => template,
                };
                None
            }
        };
    let ChatSurface { template, name, .. } = surface;
    let model = model_props(&split, &inputs.model);
    let tier_batch = place::tier_batch(a.place, &inputs.hp);
    // The context the load takes with the encoder's reserve `vision` and the side model's
    // (`place::ctx_choice`): asked once for a projector's cost, once for the load.
    let ctx_of = |vision: Option<u64>| {
        place::ctx_choice(
            &split,
            &inputs,
            a.place,
            (
                reserve,
                tier_batch,
                Beside {
                    image: vision,
                    embedding,
                },
            ),
            (a.ctx, slots),
            &cfg.place,
        )
    };
    // The encoder's file is read before the plan, which reserves its bytes, and held against the
    // text model's embedding width and vocabulary.
    let n_embd = inputs.hp.n_embd;
    let token = |id| vocab.tokenizer().text(id).map(str::to_owned);
    // A projector the `--hf` repo picked stays where the plan gives the text model the same
    // experts with its reserve as without ([`vision_seat::kept`]). Both costs are read at the
    // plain context, so a reserve that only lowers the context does not drop the encoder: the
    // `ctx` line names it. A reserve the fit places at no context, or not at a set `--ctx`, is
    // a refused cost, and the projector goes as before.
    #[cfg(feature = "vision")]
    let vision = vision_seat::kept(
        vision_of(&a, cfg.body.prefill, n_embd, &token)?,
        &a.vision,
        |v| {
            let vision = v.and_then(|v| v.seat.reserve);
            let ctx = ctx_of(None)?.slot_ctx;
            let cost_with = |vision: Option<u64>| -> Result<_, GateError> {
                let mut machine = a.place.machine(reserve, tier_batch)?(inputs.model.layers);
                beside::reserve_on(
                    &mut machine,
                    Beside {
                        image: vision,
                        embedding,
                    },
                );
                reserve_checkpoints(&mut machine, Seqs::none(slots.get()), ctx)?;
                let plan = inputs.plan_with_slots(&machine, ctx, &cfg.place, slots)?;
                Ok(vision_seat::Cost::of(&plan, ctx))
            };
            match cost_with(vision) {
                Err(_) if vision.is_some() && ctx_of(vision).is_ok() => cost_with(None),
                r => r,
            }
        },
    )?;
    #[cfg(feature = "vision")]
    let seated = vision.as_ref();
    #[cfg(not(feature = "vision"))]
    vision_of(&a, cfg.body.prefill, n_embd, &token)?;
    #[cfg(not(feature = "vision"))]
    let seated: Option<&vision_seat::Opened> = None;
    let vision_reserve = seated.and_then(|v| v.seat.reserve);
    let chosen = ctx_of(vision_reserve)?;
    let ctx = usize::try_from(chosen.slot_ctx)?;
    // The encoder stays on. When its reserve lowers the context, the `ctx`
    // line names both ([`bloomery_levers::RungCost`]). The same context adds
    // nothing.
    let mut ctx_line = chosen.line();
    if let Some(bytes) = vision_reserve {
        let plain = ctx_of(None)?;
        let image = bloomery_levers::Reserve::ImageEncoder;
        if let Some(note) =
            bloomery_levers::RungCost::of(image, chosen.slot_ctx, plain.slot_ctx, bytes)
        {
            ctx_line = format!("{ctx_line} ({note})");
        }
    }
    drop(split);
    eprintln!("{ctx_line}");
    parallel_line(slots.get(), ctx, from, pass);
    let plan_args = (
        (reserve, tier_batch, seated, embedding),
        (ctx, slots),
        &cfg.place,
    );
    if a.plan_only {
        // The plan's records and its need are out, as the load would print
        // them; nothing was opened on a card.
        print_plan(
            &inputs,
            a.place,
            plan_args.0,
            plan_args.1,
            plan_args.2,
            residency,
            a.cache_ram,
        )?;
        bloomery_gpu_gates::start::plan_exit();
    }
    let trace = route_trace(&levers, &cfg, a.place, &path, &inputs)?;
    let (card, placement, cache, residency) = print_plan(
        &inputs,
        a.place,
        plan_args.0,
        plan_args.1,
        plan_args.2,
        residency,
        a.cache_ram,
    )?;
    let cfg = body::OpenCfg::from_levers_with(&levers, Residency::parse(residency.word)?)?;
    // The record's `headroom` is the room the budget came from: what
    // `MemAvailable` leaves past the plan's host need and the churn pool.
    Record::new(&record::CACHE_CONFIG)
        .u("ram", cache.ram)
        .u(
            "headroom",
            i128::from(cache.available) - i128::from(cache.need) - i128::from(cache.pool),
        )
        .w("user_start", USER_START)
        .w("in_template", template.contains(USER_START))
        .eprint();
    let props = EngineProps {
        model: Some(model),
        placement,
        ctx_verified: Some(VERIFIED_POSITIONS),
        ..EngineProps::default()
    };
    let open = SeatArgs {
        place: a.place,
        lane: lane.expect("a serving run probed the lane before the plan"),
        ctx,
        slots,
        pass,
        cfg,
        pin_main: levers.pin_main(),
        want_top_k: inputs.hp.indexer.top_k,
        n_layer: inputs.hp.n_layer,
        path: path.clone(),
        draft,
        draft_file,
        reserve,
        tier_batch,
        trace,
        stats: levers.step_stats(),
        width,
        n_embd: inputs.hp.n_embd,
        #[cfg(feature = "vision")]
        vision,
        embedding,
    };
    let engine = SeatEngine::spawn(move || V41::open(open), ctx, vocab, card, props, cache.ram)?;
    let side = side_seat::Side::start(side.zip(side_seat))?;
    // The seat's resident slots are the server's, one sequence each: the
    // server selects and steps them together, no turns and no park.
    let server = bind_server(
        engine,
        Listen {
            host: &a.host,
            port: a.port,
            alias: a.alias.unwrap_or(name),
            path: &path,
            template,
            preparation,
            slot_save_path: a.slot_save_path,
            api_keys: a.api_keys,
            parallel: slots.get(),
            queue_depth: a.queue_depth,
            schema_jump: levers.schema_jump(),
            side,
        },
    )?;
    Record::new(&record::LISTENING)
        .w("place", a.place.name())
        .u("ctx", ctx)
        .w("addr", server.local_addr()?)
        .eprint();
    Ok(server.run())
}

/// The model-independent restrictions of a route trace.
fn check_trace(
    levers: &bloomery_levers::Levers,
    cfg: &body::OpenCfg,
    residency: ResidencyPick,
    (draft, slots): (Draft, NonZeroUsize),
) -> Result<(), GateError> {
    if levers.route_trace().is_none() {
        return Ok(());
    }
    if cfg.body.prefill != body::PrefillMode::Steps {
        return Err(
            "BLOOMERY_ROUTE_TRACE records the step feed: set BLOOMERY_PREFILL=steps \
                    (the batched prompt call routes a CED layer only at the positions a later \
                    reader needs)"
                .into(),
        );
    }
    if draft != Draft::Off {
        return Err(format!(
            "BLOOMERY_ROUTE_TRACE records one-row steps; BLOOMERY_DRAFT={} runs two-row passes",
            draft.name()
        )
        .into());
    }
    if slots.get() > 1 {
        return Err(format!(
            "BLOOMERY_ROUTE_TRACE records one sequence's steps after its call row; \
             --parallel {slots} puts another sequence's steps between them"
        )
        .into());
    }
    // The trace is the input the residency model replays under a fixed
    // seed; under the machine its slot files would record the machine's
    // own moves, its header would not say so, and the step feed would
    // count each prompt id as a kept row where the batched call keeps 0.
    if cfg.body.residency != Residency::Off {
        return Err(format!(
            "BLOOMERY_ROUTE_TRACE records a fixed placement's routing; \
             BLOOMERY_RESIDENCY={} moves the slot map under it",
            residency.word
        )
        .into());
    }
    Ok(())
}

/// The route trace `BLOOMERY_ROUTE_TRACE` asks for, its directory made
/// now, before the load, after `check` validated its feed, draft and slot count.
fn route_trace(
    levers: &bloomery_levers::Levers,
    cfg: &body::OpenCfg,
    place: Place,
    path: &Path,
    inputs: &PlanInputs,
) -> Result<Option<RouteTrace>, GateError> {
    let Some(dir) = levers.route_trace() else {
        return Ok(None);
    };
    let hp = &inputs.hp;
    let yes = |b: bool| if b { "on" } else { "off" }.to_owned();
    let header = TraceHeader {
        model: path.to_path_buf(),
        arch: "deepseek41".to_owned(),
        build: format!("bloomery-serve-ds41 {}", serve::VERSION),
        n_expert: hp.experts.n_expert,
        n_used: hp.experts.n_used,
        first_layer: 0,
        n_layer: hp.n_layer,
        extra: vec![
            ("placement".to_owned(), place.name().to_owned()),
            (
                "card_budget".to_owned(),
                match levers.card_budget() {
                    bloomery_levers::CardBudget::Unset => "each card's own".to_owned(),
                    b => b.to_string(),
                },
            ),
            ("r8".to_owned(), yes(cfg.body.host.r8)),
            ("prefill".to_owned(), cfg.body.prefill.name().to_owned()),
        ],
    };
    Ok(Some(RouteTrace::create(dir, header)?))
}

/// The plan the engine is about to load under the placement's `levers`
/// (with `reserve`, the DSpark draft's, and `tier_batch`, the tier's
/// prompt-batch bytes, on its tier card, and `vision`, the image encoder
/// `--mmproj` seated: its bytes are a reserve of the stage card when no held
/// card sits off the plan, and its `vision_seat` record follows the plan's),
/// `slots` resident sequences of `ctx` positions each
/// (`PlanInputs::plan_with_slots`), on stderr. The
/// unset residency word resolves against that plan first
/// ([`residency::at_plan`]), its `residency lever` record — the plan-side
/// detail on the line beside it when the plan moved or refused the word —
/// before the plan's own record; the encoder's `vision_seat` record follows
/// the plan's, then under the word it resolved to the churn pool's, a set
/// word's pool refused by name when it does not fit. Returns its cards'
/// names, the plan's placement for `/props` (`None`, and a line saying why,
/// when a card's nvidia-smi index cannot be found), the prompt cache's
/// budget of the real host's room
/// ([`CacheRam::of`]) and the residency word the load runs by. The `need`
/// record follows the prompt cache's budget it states. A plan the
/// vision reserve or the side model's `embedding` reserve does not fit is the
/// plan's own refusal with the reserves named before its terms.
fn print_plan(
    inputs: &PlanInputs,
    place: Place,
    (reserve, tier_batch, vision, embedding): (
        Option<DraftSeat>,
        Option<TierBatchBytes>,
        Option<&vision_seat::Opened>,
        Option<u64>,
    ),
    (ctx, slots): (usize, NonZeroUsize),
    levers: &PlanLevers,
    pick: ResidencyPick,
    cache_ram: Option<u64>,
) -> Result<(String, Option<PlacementProps>, CacheRam, ResidencyPick), GateError> {
    let mut machine = place.machine(reserve, tier_batch)?(inputs.model.layers);
    let held = Beside {
        image: vision.and_then(|v| v.seat.reserve),
        embedding,
    };
    beside::reserve_on(&mut machine, held);
    // V4.1's body makes no checkpoints: its slots pin none.
    reserve_checkpoints(&mut machine, Seqs::none(slots.get()), u64::try_from(ctx)?)?;
    let plan = inputs
        .plan_with_slots(&machine, u64::try_from(ctx)?, levers, slots)
        .map_err(|e| beside::refusal(held, e))?;
    let pick = residency::at_plan(
        <Body as bloomery_gpu::model::ChainBody>::UNHOSTED_POOL,
        &plan,
        0,
        pick,
    )?;
    record::residency_lever(pick).eprint();
    if let Some(d) = pick.why.detail() {
        eprintln!("{d}");
    }
    record::plan(place.name(), &machine, &plan).eprint();
    if let Some(v) = vision {
        vision_seat::seat_record(v).eprint();
    }
    let mut pool_bytes = 0;
    if let Some(pool) = swap::churn(&plan, 0, Residency::parse(pick.word)?)? {
        record::residency_host(pick.word, &pool, &plan).eprint();
        // A set word is refused by name when its pool does not fit; the
        // unset word was resolved against this plan, which fitted it.
        if pick.why == ResidencyWhy::Set {
            pool.check(&plan)?;
        }
        pool_bytes = pool.bytes;
    }
    let cache = CacheRam::of(
        cache_ram,
        HostNeed::of(&plan, 0).bytes(),
        pool_bytes,
        checkpoints_reserved(plan.machine),
    )?;
    need::print(&plan, pool_bytes, cache.ram, true)?;
    let placement = placement_of("bloomery-serve-ds41", &machine, &plan);
    let cards: Vec<&str> = machine.all_cards().map(|c| c.name.as_str()).collect();
    Ok((cards.join("+"), placement, cache, pick))
}

const WHAT: &str = "bloomery-serve-ds41";

/// What the engine thread opens the seat with.
struct SeatArgs {
    place: Place,
    /// The lane's rate probed before the plan, and the card it ran on: the
    /// `lane` line beside the `load` line.
    lane: Lane,
    /// A slot's context: the positions each resident sequence serves.
    ctx: usize,
    /// The resident sequences the plan counts and the load makes.
    slots: NonZeroUsize,
    /// The shape of a round of several slots, which the `parallel` line
    /// named and the open runs.
    pass: Pass,
    cfg: body::OpenCfg,
    pin_main: bool,
    /// The file's `top_k` and layer count, from its headers.
    want_top_k: usize,
    n_layer: usize,
    path: PathBuf,
    draft: Draft,
    draft_file: Option<(Split, DraftHparams)>,
    /// The draft's reserve on its card of the plan.
    reserve: Option<DraftSeat>,
    /// The tier's prompt-batch bytes on the placement's tier card.
    tier_batch: Option<TierBatchBytes>,
    /// The route trace, attached once the session is ready.
    trace: Option<RouteTrace>,
    /// [`Seat::step_stats`]: the `BLOOMERY_STEP_STATS` the binary parsed.
    stats: bool,
    /// `BLOOMERY_MTP_WIDTH`: the width a drafted window verifies, the
    /// chooser's (`cost`) or the draft's own (`fixed`).
    width: WidthMode,
    /// The model file's embedding width: the width of a media span's row.
    n_embd: usize,
    /// The encoder's seat and image input under `--mmproj` ([`vision_of`]'s
    /// pick); `None` without the flag.
    #[cfg(feature = "vision")]
    vision: Option<Opened>,
    /// The side model's reserve on the stage card, which the plan counted.
    embedding: Option<u64>,
}

/// The selected slot's token history ([`Body::history`]), which the lookup
/// draft's context is rebuilt from when it falls behind the target.
fn history(s: &Session<Body>) -> Option<&[u32]> {
    s.model().body(WHAT).ok().map(Body::history)
}

/// The V4.1 session on the engine thread, and its draft.
struct V41 {
    s: Session<Body>,
    /// The positions served (`--ctx`).
    ctx: usize,
    /// The draft the seat serves ([`ServedDraft`], the serve seats' one
    /// owner of it), verified by the pair pass behind the width chooser
    /// `BLOOMERY_MTP_WIDTH` names (its width 1: the chooser's cut of 0 is the
    /// gate, the pair pass against the plain step).
    draft: ServedDraft<Body>,
    /// Whether a round of several slots runs as one pass of the busy rows
    /// ([`V41::step_slots`]): a load of more than one resident slot whose
    /// shape the open decided by what the body runs ([`Pass::of_body`]):
    /// with no draft. Every other load keeps the
    /// fallback loop: one slot (a round of several never runs), the DSpark
    /// draft (which serves one slot), and the lookup draft, whose windows
    /// this body cannot verify together (it keeps a pass's rows whole).
    one_pass: bool,
    /// [`Seat::step_stats`]: the `BLOOMERY_STEP_STATS` the binary parsed.
    stats: bool,
    /// The resident image encoder and its splice halves (`--mmproj`).
    #[cfg(feature = "vision")]
    vision: Option<Vision>,
}

/// The seat's resident image encoder ([`vision_seat::Resident`], which holds the
/// media description the engine answers, [`Seat::media_model`]) and the delimiter
/// rows of its file that its spans splice around the aligner's. Behind the
/// `vision` feature.
#[cfg(feature = "vision")]
struct Vision {
    resident: vision_seat::Resident,
    delims: ds41_media::Delims,
}

#[cfg(feature = "vision")]
impl Vision {
    /// Step (e) of the seat's media call ([`vision_seat::Resident::encode_with`]): every feed's
    /// image encoded on the resident encoder, its aligner rows read back as bf16 and spliced
    /// ([`ds41_media::splice`]: the delimiter rows around the aligner's, the kind of each
    /// position), all in one pair of buffers, with `model`'s card bound back after the tower. The
    /// aligner emits the plan's token grid, one row a cell.
    fn encode(
        &mut self,
        model: &CudaContext,
        feeds: &[MediaFeed],
    ) -> Result<vision_seat::Encoded<body::MediaKind>, GateError> {
        let (n_embd, delims) = (self.resident.n_embd(), &self.delims);
        self.resident.encode_with(
            model,
            feeds
                .iter()
                .map(|f| vision_seat::Image::of(f.at, &f.prepared)),
            ds41_media::splice(n_embd, delims),
        )
    }
}

/// A body's saved sequence state, as the server's prompt cache holds it,
/// and the positions a request after its resume leaves the DSpark draft
/// to feed (0 without it).
struct Ds41Saved {
    state: SeqSnapshot,
    window: usize,
}

impl Saved for Ds41Saved {
    fn n_tokens(&self) -> usize {
        self.state.positions()
    }

    fn n_bytes(&self) -> u64 {
        self.state.bytes() as u64
    }

    /// A resume starts the DSpark draft over: a window less.
    fn keepable(&self, n: usize) -> usize {
        self.state.keep_point(n.saturating_sub(self.window))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The resident image encoder of `--mmproj` ([`Vision`]), opened on the seat
/// the plan picked ([`vision_seat::EncoderSeat`]): the tower loaded on its card
/// ([`vision_seat::Resident::open`]) — refused by name when the file is not the V4.1
/// encoder's, or when its uploads are not the card figure the plan reserved — the
/// delimiter rows of its file read once, and the prompt batch's media half
/// made ([`body::prepare_media`]). The `vision` record that follows the load
/// names the card, which of the two seats it is, and the encoder's bytes.
#[cfg(feature = "vision")]
fn open_encoder(s: &mut Session<Body>, a: &SeatArgs) -> Result<Option<Vision>, GateError> {
    let Some(v) = &a.vision else {
        return Ok(None);
    };
    let t = Instant::now();
    let resident = vision_seat::Resident::open(s.model().gpu().context(), v)?;
    let file =
        gguf::Gguf::open(&v.path).map_err(|e| format!("--mmproj {}: {e}", v.path.display()))?;
    let delims = ds41_media::delims(&file, a.n_embd)?;
    // The prompt batch's media half, beside the batch buffers the session's
    // own ready made.
    body::prepare_media(s.model_mut())?;
    resident.announce(v, vision_seat::ScratchHome::Own, t);
    Ok(Some(Vision { resident, delims }))
}

impl V41 {
    /// The session by `a.place` (the `load`, host set and `capture`
    /// lines), the draft `a.draft` names on it (its `load draft=dspark`
    /// line before the capture, the pair pass's capture after it), on
    /// the calling thread, pinned to the dispatcher's cpu slot when asked.
    fn open(mut a: SeatArgs) -> Result<V41, GateError> {
        let trace = a.trace.take();
        let pinned = a.pin_main && threads::pool().pin_caller();
        let t = Instant::now();
        let file = Split::open(&a.path).map_err(|e| format!("open {}: {e}", a.path.display()))?;
        // The plan counts every resident sequence, which the session's own
        // open (one sequence) does not: the model is opened here by that
        // plan, then handed to the session. The encoder's reserve rides the
        // same plan when it sits on the stage card, and so does the side model's ([`print_plan`]'s
        // rule).
        #[cfg(feature = "vision")]
        let vision_reserve = a.vision.as_ref().and_then(|v| v.seat.reserve);
        #[cfg(not(feature = "vision"))]
        let vision_reserve = None;
        let machine = beside::with_reserve(
            a.place.machine(a.reserve, a.tier_batch)?,
            Beside {
                image: vision_reserve,
                embedding: a.embedding,
            },
        );
        let opened = t.elapsed();
        let mut m = body::open_slots(file, machine, a.ctx, &a.cfg, a.slots)?;
        m.set_mode(StepMode::Graph);
        let mut log = Log {
            a: &a,
            pinned,
            t,
            opened,
        };
        log.load(&m)?;
        let cfg = Ds41Cfg {
            feed: a.cfg.body.prefill,
            open: a.cfg.clone(),
            card_timing: false,
        };
        let mut loaded = Loaded::<Body>::from_model(m, cfg, u32::try_from(a.ctx)?);
        let spark = match a.draft_file.as_ref().zip(a.reserve) {
            Some((file, reserve)) => {
                let (d, load) = open_dspark(&mut loaded, file, &a.path, WHAT, a.place, reserve)?;
                load.eprint();
                Some(d)
            }
            None => None,
        };
        let mut s = loaded.ready(&mut log)?;
        #[cfg(feature = "vision")]
        let vision = open_encoder(&mut s, &a)?;
        if let Some(t) = trace {
            s.model_mut()
                .body_parts(WHAT)?
                .2
                .hybrid_mut()
                .attach_route_trace(t)?;
        }
        if a.cfg.body.residency != Residency::Off {
            // A request's passes are not known at load, and no request
            // is a timed window: the log grows as it must.
            s.model_mut().body_parts(WHAT)?.2.log_residency(0);
        }
        let draft = match (a.draft, spark) {
            (Draft::Off, _) => ServedDraft::off(),
            (Draft::Lookup, _) => {
                ServedDraft::lookup(&mut s, a.slots.get(), a.width, &mut log, Some(history))?
            }
            (Draft::Dspark, Some(d)) => ServedDraft::dspark::<_, PAIR_ROWS>(
                &mut s,
                d,
                a.width,
                &mut log,
                dspark::draft_path().ok(),
            )?,
            (Draft::Dspark, None) => {
                return Err("bloomery-serve-ds41: the DSpark draft did not load".into());
            }
        };
        // The resident sequences, made after the captures: the live slot's
        // chains stay with it, each other slot capturing on its first use.
        s.add_slots(a.slots.get())?;
        let b = s.model().body(WHAT)?;
        eprintln!(
            "slots made={} planned={} seq_bytes={} seq_shadow_bytes={} resident_bytes={}",
            s.slots(),
            b.slots_planned(),
            b.seq_bytes(),
            b.seq_shadow_bytes(),
            s.model().resident_bytes()
        );
        Ok(V41 {
            s,
            ctx: a.ctx,
            draft,
            one_pass: a.pass.together(a.slots.get()),
            stats: a.stats,
            #[cfg(feature = "vision")]
            vision,
        })
    }

    fn model(&self) -> &Deepseek41Model {
        self.s.model()
    }

    /// The DSpark draft's window: the positions a request that starts
    /// the draft over feeds it before its first proposal.
    fn window(&self) -> usize {
        self.draft.window()
    }

    /// The `residency pass` records of the boundaries the last call made,
    /// on stderr.
    fn print_passes(&mut self) -> Result<(), GateError> {
        for (kind, r) in self
            .s
            .model_mut()
            .body_parts(WHAT)?
            .2
            .take_residency_passes()
        {
            record::residency_pass_of(kind, &r).eprint();
        }
        Ok(())
    }

    /// The selected slot's sequence was put back from a saved state: the
    /// DSpark draft starts over, the lookup rebuilds the slot's context from
    /// the body's history at its next call ([`ServedDraft::replaced`]).
    fn replaced(&mut self) -> Result<(), GateError> {
        self.draft.replaced(self.s.selected())
    }
}

impl Seat for V41 {
    fn pos(&self) -> usize {
        self.s.pos() as usize
    }

    fn ctx_max(&self) -> usize {
        self.ctx
    }

    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        let next = self.draft.prefill(&mut self.s, ids, |s| {
            Ok(s.prompt(ids, Want::Argmax)?.argmax())
        })?;
        self.print_passes()?;
        Ok(next)
    }

    /// The resident encoder's media description ([`vision_seat::Resident::media`]); `None`
    /// without `--mmproj`, an engine that takes no image.
    #[cfg(feature = "vision")]
    fn media_model(&self) -> Option<serve::media::SharedMediaModel> {
        self.vision.as_ref().map(|v| v.resident.media())
    }

    /// Each feed's image encoded on the resident encoder, its aligner rows
    /// read back as bf16 and spliced — the one assembly [`ds41_media`] gives
    /// the media gate — ([`Vision::encode`]), then one prompt call over the
    /// spans of those rows, each with its positions' kinds
    /// ([`vision_seat::Encoded::prompt`], the body's `prefill_media`); the
    /// encode binds the model's card back before the prompt call.
    /// The lookup draft takes the call's ids into the slot's context, as
    /// after any prompt call; the DSpark draft is refused by name (its window
    /// is fed by the prompt call's features, which a media call hands none)
    /// ([`ServedDraft::prefill_media`]).
    #[cfg(feature = "vision")]
    fn prefill_media(&mut self, ids: &[u32], feeds: &[MediaFeed]) -> Result<u32, GateError> {
        let V41 {
            s, vision, draft, ..
        } = self;
        let Some(v) = vision.as_mut() else {
            return Err("no --mmproj: this seat serves no image".into());
        };
        let next = draft.prefill_media(s, ids, |s| {
            let encoded = v.encode(s.model().gpu().context(), feeds)?;
            // Each span borrows its own ranges of the call's one pair of buffers.
            encoded.prompt(s, ids, ds41_media::spans)
        })?;
        self.print_passes()?;
        Ok(next)
    }

    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        let next = self.s.step(last, Want::Argmax)?.argmax();
        self.print_passes()?;
        self.draft.stepped(&mut self.s, last, next)?;
        Ok(next)
    }

    fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError> {
        Ok(self.model().logits_into(row)?)
    }

    fn pass(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, GateError> {
        let counts = self.draft.pass(&mut self.s, last, out)?;
        let Some(counts) = counts else {
            out.push(self.step(last)?);
            return Ok(Drafted::default());
        };
        self.print_passes()?;
        Ok(counts)
    }

    fn pass_rows(&self) -> usize {
        self.draft.rows()
    }

    /// The resident sequences the load made ([`Session::slots`]): one a
    /// `--parallel 1` load (and under the DSpark draft), the `--parallel`
    /// the seat served past it.
    fn slots(&self) -> usize {
        self.s.slots()
    }

    /// The session's slot ([`Session::select_slot`]): the target stands
    /// on another sequence, so the DSpark draft no longer follows it
    /// ([`ServedDraft::moved`]); the lookup keeps a context a slot.
    /// Selecting the live slot moves nothing.
    fn select(&mut self, slot: usize) -> Result<(), GateError> {
        if slot == self.s.selected() {
            return Ok(());
        }
        self.s.select_slot(slot)?;
        self.draft.moved();
        Ok(())
    }

    /// The lookup's state is a context a slot, rebuilt from the slot's
    /// history when it falls behind ([`ServedDraft::slot_drafts`]): a slot's
    /// drafted passes are the passes it would run alone. The DSpark draft
    /// serves one slot.
    fn slot_drafts(&self) -> bool {
        self.draft.slot_drafts()
    }

    /// The lever the binary parsed ([`V41::stats`]).
    fn step_stats(&self) -> bool {
        self.stats
    }

    /// One round of several slots ([`super::rounds::step_round`]): one pass
    /// of the busy rows while the open decided so ([`V41::one_pass`], through
    /// `serve_seats::rounds::step_rows_one_pass` — cut at the body's two
    /// rows, the rows' answers and lent logits rows from the pass's own
    /// per-row heads, slot 0 left selected as [`Session::step_slots_rounds`]
    /// decides), then the call's `residency pass` records, else the seat's
    /// fallback — a select and a step a row — every round of a load the open
    /// kept on it ([`V41::one_pass`]'s doc). A refusal on either path is
    /// the server's to die on: the open decides once, so a load that cannot
    /// run one pass never tries it at run time.
    fn step_slots(&mut self, rows: &mut [SlotStep]) -> Result<(), String> {
        let one_pass = self.one_pass;
        super::rounds::step_round(
            self,
            one_pass,
            rows,
            |v, rows| super::rounds::step_rows_one_pass(&mut v.s, rows),
            V41::print_passes,
        )
    }

    fn reset(&mut self) -> Result<(), GateError> {
        self.s.reset()?;
        self.draft.reset(self.s.selected())?;
        Ok(())
    }

    fn rollback(&mut self, pos: u32) -> Result<(), GateError> {
        self.draft.moved();
        Ok(self.s.model_mut().rollback(pos)?)
    }

    /// [`Session::residency_reset`], its `residency reset` record on
    /// stderr.
    fn residency_reset(&mut self) -> Result<Option<ResidencyReset>, GateError> {
        let Some(r) = self.s.residency_reset()? else {
            return Ok(None);
        };
        record::residency_reset(&r).eprint();
        let n = |v: usize| v as u64;
        Ok(Some(ResidencyReset {
            cancelled: n(r.cancelled),
            copies: n(r.copies),
            diff: n(r.diff),
            dropped_bytes: r.dropped_bytes,
        }))
    }

    /// The draft's record printed ([`ServedDraft::close`]), then the tier
    /// nodes closed.
    fn close(&mut self) -> Result<(), GateError> {
        self.draft.close();
        Ok(self.s.model_mut().close_nodes()?)
    }

    /// The body's rule; under the DSpark draft, a cut anywhere but where
    /// the draft stands also leaves the request's prompt call a window.
    fn keep(&self, n: usize) -> (usize, Option<String>) {
        let b = match self.model().body(WHAT) {
            Ok(b) => b,
            Err(e) => return (0, Some(e.to_string())),
        };
        let body = |m| {
            let (k, why) = b.keep_why(m);
            (k, why.map(|w| w.to_string()))
        };
        self.draft.keep(n, self.s.pos(), body)
    }

    /// The body's cuts under the batched feed; the step feed leaves no
    /// hole and needs none.
    fn splits(&self, first: usize, end: usize, marks: &[usize]) -> Vec<usize> {
        match self.model().body(WHAT) {
            Ok(b) if b.prefill_mode() == body::PrefillMode::Batch => {
                b.prefill_splits(first, end, marks, SPLIT_MIN)
            }
            _ => Vec::new(),
        }
    }

    fn snapshot_size(&mut self) -> Result<Option<u64>, GateError> {
        Ok(Some(body::snapshot_size(self.s.model_mut())?))
    }

    fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
        let window = self.window();
        Ok(Arc::new(Ds41Saved {
            state: body::snapshot(self.s.model_mut())?,
            window,
        }))
    }

    fn resume(&mut self, state: &dyn Saved) -> Result<(), GateError> {
        let saved = state
            .as_any()
            .downcast_ref::<Ds41Saved>()
            .ok_or("a saved state that is not a V4.1 body's")?;
        self.replaced()?;
        Ok(body::resume(self.s.model_mut(), &saved.state)?)
    }

    /// The selected slot's state in the body's byte form
    /// ([`body::save_state`]).
    fn save_state(&mut self, out: &mut dyn Write) -> Result<u64, StateError> {
        body::save_state(self.s.model_mut(), out).map_err(state_error)
    }

    /// The body's byte form put back into the selected slot
    /// ([`body::restore_state`]); the drafts follow as after a
    /// [`Seat::resume`]: the DSpark draft starts over, the lookup rebuilds
    /// the slot's context from the history at its next call.
    fn restore_state(&mut self, input: &mut dyn Read) -> Result<usize, StateError> {
        self.replaced()
            .map_err(|e| StateError::Engine(EngineError(e.to_string())))?;
        body::restore_state(self.s.model_mut(), input).map_err(state_error)
    }

    /// `engine.draft`, and the draft's resident bytes in its card's
    /// placement row, the class `draft` (a row of its own when the target
    /// has no layer there). The card's index missing leaves the device out.
    fn props(&self, p: EngineProps) -> EngineProps {
        self.draft.props(p)
    }

    fn note(note: &CacheNote) {
        let r = match note {
            CacheNote::Reuse {
                common,
                ask,
                kept,
                held,
                reason,
            } => Record::new(&record::CACHE_REUSE)
                .u("common", common)
                .u("ask", ask)
                .u("kept", kept)
                .u("held", held)
                .w("reason", reason.as_deref().unwrap_or("unstated")),
            CacheNote::Save {
                positions,
                bytes,
                ms,
                entries,
                cache_bytes,
                copied: _,
                off_path,
                fallback,
            } => Record::new(&record::CACHE_SAVE)
                .u("positions", positions)
                .u("bytes", bytes)
                .f("ms", *ms)
                .u("entries", entries)
                .u("cache_bytes", cache_bytes)
                .w("off_path", off_path)
                .w("fallback", fallback.unwrap_or("none")),
            CacheNote::Landed {
                positions,
                bytes,
                calls,
                copy_ms,
                alloc_ms,
                waited_ms,
            } => Record::new(&record::CACHE_LANDED)
                .u("positions", positions)
                .u("bytes", bytes)
                .u("calls", calls)
                .f("copy_ms", *copy_ms)
                .f("alloc_ms", *alloc_ms)
                .f("waited_ms", *waited_ms),
            CacheNote::Load {
                positions,
                common,
                kept,
                slot_kept,
                bytes,
                ms,
                resumed,
                waited_ms,
            } => Record::new(&record::CACHE_LOAD)
                .u("positions", positions)
                .u("common", common)
                .u("kept", kept)
                .u("slot_kept", slot_kept)
                .u("bytes", bytes)
                .f("ms", *ms)
                .u("resumed", resumed)
                .f("waited_ms", *waited_ms),
            CacheNote::Evict {
                positions,
                bytes,
                why,
            } => Record::new(&record::CACHE_EVICT)
                .u("positions", positions)
                .u("bytes", bytes)
                .w("why", why),
            CacheNote::Skip { positions, why } => Record::new(&record::CACHE_SKIP)
                .u("positions", positions)
                .w("why", why),
            CacheNote::Split { first, end, at } => Record::new(&record::PREFILL_SPLIT)
                .u("first", first)
                .u("end", end)
                .csv("at", at),
            // The seat shares no prefix between slots (`Seat::share_keep`).
            CacheNote::Share { .. } => return bloomery_gpu_gates::bind::note_line(WHAT, note),
        };
        r.eprint();
    }
}

/// The body's state failure as the server's kind: a refusal is the
/// request's (a 400 or a 500 that keeps the server), a stream's error is the
/// file's (`Io`), a card failure the model's, which ends it.
fn state_error(e: body::StateFail) -> StateError {
    match e {
        body::StateFail::Refused(e) => StateError::Format(e.to_string()),
        body::StateFail::Io(e) => StateError::Io(e),
        body::StateFail::Card(e) => StateError::Engine(EngineError(e.to_string())),
    }
}

/// What the open prints and checks: the body's selection against the
/// file's, the `load` record and the host set's, the captures.
struct Log<'a> {
    a: &'a SeatArgs,
    pinned: bool,
    t: Instant,
    /// The wall from `t` to the loader's call: the phases record's `open_s`.
    opened: Duration,
}

impl OpenLog<Body> for Log<'_> {
    /// The plan was printed before the engine thread started.
    fn plan(
        &mut self,
        _place: &'static str,
        _inputs: &PlanInputs,
        _machine: &Machine,
        _plan: &Plan<'_>,
    ) -> Result<bool, SessionError> {
        Ok(true)
    }

    fn load(&mut self, m: &Deepseek41Model) -> Result<(), SessionError> {
        let a = self.a;
        let b = m.body(WHAT)?;
        let top_k = b.indexer_top_k();
        let shadow = b.shadow_host();
        if top_k != a.want_top_k {
            return Err(SessionError::Refused(format!(
                "the body selects {top_k} rows per stream, the file's top_k is {}: a step \
                 past that many visible rows would not be the model's",
                a.want_top_k
            )));
        }
        let total = self.t.elapsed();
        let load = Record::new(&record::LOAD_GENERATOR)
            .u("resident_bytes", m.resident_bytes())
            .u("ctx", a.ctx)
            .u("layers", a.n_layer)
            .u("top_k", top_k)
            .w("shadow", "host")
            .u("shadow_bytes", shadow.bytes)
            .u("unified_addressing", shadow.unified_addressing);
        record::lane(a.lane.0, &a.lane.1).eprint();
        place::with_cards(m, a.place, WHAT, load)
            .map_err(|e| SessionError::Refused(e.to_string()))?
            .w("prefill", b.prefill_mode().name())
            .w("mode", mode_name(StepMode::Graph))
            .w("place", a.place.name())
            .w("pin_main", if a.pin_main { "on" } else { "off" })
            .w("pinned", self.pinned)
            .f("load_s", total.as_secs_f64())
            .eprint();
        record::eprint_load_phases(m, total, self.opened);
        if let Some(h) = b.hybrid().residency() {
            for r in record::host_residency(h) {
                r.eprint();
            }
        }
        // The NVMe expert tier's arena as the load was granted it, beside
        // the one its plan reserved; a load with no tier prints nothing.
        if let Some(r) = record::nvtier_of(b.nvme_tier().map(|t| &**t)) {
            r.eprint();
        }
        for h in threads::helper::helpers() {
            record::helper(
                &h.name,
                h.asked,
                h.pinned,
                h.cpus,
                (h.parts.threads, h.parts.from),
            )
            .eprint();
        }
        Ok(())
    }

    fn capture(&mut self, nodes: usize) -> Result<(), SessionError> {
        Record::new(&record::CAPTURE)
            .u("graph_nodes", nodes)
            .eprint();
        Ok(())
    }

    fn prompt_buffers(&mut self, _m: &Deepseek41Model) -> Result<(), SessionError> {
        Ok(())
    }
}

impl RowsLog for Log<'_> {
    fn capture_rows(&mut self, rows: usize, nodes: usize) -> Result<(), SessionError> {
        if rows != PAIR_ROWS {
            return Err(SessionError::Refused(format!(
                "a verify capture of {rows} rows; the pair pass runs {PAIR_ROWS}"
            )));
        }
        Record::new(&record::CAPTURE_PAIR)
            .u("pair_graph_nodes", nodes)
            .eprint();
        Ok(())
    }
}
