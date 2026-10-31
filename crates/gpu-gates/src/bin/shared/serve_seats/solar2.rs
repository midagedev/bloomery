//! `bloomery-serve --model solar2` — the llama-server-compatible HTTP API on
//! the Solar-Open2 engine: a solar-open2 file, its layers on one card, the
//! routed experts the plan puts on the card and the rest on the host tier
//! (`placement::plan_routed` through `PlanInputs::plan`), and on a placement that names an expert
//! tier card (`bp`) the next ids of each card layer on that card, opened as
//! `gate_solar2_e2e` opens it (`app::Loaded`, in graph mode with the step
//! captured).
//!
//!     bloomery-serve --model solar2 [-m PATH | --hf <repo>[:<quant>]]
//!                    [--host 127.0.0.1] [--port 8080] [--ctx C] [--alias NAME]
//!                    [--chat-template-file PATH] [--place W] [--plan]
//!                    [--queue-depth Q] [--slot-save-path DIR]
//!                    [--api-key KEY] [--api-key-file FNAME]
//!
//! The server takes `-m`/`--hf` out before this seat parses (its module
//! doc); `--ctx-size` and `-c` are `--ctx` under llama-server's spellings. The
//! model's first shard gives the vocabulary, `tokenizer.chat_template` the chat
//! template (`--chat-template-file` replaces it) and `general.name` the alias
//! (`--alias` replaces it; `solar-open2` for a file that names none).
//! Sampling is the sampler crate's chain with no repetition penalty,
//! `temperature <= 0` the engine's argmax. Any other flag is refused as an
//! unknown argument.
//!
//! Two facts of the architecture set this seat's shape, and nothing else
//! does. The program runs no verify pass of several rows and reads no draft,
//! so the server feeds the prompt less its last id as one prompt call, then
//! steps the last, as `bloomery-serve --model qwen3` does. And the 36 KDA
//! layers each hold one recurrent state written in place by every position,
//! so the body holds one sequence: `--parallel` above 1 is refused by name, a
//! prompt cache is refused by name, and a request keeps the prefix it shares
//! with what the slot holds only where the sequence stands (the model's
//! checkpoints, `app::Keep for Body`, widen that rule once a prompt call takes
//! them); any shorter prefix re-feeds the prompt from an empty model.
//!
//! The prompt call is fed in batches of up to a unit's columns
//! (`app::Prompt for Body`), the same bits as one decode step an id; the
//! batch's buffers are the plan's prompt reserve
//! (`place::prompt_reserve_bytes`), which the body holds its made bytes to by
//! name at the load, and the units a prompt group past one adds come out of the
//! card's margin.
//!
//! `--place W` is the shared placement word (`generate::Place`): `a` (the
//! largest visible card), `gate`, `bp` (plan (b′): the next-largest card the
//! largest's expert tier), or one card of the census by name or ordinal. Unset,
//! the common rule every serving seat takes decides (`Place::choose_untiered`:
//! no sitting has shown the tier not slower), on one census reading: a `place
//! unset` record names the word `a` and why. A placement that names more
//! expert tier cards than the host tier serves is refused by name before any
//! upload (`bloomery_gpu::host::refuse_tier_count`), and a tier card the
//! census does not show is refused by the placement.
//!
//! `--ctx` is the context the one slot serves (`--ctx 0` is refused). The
//! context comes first and the plan fills the machine around it
//! (`generate::choose_ctx`, every seat's): a set value past the file's
//! trained context stops by name, and one the machine cannot hold stops with
//! the largest it would hold. Unset, the slot asks the trained context, or the
//! largest context under it whose plan holds where the card cannot hold the
//! stores and resident bytes. The plan reads the host room
//! (`BLOOMERY_HOST_ROOM` replaces the reading) and pages the routed experts
//! the room cannot hold on the NVMe expert tier, which the body's host run
//! serves; a room under that tier's floor is refused by name.
//!
//! The records on stderr, in order: `place unset` (word and why), one `ctx`
//! line (`placement::ctx_ask::Chosen::line`), the `residency lever` record
//! (the word and why), the `plan` record, under `mid` the `residency host`
//! record of the churn pool, the `need` record, the `parallel` line
//! (`rule=slots slots=1 slot_ctx=C total=C from=<word> pass=one`); then the
//! `load` record (architecture, resident bytes, the
//! slot's context, layers, the routed experts the plan keeps on the host and
//! on the card, the step graph's nodes, the prompt group and the load's wall),
//! the `nvtier` record (on a load whose plan paged routed experts: the arena's
//! budget, the paged bytes and its counters, again after each prompt call) and,
//! once the port is bound, the `listening` record
//! (`record::BLOOMERY_SERVE_SOLAR2`). `--plan` ends the process after the
//! `parallel` line, nothing loaded. An engine error ends the process with the
//! crash block and exit code 70, as every seat's.
//!
//! The levers this seat acts on ([`ACTS_ON`]) are parsed once, at `main`
//! (`bloomery_levers::at_main`), which refuses by name a lever set outside
//! them: the plan's card budget, the host set's load settings,
//! `BLOOMERY_PREFILL_GROUP` (the batches a prompt group runs layer by layer:
//! unset, the row's default), and `BLOOMERY_PIN_MAIN` with
//! `BLOOMERY_HOST_LANES` (the engine thread pinned to the dispatcher's cpu
//! slot, which gives the host legs the pool's CCD map), and the schema jump
//! (`BLOOMERY_SCHEMAJUMP`).
//!
//! `BLOOMERY_RESIDENCY` is the common machine's lever: set, its word as given
//! (refused by name where the plan's card slots leave it no room); unset, the
//! serving word on every placement but `gate`, which the plan then keeps,
//! moves or turns `off` with why (`bloomery_levers::residency_unset`,
//! `residency::at_plan`). The `residency lever` record comes before the plan's
//! own; under `mid`, the `residency host` record of the churn pool comes after
//! the plan. Under `mid` the experts the card holds are the machine's to move
//! by use.
//!
//! The model writes its reasoning between `<|think:start|>` and
//! `<|think:end|>`, the pair its description names
//! (`models::ReasoningFormat::ThinkSpan`); the server splits the reasoning, runs
//! the reasoning budget and writes a structured-output request's close on
//! them, not on `<think>` (`Server::with_think_tags`).
//!
//! The seat sits behind the `deepseek41` feature beside `solar2`: the server
//! surface it binds through (`bind`, the `serve` and `sampler` crates) is
//! scoped to `deepseek41`; it runs no V4.1 code.
//!
//! The seat itself is the common owner's ([`super::hybrid`]): this module
//! holds what Solar-Open2 forces — the architectures it serves, the plan's
//! inputs, the words its refusals name and the records it prints.

use bloomery_gpu_gates::generate::Place;
use bloomery_gpu_gates::record;
use bloomery_gpu_gates::side_seat;
use gguf::Split;
use model::arch::Arch;
use model::arch::place_hybrid::PlanInputs;
use model::arch::solar2::place;
use model::placement::PlacementError;
use model::placement::workstation::TierBatchBytes;
use serve::ServeError;

use super::hybrid::Family;

/// The binary's name, its records' owner and the `note` line's.
const NAME: &str = "bloomery-serve-solar2";

const USAGE: &str = "usage: bloomery-serve --model solar2 [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C] [--alias NAME] [--chat-template-file PATH] \
                     [--place W] [--plan] [--queue-depth Q] [--slot-save-path DIR] \
                     [--api-key KEY] [--api-key-file FNAME]";

/// The levers this seat acts on: the plan's card budget and the host set's
/// load settings, which the open reads (`PlanLevers::from_levers`,
/// `Levers::host`), the residency machine's word, the batches a prompt group
/// runs, the engine thread's pin, the host legs' lane rule, the start screen
/// and the schema jump.
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::RESIDENCY,
    bloomery_levers::PREFILL_GROUP,
    bloomery_levers::HOST_POPULATE,
    bloomery_levers::HOST_LOCK,
    bloomery_levers::CARD_DONTNEED,
    bloomery_levers::R8,
    bloomery_levers::PIN_MAIN,
    bloomery_levers::HOST_LANES,
    bloomery_levers::SCREEN,
    bloomery_levers::SCHEMAJUMP,
];

/// The Solar family: what the common seat owner needs of it.
pub struct Solar;

impl Family for Solar {
    type Body = bloomery_gpu_solar2::Body;
    type H = model::arch::solar2::hparams::Hparams;

    const NAME: &'static str = self::NAME;
    const USAGE: &'static str = self::USAGE;
    const ACTS_ON: &'static [&'static str] = self::ACTS_ON;
    const ALIAS: &'static str = "solar-open2";
    const FAMILY: &'static str = "solar2";
    const SERVES: &'static str = "solar-open2";

    fn accepts(arch: Arch) -> bool {
        matches!(arch, Arch::SolarOpen2)
    }

    fn read(split: &Split) -> Result<PlanInputs<Self::H>, PlacementError> {
        model::arch::solar2::place::PlanInputs::read(split)
    }

    fn register(args: &[String]) {
        record::at_main_with_flags(NAME, record::BLOOMERY_SERVE_SOLAR2, |flag| {
            bloomery_gpu_gates::llama_launch::serve_option(args, flag)
        });
    }

    fn listening() -> record::Record {
        record::Record::new(&record::LISTENING_SOLAR2)
    }

    fn load_record() -> record::Record {
        record::Record::new(&record::LOAD_SOLAR2)
    }

    /// The tier's prompt-batch bytes of the file's widths ([`place::tier_batch`]), reserved on
    /// the placement's tier card and the host.
    fn tier_batch(
        place: Place,
        inputs: &PlanInputs<Self::H>,
    ) -> Result<Option<TierBatchBytes>, bloomery_gpu_gates::GateError> {
        Ok((!place.tier_cards().is_empty())
            .then(|| place::tier_batch(inputs))
            .transpose()?)
    }
}

/// Loads the model, and the embedding model `side` names beside it, and serves until the listener
/// or the engine fails; `Ok` carries why the server ended.
pub fn run(
    args: &[String],
    side: &side_seat::Flags,
) -> Result<ServeError, bloomery_gpu_gates::GateError> {
    super::hybrid::run::<Solar>(args, side)
}

/// The seat's levers (unless `read_levers` is false) and flags, checked before the model is fetched
/// ([`super::hybrid::check`]).
pub fn check(args: &[String], read_levers: bool) -> Result<(), bloomery_gpu_gates::GateError> {
    super::hybrid::check::<Solar>(args, read_levers)
}
