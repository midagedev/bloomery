//! `bloomery-serve --model minimaxm3` — the llama-server-compatible HTTP API on
//! the MiniMax-M3 engine: a minimax-m3 file, its layers on one card, the
//! routed experts the plan puts on the card and the rest on the host tier
//! (`placement::plan_routed` through `PlanInputs::plan`), opened as
//! `gate_minimaxm3_e2e` opens it (`app::Loaded`, in graph mode with the step
//! captured).
//!
//!     bloomery-serve --model minimaxm3 [-m PATH | --hf <repo>[:<quant>]]
//!                    [--host 127.0.0.1] [--port 8080] [--ctx C] [--alias NAME]
//!                    [--chat-template-file PATH] [--place W] [--plan]
//!                    [--queue-depth Q] [--api-key KEY] [--api-key-file FNAME]
//!
//! The server takes `-m`/`--hf` out before this seat parses (its module
//! doc); `--ctx-size` and `-c` are `--ctx` under llama-server's spellings. The
//! model's first shard gives the vocabulary, `tokenizer.chat_template` the chat
//! template (`--chat-template-file` replaces it) and `general.name` the alias
//! (`--alias` replaces it; `minimax-m3` for a file that names none).
//! Sampling is the sampler crate's chain with no repetition penalty,
//! `temperature <= 0` the engine's argmax. Any other flag is refused as an
//! unknown argument.
//!
//! The body holds one sequence: every store is a plane of positions a later
//! position never reads before it writes, so the program runs no verify pass of
//! several rows and reads no draft — the server feeds the prompt less its last
//! id as one prompt call, then steps the last, as `bloomery-serve --model
//! qwen3` does. `--parallel` above 1 is refused by name, a prompt cache is
//! refused by name, and a request keeps the prefix it shares with what the
//! slot holds only where the sequence stands; any shorter prefix re-feeds the
//! prompt from an empty model.
//!
//! The prompt call is fed in batches of up to a unit's columns
//! (`app::Prompt for Body`), the same bits as one decode step an id; the
//! batch's buffers come out of the card's free bytes at the load, refused by
//! name when they do not fit.
//!
//! `--place W` is the shared placement word (`generate::Place`): `a` (the
//! largest visible card), `gate`, or one card of the census by name or
//! ordinal. Unset, the common rule every serving seat takes decides
//! (`Place::choose_untiered`: the body serves no tier card), on one census
//! reading: a `place unset` record names the word `a` and why. A placement
//! that hangs an expert tier card (`bp`, a two-card list) is refused before
//! any load by the plan's own refusal (`PlaceError::Tier`), which names the
//! card and its tier.
//!
//! `--ctx` is the context the one slot serves (`--ctx 0` is refused). The
//! context comes first and the plan fills the machine around it
//! (`generate::choose_ctx`, every seat's): a set value past the file's
//! trained context stops by name, and one the machine cannot hold stops with
//! the largest it would hold. Unset, the slot asks the trained context or
//! 262,144 positions where that is less, halved while the card cannot hold
//! the stores and resident bytes, or while the host cannot hold the routed
//! experts the card then leaves it: the plan reads the host room
//! (`BLOOMERY_HOST_ROOM` replaces the reading) and the body serves no NVMe
//! expert tier (`PlaceError::NoNvmeTier`), so a shorter context, which keeps
//! more experts on the card, is what that refusal asks for.
//!
//! The records on stderr, in order: `place unset` (word and why), one `ctx`
//! line (`placement::ctx_ask::Chosen::line`), the `plan` record, the
//! `parallel` line (`rule=slots slots=1 slot_ctx=C total=C from=<word>
//! pass=one`); then the `load` record (architecture, resident bytes, the
//! slot's context, layers, the routed experts the plan keeps on the host and
//! on the card, the step graph's nodes, the prompt group and the load's wall)
//! and, once the port is bound, the `listening` record
//! (`record::BLOOMERY_SERVE_MINIMAXM3`). `--plan` ends the process after the
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
//! (`BLOOMERY_SCHEMAJUMP`). The load runs no adaptive expert residency.
//!
//! The model writes its reasoning between the think span's two tags (the pair
//! its description names, `models::ReasoningFormat::ThinkSpan`: the template's
//! `think_begin_token` and `think_end_token`); the server splits the
//! reasoning, runs the reasoning budget and writes a structured-output
//! request's close on them (`Server::with_think_tags`).
//!
//! The seat sits behind the `deepseek41` feature beside `minimaxm3`: the
//! server surface it binds through (`bind`, the `serve` and `sampler` crates)
//! is scoped to `deepseek41`; it runs no V4.1 code.
//!
//! The seat itself is the common owner's ([`super::hybrid`]): this module
//! holds what MiniMax-M3 forces — the architectures it serves, the plan's
//! inputs, the words its refusals name and the records it prints.

use bloomery_gpu_gates::record;
use bloomery_gpu_gates::side_seat;
use gguf::Split;
use model::arch::Arch;
use model::arch::place_hybrid::PlanInputs;
use model::placement::PlacementError;
use serve::ServeError;

use super::hybrid::Family;

/// The binary's name, its records' owner and the `note` line's.
const NAME: &str = "bloomery-serve-minimaxm3";

const USAGE: &str = "usage: bloomery-serve --model minimaxm3 [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C] [--alias NAME] [--chat-template-file PATH] \
                     [--place W] [--plan] [--queue-depth Q] [--api-key KEY] \
                     [--api-key-file FNAME]";

/// The levers this seat acts on: the plan's card budget and the host set's
/// load settings, which the open reads (`PlanLevers::from_levers`,
/// `Levers::host`), the batches a prompt group runs, the engine thread's pin,
/// the host legs' lane rule, the start screen and the schema jump. The V4.1
/// host tier's r8 sidecar (`BLOOMERY_R8`) is not among them: this body reads
/// its routed experts from the source.
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::PREFILL_GROUP,
    bloomery_levers::HOST_POPULATE,
    bloomery_levers::HOST_LOCK,
    bloomery_levers::CARD_DONTNEED,
    bloomery_levers::PIN_MAIN,
    bloomery_levers::HOST_LANES,
    bloomery_levers::SCREEN,
    bloomery_levers::SCHEMAJUMP,
];

/// The MiniMax-M3 family: what the common seat owner needs of it.
pub struct MiniMax3;

impl Family for MiniMax3 {
    type Body = bloomery_gpu_minimaxm3::Body;
    type Flags = ();
    type H = model::arch::minimaxm3::hparams::Hparams;

    const NAME: &'static str = self::NAME;
    const USAGE: &'static str = self::USAGE;
    const ACTS_ON: &'static [&'static str] = self::ACTS_ON;
    const ALIAS: &'static str = "minimax-m3";
    const FAMILY: &'static str = "minimaxm3";
    const SERVES: &'static str = "minimax-m3";

    fn accepts(arch: Arch) -> bool {
        matches!(arch, Arch::MiniMaxM3)
    }

    /// The family's own reading (`PlanInputs::describe`), as its end-to-end
    /// gate's seat reads the file: the coverage list still names the
    /// template's tool-call parser.
    fn read(split: &Split) -> Result<PlanInputs<Self::H>, PlacementError> {
        model::arch::minimaxm3::place::PlanInputs::describe(split)
    }

    fn flags(
        _flag: &str,
        _value: &str,
        _out: &mut (),
    ) -> Result<bool, bloomery_gpu_gates::GateError> {
        Ok(false)
    }

    fn register() {
        record::at_main(NAME, record::BLOOMERY_SERVE_MINIMAXM3);
    }

    fn listening() -> record::Record {
        record::Record::new(&record::LISTENING_MINIMAXM3)
    }

    fn load_record() -> record::Record {
        record::Record::new(&record::LOAD_MINIMAXM3)
    }
}

/// Loads the model, and the embedding model `side` names beside it, and serves until the listener
/// or the engine fails; `Ok` carries why the server ended.
pub fn run(
    args: &[String],
    side: &side_seat::Flags,
) -> Result<ServeError, bloomery_gpu_gates::GateError> {
    super::hybrid::run::<MiniMax3>(args, side)
}
