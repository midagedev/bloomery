//! `gate_minimaxm3_serve` — the minimaxm3 seat of `bloomery-serve` against the model it serves,
//! on the minimax-m3 file of the profile (`BLOOMERY_REF_MODEL`): the seat starts, loads the file
//! and answers one chat request whose reasoning is split at the model's own tags.
//!
//!     gate_minimaxm3_serve --dir <dir>
//!
//! The seat serves one sequence on one card (`--parallel` above 1 is refused by name, and a
//! placement that hangs an expert tier card (`--place bp`) is refused by the plan). Every server
//! is `bloomery-serve --model minimaxm3 -m <file> …` beside this binary, `BLOOMERY_REF_MODEL`
//! removed from its environment. Logs per server in `<dir>/<name>/` (`server.err`, the answer).
//!
//! The clauses are the shared ones (`shared/serve_gate.rs`): the seat refuses a second slot,
//! loads the file at the context it was asked for, answers a chat turn with its reasoning split
//! at the model's tags and ends its turn at the model's own end token. The model's facts are
//! this file's: the think tags and the turn end are written here as the model's vocabulary
//! spells them; the seat reads the tags from the model's description
//! (`models::ReasoningFormat::ThinkSpan`).
//!
//! - `parallel_above_one_is_refused`: `--parallel 2 --plan` exits non-zero, its stderr names the
//!   body's one sequence, and no `plan` record is printed (the refusal comes before any plan).
//!   FAIL-first: a seat that takes a second slot prints its plan and the clause is red.
//! - `place_bp_is_refused`: `--place bp --plan` exits non-zero, its stderr names the place word it
//!   refuses, and no `plan` record is printed. The body serves no expert tier card: the refusal is
//!   the plan's own (`PlaceError::Tier`, naming the tier card the placement hangs) where the census
//!   holds a second card, and the placement's own (naming the visible cards) where it does not.
//!   FAIL-first: a plan that hangs the tier card would print its plan and the clause is red.
//! - `load_and_listen`: one server at `--ctx` 2560, `--port 0`, loads and listens; the `load`
//!   record comes before the `listening` record; the load names the file's architecture word, one
//!   slot, the file's layer count and 2560; the listening record names one slot and 2560;
//!   `/health` answers 200. FAIL-first: a seat that loads another context than the one it was
//!   asked for names it in the load record and the clause is red.
//! - `chat_splits_the_reasoning_at_the_models_tags`: `/v1/chat/completions` of one user turn at
//!   temperature 0, `max_tokens` 64 and `reasoning_budget` 16 is a 200 whose `usage` counts a
//!   prompt and at most 64 completion tokens, whose message holds reasoning and content both, and
//!   neither of them holds the model's think span's tags (<mm:think>, </mm:think>): the template's generation
//!   prompt opens the span, the budget forces the close of the model's own tags, and the server
//!   splits there. The tags are written here as the model's vocabulary spells them; the seat
//!   reads them from the model's description (`models::ReasoningFormat::ThinkSpan`). FAIL-first: a
//!   seat that serves on the default `<think>` tags (no `Server::with_think_tags`) splits
//!   nothing, the message has no reasoning and the clause is red.
//! - `chat_ends_at_the_turn_end`: the same answer finishes with `stop`, and its content holds no
//!   `[e~[`: the model closes its turn with that token (the template writes it after every
//!   assistant message, and the reference vocabulary counts it an end-of-generation id), so the
//!   server stops there and leaves it out of the text. FAIL-first: a vocabulary that lists only
//!   the file's `eos_token_id` (200020) lets the model run past its turn end, the token's text
//!   is in the content and the clause is red. The clause reads the file's text behaviour.

#[cfg(feature = "minimaxm3")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_minimaxm3_serve", gate::run())
}

#[cfg(feature = "minimaxm3")]
#[allow(
    dead_code,
    reason = "the shared clauses serve several gates; each takes the helpers it reads"
)]
#[path = "shared/serve_gate.rs"]
mod serve_gate;

#[cfg(feature = "minimaxm3")]
mod gate {
    use std::path::Path;

    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::serve_client::{SeatGate, check};
    use bloomery_gpu_gates::{GateError, ref_model_path};
    use gguf::Split;
    use model::arch::minimaxm3::place::PlanInputs;

    use crate::serve_gate;

    /// The slot's positions the server is asked for.
    const CTX: u64 = 2560;

    /// The tags of the model's think span, as its vocabulary spells them.
    const THINK_TAGS: [&str; 2] = ["<mm:think>", "</mm:think>"];

    /// The token that closes the model's turn, as its vocabulary spells it.
    const TURN_END: &str = "[e~[";

    /// The words of the refusal of the placement: the place word it refuses.
    const PLACED: &str = "--place bp";

    /// The seat's facts, as the shared clauses read them.
    fn clauses() -> Result<serve_gate::Clauses, GateError> {
        let file = ref_model_path()?;
        let split = Split::open(&file).map_err(|e| format!("open {}: {e}", file.display()))?;
        let layers = u64::try_from(PlanInputs::describe(&split)?.model.layers)?;
        drop(split);
        Ok(serve_gate::Clauses {
            gate: "gate_minimaxm3_serve",
            model: "minimaxm3",
            kinds: record::BLOOMERY_SERVE_MINIMAXM3,
            load: &record::LOAD_MINIMAXM3,
            listening: &record::LISTENING_MINIMAXM3,
            arch: "minimax-m3",
            layers,
            think: THINK_TAGS,
            turn_end: TURN_END,
            ctx: CTX,
        })
    }

    /// `place_bp_is_refused`: `--place bp --plan` exits non-zero, its stderr names the place word
    /// it refuses, and no `plan` record is printed. FAIL-first: a plan that hangs the tier card
    /// prints its plan and the clause is red.
    fn place_bp_is_refused(c: &serve_gate::Clauses, dir: &Path) -> Result<bool, GateError> {
        let mut ok = true;
        let file = ref_model_path()?;
        let seat = SeatGate {
            model: c.model,
            kinds: c.kinds,
        };
        let (status, log, text) = seat.plan(dir, "refuse-place-bp", &file, &["--place", "bp"])?;
        let named = text.contains(PLACED);
        let no_plan = log.first(&record::PLAN)?.is_none();
        println!(
            "refuse-place-bp: exit non-zero {}; names {PLACED:?} {named}; no plan record {no_plan}",
            !status.success()
        );
        if !named {
            println!("refuse-place-bp stderr:\n{text}");
        }
        check(
            &mut ok,
            "place_bp_is_refused",
            !status.success() && named && no_plan,
        );
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let c = clauses()?;
        let dir = serve_gate::dir_of(&c)?;
        std::fs::create_dir_all(&dir)?;
        let mut ok = serve_gate::parallel_above_one_is_refused(&c, &dir)?;
        ok &= place_bp_is_refused(&c, &dir)?;
        ok &= serve_gate::served(&c, &dir)?;
        serve_gate::finish(&c, ok)
    }
}
