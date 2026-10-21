//! `gate_solar2_serve` — the solar2 seat of `bloomery-serve` against the model it serves, on the
//! solar-open2 file of the profile (`BLOOMERY_REF_MODEL`; the fixture under the fixture tier): the
//! seat starts, loads the file and answers one chat request whose reasoning is split at the
//! model's own tags.
//!
//!     gate_solar2_serve --dir <dir>
//!
//! The seat serves one sequence on one card (`--parallel` above 1 is refused by name). Every
//! server is `bloomery-serve --model solar2 -m <file> …` beside this binary, `BLOOMERY_REF_MODEL`
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
//! - `load_and_listen`: one server at `--ctx` [`CTX`], `--port 0`, loads and listens; the `load`
//!   record comes before the `listening` record; the load names the file's architecture word, one
//!   slot, the file's layer count and [`CTX`]; the listening record names one slot and [`CTX`];
//!   `/health` answers 200. FAIL-first: a seat that loads another context than the one it was
//!   asked for names it in the load record and the clause is red.
//! - `chat_splits_the_reasoning_at_the_models_tags`: `/v1/chat/completions` of one user turn at
//!   temperature 0, `max_tokens` 64 and `reasoning_budget` 16 is a 200 whose
//!   `usage` counts a prompt and at most 64 completion tokens, whose message holds
//!   reasoning and content both, and neither of them holds `<|think:start|>` or `<|think:end|>`:
//!   the template's generation prompt opens the span, the budget forces the close of the model's
//!   own tags, and the server splits there. The tags are written here as the model's vocabulary
//!   spells them; the seat reads them from the model's description
//!   (`models::ReasoningFormat::ThinkSpan`). FAIL-first: a seat that serves on the default
//!   `<think>` tags (no `Server::with_think_tags`) splits nothing, the message has no reasoning
//!   and the clause is red.
//! - `chat_ends_at_the_turn_end`: the same answer finishes with `stop`, and its content holds no
//!   `<|im:end|>`: the model closes its turn with that token (the template writes it after every
//!   assistant message, and the reference vocabulary counts it an end-of-generation id), so the
//!   server stops there and leaves it out of the text. FAIL-first: a vocabulary that lists only
//!   the file's `eos_token_id` (``) lets the model run past its turn end, the token's text
//!   is in the content and the clause is red. The clause reads the file's text behaviour.
//!   PIN(2026-10-21): the fixture's weights never reach the turn end, so the clause is
//!   `Tag::FileBound` and the fixture tier leaves it to the real tier by name; the end id's place
//!   in the vocabulary is pinned on any machine by `tokenizer`'s
//!   `a_familys_turn_end_token_ends_generation`.

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_solar2_serve", gate::run())
}

#[cfg(feature = "deepseek41")]
#[path = "shared/serve_gate.rs"]
mod serve_gate;

#[cfg(feature = "deepseek41")]
mod gate {
    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::{GateError, ref_model_path};
    use gguf::Split;
    use model::arch::solar2::place::PlanInputs;

    use crate::serve_gate;

    /// The slot's positions the server is asked for.
    const CTX: u64 = 4096;

    /// The tags of the model's think span, as its vocabulary spells them.
    const THINK_TAGS: [&str; 2] = ["<|think:start|>", "<|think:end|>"];

    /// The token that closes the model's turn, as its vocabulary spells it.
    const TURN_END: &str = "<|im:end|>";

    pub fn run() -> Result<(), GateError> {
        let file = ref_model_path()?;
        let split = Split::open(&file).map_err(|e| format!("open {}: {e}", file.display()))?;
        let layers = u64::try_from(PlanInputs::describe(&split)?.model.layers)?;
        drop(split);
        serve_gate::run(&serve_gate::Clauses {
            gate: "gate_solar2_serve",
            model: "solar2",
            kinds: record::BLOOMERY_SERVE_SOLAR2,
            load: &record::LOAD_SOLAR2,
            listening: &record::LISTENING_SOLAR2,
            arch: "solar-open2",
            layers,
            think: THINK_TAGS,
            turn_end: TURN_END,
            ctx: CTX,
        })
    }
}
