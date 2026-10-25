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
//! at the model's tags and ends its turn at the model's own end token, and runs the prefix
//! clause `live_slot_keeps_the_prefix` at its own cut's fact: the body's `app::Keep` keeps a
//! shared prefix only where the sequence stands — at the nearest checkpoint at or below it, and
//! the serving prompt call takes no checkpoint (`Body::kept`,
//! crates/app/src/arch/solar2/mod.rs:75), so a request that shares less than the slot holds
//! re-feeds its prompt from an empty model and the clause holds it to the counting rule alone.
//! The model's facts are this file's: the think tags and the turn end are written here as the
//! model's vocabulary spells them; the seat reads the tags from the model's description
//! (`models::ReasoningFormat::ThinkSpan`).
//!
//! - `parallel_above_one_is_refused`: `--parallel 2 --plan` exits non-zero, its stderr names the
//!   body's one sequence, and no `plan` record is printed (the refusal comes before any plan).
//!   FAIL-first: a seat that takes a second slot prints its plan and the clause is red.
//! - `residency_default_is_the_machine`: `--plan` at [`CTX`] with `BLOOMERY_RESIDENCY` removed
//!   from the environment prints the `residency lever` record at the serving word
//!   (`bloomery_levers::RESIDENCY_SERVING`) with why `place`, ahead of the plan, and the
//!   `residency host` record of the churn pool after the plan. `--place gate` prints `off` with
//!   why `fixed_place` and no pool. `BLOOMERY_RESIDENCY=off` prints `off` with why `set` and no
//!   pool. The body refuses `--parallel` above 1 before any plan, so the clause reads
//!   `--place gate` for a placement the machine does not serve. The clause is a `--plan`
//!   reading, so the fixture tier runs it on the fixture file.
//!   FAIL-first: a seat whose residency returns no pick prints no lever and the clause is red; a
//!   seat that treats `gate` as a serving placement prints the serving word and the gate half is
//!   red; a seat that omits the lever from the ones it acts on refuses `off` by name and the off
//!   half is red.
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
#[allow(
    dead_code,
    reason = "the shared clauses serve several gates; each takes the helpers it reads"
)]
#[path = "shared/serve_gate.rs"]
mod serve_gate;

#[cfg(feature = "deepseek41")]
mod gate {
    use std::path::Path;
    use std::process::ExitStatus;

    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::serve_client::{
        SeatGate, check, no_residency_host, residency_lever_is,
    };
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

    /// `--plan` of `args` with `BLOOMERY_RESIDENCY` set to `word`, or removed when it is `None`
    /// ([`SeatGate::plan_under`]).
    fn plan_under(
        seat: &SeatGate,
        dir: &Path,
        name: &str,
        file: &Path,
        args: &[&str],
        word: Option<&str>,
    ) -> Result<(ExitStatus, record::Log, String), GateError> {
        seat.plan_under(
            (dir, name),
            file,
            args,
            &[(bloomery_levers::RESIDENCY, word)],
        )
    }

    /// The lever, then the plan, then the churn pool's host record.
    fn hosted_in_order(log: &record::Log) -> Result<bool, GateError> {
        let (lever, plan, host) = (
            log.first(&record::RESIDENCY_LEVER)?,
            log.first(&record::PLAN)?,
            log.first(&record::RESIDENCY_HOST)?,
        );
        Ok(matches!(
            (&lever, &plan, &host),
            (Some(l), Some(p), Some(h)) if l.at() < p.at() && p.at() < h.at()
        ))
    }

    /// The rendered lever line, or a named absence.
    fn lever_line(log: &record::Log) -> Result<String, GateError> {
        Ok(match log.first(&record::RESIDENCY_LEVER)? {
            Some(l) => l.line().to_owned(),
            None => "(no residency lever record)".to_owned(),
        })
    }

    /// `residency_default_is_the_machine` (the module header).
    fn residency_default_is_the_machine(
        c: &serve_gate::Clauses,
        dir: &Path,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let file = ref_model_path()?;
        let seat = SeatGate {
            model: c.model,
            kinds: c.kinds,
        };
        let ctx = c.ctx.to_string();
        let (serving_status, serving, _) =
            plan_under(&seat, dir, "plan-residency", &file, &["--ctx", &ctx], None)?;
        let (gate_status, gate, _) = plan_under(
            &seat,
            dir,
            "plan-residency-gate",
            &file,
            &["--ctx", &ctx, "--place", "gate"],
            None,
        )?;
        let (off_status, off, _) = plan_under(
            &seat,
            dir,
            "plan-residency-off",
            &file,
            &["--ctx", &ctx],
            Some("off"),
        )?;
        let serving_word =
            residency_lever_is(&serving, bloomery_levers::RESIDENCY_SERVING, "place")?;
        let in_order = hosted_in_order(&serving)?;
        let gate_ok = residency_lever_is(&gate, "off", "fixed_place")? && no_residency_host(&gate)?;
        let off_ok = residency_lever_is(&off, "off", "set")? && no_residency_host(&off)?;
        let exits = serving_status.success() && gate_status.success() && off_status.success();
        println!(
            "residency default: exit 0 {exits}; unset lever {:?} is {} (place), then the plan, \
             then the pool {serving_word} {in_order}; --place gate {:?} is off (fixed_place) \
             with no pool {gate_ok}; set off {:?} is off (set) with no pool {off_ok}",
            lever_line(&serving)?,
            bloomery_levers::RESIDENCY_SERVING,
            lever_line(&gate)?,
            lever_line(&off)?,
        );
        check(
            &mut ok,
            "residency_default_is_the_machine",
            exits && serving_word && in_order && gate_ok && off_ok,
        );
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let file = ref_model_path()?;
        let split = Split::open(&file).map_err(|e| format!("open {}: {e}", file.display()))?;
        let layers = u64::try_from(PlanInputs::describe(&split)?.model.layers)?;
        drop(split);
        let c = serve_gate::Clauses {
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
            // The KDA layers' recurrent state keeps a shared prefix only where the sequence
            // stands or at a checkpoint (`Body::kept`, crates/app/src/arch/solar2/mod.rs:75),
            // and the serving prompt call takes no checkpoint.
            cut: serve_gate::Cut::WhereItStands,
        };
        let dir = serve_gate::dir_of(&c)?;
        std::fs::create_dir_all(&dir)?;
        let mut ok = serve_gate::parallel_above_one_is_refused(&c, &dir)?;
        ok &= residency_default_is_the_machine(&c, &dir)?;
        ok &= serve_gate::served(&c, &dir)?;
        serve_gate::finish(&c, ok)
    }
}
