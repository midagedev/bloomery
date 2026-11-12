//! `gate_raonnano_serve` — the raon1 seat of `bloomery-serve` (the one seat both Raon dialects
//! take) against the model it serves on the Nano dialect's Q8_0 file of the profile
//! (`BLOOMERY_REF_MODEL`): the seat starts, opens the file by its detected architecture, loads it
//! and answers one chat request whose reasoning is split at the model's own tags.
//!
//!     gate_raonnano_serve --dir <dir>
//!
//! The seat serves one sequence on one card (`--parallel` above 1 is refused by name). Every
//! server is `bloomery-serve --model raon1 -m <file> …` beside this binary, `BLOOMERY_REF_MODEL`
//! removed from its environment. Logs per server in `<dir>/<name>/` (`server.err`, the answer).
//!
//! The load, chat, prefix and max-context clauses are the shared ones (`shared/serve_gate.rs`):
//! the seat refuses a second slot, loads the file at the context it was asked for (its load record
//! naming the body's word `raon1` — one body serves both dialects — the file's 24 layers and one
//! slot), answers a chat turn with its reasoning split at the model's tags (`<think>` and
//! `</think>`, the description's own `ThinkSpan`) and ends its turn at the model's own end token
//! (`<|im_end|>`, the file's eos), and keeps the prefix a request shares with the slot by the
//! body's rule: an RDA state cannot step back, so a shorter shared prefix is kept 0 and the prompt
//! fed again (`Cut::WhereItStands`). The seat runs no prompt cache, so the saved-state clause
//! skips it by name.
//!
//! - `parallel_above_one_is_refused`, `load_and_listen`, `chat_splits_the_reasoning_at_the_models_tags`,
//!   `chat_ends_at_the_turn_end`, `chat_answers_a_turn`, `live_slot_keeps_the_prefix`: the shared
//!   clauses, see `shared/serve_gate.rs`.
//! - `nano_drafts_nothing`: a Nano file drafts nothing (the DFlash2 draft is the Mini file's own,
//!   and the MTP block is run by no program). `--plan` with `BLOOMERY_DRAFT` removed prints the
//!   `draft rule` record at `off` with the why naming the raon1nano file; `BLOOMERY_DRAFT=dspark
//!   --plan` exits non-zero, its stderr names the seat's refusal and no `plan` record is printed
//!   (the refusal comes before the plan). FAIL-first: a seat that drafts a nano file prints
//!   `drafts` at another word and the unset half is red; one that runs the word it is given
//!   answers the `dspark` run's plan and the refusal half is red.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_raonnano_serve", gate::run())
}

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the shared clauses serve several gates; each takes the helpers it reads"
)]
#[path = "shared/serve_gate.rs"]
mod serve_gate;

#[cfg(feature = "gpu")]
mod gate {
    use std::path::Path;

    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::serve_client::{SeatGate, check};
    use bloomery_gpu_gates::{GateError, ref_model_path};
    use gguf::Split;
    use model::arch::raon1::place::PlanInputs;

    use crate::serve_gate;

    /// The slot's positions the server is asked for.
    const CTX: u64 = 2048;

    /// The token that closes the model's turn, as its vocabulary spells it (the file's eos).
    const TURN_END: &str = "<|im_end|>";

    /// The words of the refusal a `dspark` draft of a nano file gets.
    const DRAFT_REFUSED: &str = "the raon1 seat serves no draft of a raon1nano file";

    /// The words the unset draft rule's why names the file by.
    const DRAFT_WHY: &str = "the raon1nano file drafts nothing";

    /// The seat's facts, as the shared clauses read them: the body's own word `raon1` (one body
    /// serves both dialects; the file's own word is `raon1nano`, what `nano_drafts_nothing`
    /// reads), and the cut of a body whose RDA state cannot step back.
    fn clauses() -> Result<serve_gate::Clauses, GateError> {
        let file = ref_model_path()?;
        let split = Split::open(&file).map_err(|e| format!("open {}: {e}", file.display()))?;
        let layers = u64::try_from(PlanInputs::describe(&split)?.model.layers)?;
        drop(split);
        Ok(serve_gate::Clauses {
            gate: "gate_raonnano_serve",
            model: "raon1",
            kinds: record::BLOOMERY_SERVE_RAON1,
            load: &record::LOAD_RAON1,
            listening: &record::LISTENING_RAON1,
            arch: "raon1",
            layers,
            think: Some(["<think>", "</think>"]),
            turn_end: TURN_END,
            ctx: CTX,
            cut: serve_gate::Cut::WhereItStands,
        })
    }

    /// `nano_drafts_nothing` (the module header).
    fn nano_drafts_nothing(c: &serve_gate::Clauses, dir: &Path) -> Result<bool, GateError> {
        let mut ok = true;
        let file = ref_model_path()?;
        let seat = SeatGate {
            model: c.model,
            kinds: c.kinds,
        };
        let ctx = c.ctx.to_string();
        let (unset_status, unset, _) = seat.plan_under(
            (dir, "plan-draft-unset"),
            &file,
            &["--ctx", &ctx],
            &[(bloomery_levers::DRAFT, None)],
        )?;
        let unset_ok = match unset.first(&record::DRAFT_RULE_RAON1)? {
            Some(rule) => {
                let (drafts, why) = (rule.word("drafts")?, rule.word("why")?);
                let ok = unset_status.success() && drafts == "off" && why.contains(DRAFT_WHY);
                println!("draft rule unset: drafts {drafts:?} why {why:?}: {ok}");
                ok
            }
            None => {
                println!(
                    "draft rule unset: no draft rule record (exit {})",
                    unset_status
                );
                false
            }
        };
        let (dspark_status, dspark, dspark_text) = seat.plan_under(
            (dir, "plan-draft-dspark"),
            &file,
            &["--ctx", &ctx],
            &[(bloomery_levers::DRAFT, Some("dspark"))],
        )?;
        let dspark_refused = !dspark_status.success()
            && dspark_text.contains(DRAFT_REFUSED)
            && dspark.first(&record::PLAN)?.is_none();
        println!(
            "draft dspark: exit non-zero {}, names {DRAFT_REFUSED:?} {dspark_refused}; no plan {}",
            !dspark_status.success(),
            dspark.first(&record::PLAN)?.is_none()
        );
        if !dspark_refused {
            println!("plan-draft-dspark stderr:\n{dspark_text}");
        }
        check(&mut ok, "nano_drafts_nothing", unset_ok && dspark_refused);
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let c = clauses()?;
        let dir = serve_gate::dir_of(&c)?;
        std::fs::create_dir_all(&dir)?;
        let mut ok = serve_gate::parallel_above_one_is_refused(&c, &dir)?;
        ok &= serve_gate::served(&c, &dir)?;
        ok &= nano_drafts_nothing(&c, &dir)?;
        serve_gate::finish(&c, ok)
    }
}
