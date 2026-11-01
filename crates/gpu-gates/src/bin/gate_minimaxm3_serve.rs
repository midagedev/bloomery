//! `gate_minimaxm3_serve` — the minimaxm3 seat of `bloomery-serve` against the model it serves,
//! on the MiniMax-M3.1-Flash checkpoint folder of the profile (`BLOOMERY_REF_MODEL`; the fixture
//! cut under the fixture tier): the seat starts, opens the folder as stored, loads it, serves two
//! slots in turn and answers one chat request whose reasoning is split at the model's own tags.
//! The model is the folder, or a minimax-m3 GGUF file where one is named: the seat's open is one
//! owner for both (`shared/serve_seats/minimaxm3_folder.rs`).
//!
//!     gate_minimaxm3_serve --dir <dir>
//!
//! The seat serves `--parallel N` resident sequences on one card, or on plan (b′) where the census
//! holds two (the A6000 the stage, the 3090 an expert tier). A round of several is a select and a
//! step each (`pass=turns`): the step walks one token, so one pass of several slots is a host walk
//! the body does not run. Every server is `bloomery-serve --model minimaxm3 -m <file> …` beside
//! this binary, `BLOOMERY_REF_MODEL` removed from its environment. Logs per server in
//! `<dir>/<name>/` (`server.err`, the answer).
//!
//! The plan and slot clauses are this file's. The load and the chat are the shared ones
//! (`shared/serve_gate.rs`): the seat loads the file at the context it was asked for, answers a
//! chat turn with its reasoning split at the model's tags and ends its turn at the model's own
//! end token, and keeps the prefix a request shares with the slot the server holds
//! (`live_slot_keeps_the_prefix`: its cut keeps every held position, so the shorter shared
//! prefix is kept whole and its cached answer equals the uncached one on a server whose
//! residency is off). The model's facts are this file's: the think tags and the turn end are
//! written here as the model's vocabulary spells them; the seat reads the tags from the model's
//! description (`models::ReasoningFormat::ThinkSpan`).
//!
//! - `slots_total_is_planned`: `--parallel 2 --ctx` [`CTX`] `--plan` exits 0 and its stderr
//!   carries `parallel rule=slots slots=2 slot_ctx=<CTX/2> total=<CTX> from=flag pass=turns`.
//!   FAIL-first: a seat that still refuses a second slot, or that prints `pass=one`, misses the
//!   line and the clause is red.
//! - `place_bp_follows_the_cards`: `--place bp --plan` where the census holds two cards exits 0
//!   and prints a `plan` record whose devices name the tier card (`tier0`); where it holds one, it
//!   exits non-zero, its stderr names the place word the placement refuses, and no `plan` record
//!   is printed. FAIL-first: a description that leaves the tier reserve unset refuses the tier
//!   card (`PlaceError::Tier`) and the clause is red on two cards.
//! - `residency_default_is_the_machine`: `--plan` at [`CTX`] with `BLOOMERY_RESIDENCY` removed
//!   from the environment prints the `residency lever` record at the serving word
//!   (`bloomery_levers::RESIDENCY_SERVING`) with why `place`, ahead of the plan, and the
//!   `residency host` record of the churn pool after the plan. `--place gate` prints `off` with
//!   why `fixed_place` and no pool. `BLOOMERY_RESIDENCY=off` prints `off` with why `set` and no
//!   pool. `--place gate` is a placement the machine does not serve, so the clause reads it for
//!   the `fixed_place` half. FAIL-first: a seat whose residency returns no pick prints no lever and the
//!   clause is red; a seat that treats `gate` as a serving placement prints the serving word and
//!   the gate half is red; a seat that omits the lever from the ones it acts on refuses `off` by
//!   name and the off half is red.
//!   A checkpoint folder's residency is a file's, resolved against the plan its load runs: on a
//!   folder the unset run prints a word the plan resolved (its why neither `set` nor
//!   `fixed_place`; the serving word where it is `place`, `off` where the plan left it no room or
//!   no host expert, as the fixture's cut folder holds every expert on the card) ahead of the
//!   plan, and the churn pool's `residency host` record after the plan exactly where the word is
//!   not `off` (`serve_client::residency_lever_resolved`). FAIL-first: a seat that runs a
//!   folder's load under `off` whatever the plan prints `off` with why `fixed_place` and the
//!   folder half is red.
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
//!   The template's generation prompt opens no span (the thinking mode is always adaptive), so
//!   the model opens it on a leading tag or answers with none: the clause above reads the real
//!   file's choice for this turn, and the split itself is pinned on any machine by `serve`'s
//!   `hw_minimaxm31_reasoning_splits_at_its_tags`.
//!   PIN(2026-10-23): the fixture cut keeps three of the model's layers, whose weights write
//!   neither a span nor the turn end, so the last two clauses are `Tag::FileBound` and the
//!   fixture tier leaves them to the real tier by name; the end id's place in the vocabulary is
//!   pinned on any machine by `tokenizer`'s `a_familys_turn_end_token_ends_generation`.
//! - `saved_state_returns` (shared, `serve_gate::saved_state_returns`, on the server of
//!   `load_and_listen`): the seat prints a `cache` line with a budget above 0 (a seat with none
//!   skips the clause by name). Raw-id conversation A of 1024 ids is opened by its head (its ids
//!   less the last) and answered, so the live slot holds the head's rows and A continues it by one
//!   id: that answer is the truth. An unrelated conversation B of 256 ids takes the slot and A
//!   returns: `timings.cache_n` is at least 1023 (restored from the saved state, not fed again)
//!   and, where the server's `residency lever` record is `off` (any other word skips this half by
//!   name), the ids equal the truth's, the same arithmetic on the same rows. Then B takes the slot
//!   again and A returns extended by its answer and a tail: `cache_n` is at least A's 1024 ids.
//!   An uncached run is no oracle: another cut of the prompt or its answer's stepped rows in a
//!   prompt pass differ by rounding. FAIL-first: a resume that puts back one position leaves
//!   `cache_n` at 1 and the clause is red on it; one that stands at the positions asked over the
//!   rows of one is red on the ids (the fixture cut answers one id whatever the rows: that half is
//!   the real tier's).
//! - `chat_answers_a_turn`: the same request, on either tier, is a 200 whose `usage` counts a
//!   prompt and at most 512 completion tokens, whose message holds neither think tag, and whose
//!   `finish_reason` is `stop` or `length`. FAIL-first: a seat whose engine ends the request
//!   without a token, or a vocabulary that writes a tag into the message, is red.
//! - `two_slots_decode_as_one_slot_runs`: one server at `--ctx` [`CTX`] greedy-completes two
//!   tokenized prompts in turn (`cache_prompt` false), then stops. A server at `--ctx` `2*CTX`
//!   and `--parallel 2` runs the same two prompts together, both posted at once (the second is
//!   admitted while the first holds its slot, on a body of any speed), and each answer's tokens
//!   and `stop_type` equal its solo run, both non-empty, on different `id_slot`. Both servers plan
//!   at a pinned budget on the stage card ([`gate_budget`]: a `BLOOMERY_CARD_BUDGET` list naming
//!   that card alone, the gate card's usable bytes less one slot's stores for the one-slot
//!   server, those bytes for the two-slot server): a second slot's stores take stage card room,
//!   so the two-slot plan an unpinned server makes holds fewer stage experts, and another
//!   card/host split answers other ids; the two budgets differ by one slot's stores and both sit
//!   at or under the stage card's usable bytes (the plan takes the lesser), so the stage card's
//!   experts are planned in the same room. An expert tier card (plan (b′), both cards in view)
//!   holds no stores, so the list leaves it out and it plans at its own usable bytes in both
//!   servers; one budget for every card would hold the one-slot server's tier lower by one
//!   slot's stores and the host would take those experts. The stage's ordinal is the one a
//!   `--plan` of the one-slot server's shape names. The two `plan` records then name one split
//!   (`card_experts`, `host_experts`, `n_l`); where they do not, the clause is red by name
//!   (`the two servers plan different splits`) before any id is compared. Every server of the
//!   clause runs with `BLOOMERY_RESIDENCY=off`: the residency machine moves experts between the
//!   tiers by the requests' heat, so a second request on one server, or two requests on two
//!   slots, run on another split than a fresh sequence, and a near tie answers other ids. The
//!   two-slot load and listening records name two slots and [`CTX`] (a slot's positions). The
//!   resident bytes grow by one sequence's stores at [`CTX`] (`KvLayout::bytes`) and by nothing
//!   else. `swaps_total` is absent or 0. The one-slot server is stopped before the two-slot
//!   server starts. FAIL-first: a two-slot budget without the added stores plans another split
//!   and the clause is red on the plan line; one budget for every card, the tier card included,
//!   plans the one-slot server's tier fewer experts and the clause is red on the plan line; a
//!   select that leaves both requests on one sequence makes the ids differ or the slots match,
//!   and the clause is red; a `seq_bytes` that under-counts the planes makes the resident
//!   difference miss one sequence's stores, and the clause is red.
//! - The premise `two_slots_decode_as_one_slot_runs`' solo runs stand on — that a server's second
//!   request answers the ids a fresh sequence does — is checked beside them: a third server, one
//!   slot at the one-slot
//!   budget, answers `animals` FIRST, and the sequential answer's tokens and `stop_type` equal it
//!   (`second_request_equals_a_fresh_server`). Every `animals` run and each of the two-slot
//!   round's asks each token's two best logits (`n_probs` 2, which changes no id: the row read
//!   for them is the row the id was taken from), and where two compared answers first differ the
//!   clause prints each side's top id, its margin over its runner-up, and whether the other
//!   side's id is that runner-up — the line that tells a near tie from a real divergence. FAIL
//!   -first: a seat whose second request answers another id than a fresh server's misses the
//!   equality and the clause is red.

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
#[allow(
    dead_code,
    reason = "the seats' folder open serves the gates too; the gate reads the model's header only"
)]
#[path = "shared/serve_seats/folder.rs"]
mod folder;

#[cfg(feature = "minimaxm3")]
#[allow(
    dead_code,
    reason = "the seat's open serves the gates too; the gate reads the model's header only"
)]
#[path = "shared/serve_seats/minimaxm3_folder.rs"]
mod minimaxm3_folder;

#[cfg(feature = "minimaxm3")]
mod gate {
    use std::path::{Path, PathBuf};
    use std::process::ExitStatus;
    use std::time::Duration;

    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::serve_client::{
        Answer, SeatGate, Served, check, curl, greedy, held_ms, ids_of, json_of, metric,
        no_residency_host, residency_lever_is, residency_lever_resolved, server_log, stop_of,
        together, tokenized,
    };
    use bloomery_gpu_gates::{GateError, checks_failed, gpu_census, ref_model_path};
    use model::arch::minimaxm3::place::PlanInputs;
    use model::placement::workstation::RTX_3090;
    use serde_json::json;

    use crate::minimaxm3_folder;
    use crate::serve_gate;

    /// The slot's positions the server is asked for.
    const CTX: u64 = 2560;

    /// The tags of the model's think span, as its vocabulary spells them.
    const THINK_TAGS: [&str; 2] = ["<mm:think>", "</mm:think>"];

    /// The token that closes the model's turn, as its vocabulary spells it.
    const TURN_END: &str = "[e~[";

    /// Polls of one second that bound the wait for a server to listen.
    const LISTEN_POLLS: usize = 1000;

    /// Ids each greedy run is held to. Both servers are compared at this length.
    const PREDICT: usize = 8;

    /// The shared clauses' chat turn (`serve_gate` `TURN`).
    const ONE_TURN: &str = "Name three primary colours.";

    /// The other conversation the ds41 serve gate sends (`OTHER`).
    const OTHER_TURN: &str = "Name three farm animals.";

    /// The words of the refusal of the placement: the place word it refuses.
    const PLACED: &str = "--place bp";

    /// The card budgets the servers of `two_slots_decode_as_one_slot_runs` plan at, the one-slot
    /// server's first, each a list naming the stage card (`stage`, its CUDA ordinal) alone. The
    /// stage card holds every slot's stores, so its experts' room is its budget less them: the
    /// two-slot server's budget is the gate card's usable bytes (`RTX_3090`'s), so whichever card
    /// the runner puts in view the room is fixed, and the one-slot server's is that less one
    /// slot's `stores`. Both bind at or under the card's own bytes (the plan takes `min(usable,
    /// budget)`; a budget past them is cut back and the second slot's stores come out of the
    /// experts' room), so the stage card plans the same experts in both. An expert tier card
    /// holds no stores, so its room does not move with the slots: the lists leave it out and it
    /// plans at its own usable bytes in both servers. One budget for every card would hold the
    /// one-slot server's tier lower by one slot's stores and the host would take those experts.
    fn gate_budget(
        stores: u64,
        stage: u32,
    ) -> Result<(bloomery_levers::CardBudget, bloomery_levers::CardBudget), GateError> {
        let two = RTX_3090.usable_bytes();
        let one = two
            .checked_sub(stores)
            .ok_or_else(|| format!("one slot's stores {stores} B past the gate card's {two} B"))?;
        let on_stage = |bytes| bloomery_levers::CardBudget::Per(vec![(stage, bytes)]);
        Ok((on_stage(one), on_stage(two)))
    }

    /// The seat's facts, as the shared clauses read them.
    fn clauses() -> Result<serve_gate::Clauses, GateError> {
        let file = ref_model_path()?;
        let model = minimaxm3_folder::model_at(&file)?;
        let layers = u64::try_from(PlanInputs::describe(model.header())?.model.layers)?;
        drop(model);
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
            // Every store is a plane of positions, so the session's cut keeps every held
            // position (`app::Keep`, crates/app/src/arch/minimaxm3/mod.rs:11).
            cut: serve_gate::Cut::EveryPosition,
        })
    }

    /// `place_bp_follows_the_cards` (the module header); `two` whether the census holds two cards.
    fn place_bp_follows_the_cards(
        c: &serve_gate::Clauses,
        dir: &Path,
        two: bool,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let file = ref_model_path()?;
        let seat = SeatGate {
            model: c.model,
            kinds: c.kinds,
        };
        let (status, log, text) = seat.plan(dir, "place-bp", &file, &["--place", "bp"])?;
        let plan = log.first(&record::PLAN)?;
        let follows = if two {
            let devices: Vec<String> = match &plan {
                Some(p) => p.csv("devices")?.iter().map(|d| (*d).to_owned()).collect(),
                None => Vec::new(),
            };
            let tiered = devices.iter().any(|d| d.starts_with("tier0"));
            println!(
                "place-bp on two cards: exit 0 {}; plan devices {devices:?}, a tier card {tiered}",
                status.success()
            );
            status.success() && tiered
        } else {
            let named = text.contains(PLACED);
            println!(
                "place-bp on one card: exit non-zero {}; names {PLACED:?} {named}; no plan record {}",
                !status.success(),
                plan.is_none()
            );
            !status.success() && named && plan.is_none()
        };
        if !follows {
            println!("place-bp stderr:\n{text}");
        }
        check(&mut ok, "place_bp_follows_the_cards", follows);
        Ok(ok)
    }

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
        // A checkpoint folder's unset run is a word its plan resolved, a file's the serving word:
        // either way the lever's record ahead of the plan and the pool after it where it runs.
        let folder = file.is_dir();
        let serving_ok = if folder {
            residency_lever_resolved(&serving)?
        } else {
            residency_lever_is(&serving, bloomery_levers::RESIDENCY_SERVING, "place")?
                && hosted_in_order(&serving)?
        };
        let gate_ok = residency_lever_is(&gate, "off", "fixed_place")? && no_residency_host(&gate)?;
        let off_ok = residency_lever_is(&off, "off", "set")? && no_residency_host(&off)?;
        let exits = serving_status.success() && gate_status.success() && off_status.success();
        println!(
            "residency default ({}): exit 0 {exits}; unset lever {:?} is {}, ahead of the plan, \
             with the pool after it {} {serving_ok}; --place gate {:?} is off (fixed_place) with \
             no pool {gate_ok}; set off {:?} is off (set) with no pool {off_ok}",
            if folder { "a folder" } else { "a file" },
            lever_line(&serving)?,
            if folder {
                "resolved by the plan"
            } else {
                bloomery_levers::RESIDENCY_SERVING
            },
            if folder {
                "exactly where it is not off"
            } else {
                "(place)"
            },
            lever_line(&gate)?,
            lever_line(&off)?,
        );
        check(
            &mut ok,
            "residency_default_is_the_machine",
            exits && serving_ok && gate_ok && off_ok,
        );
        Ok(ok)
    }

    /// `slots_total_is_planned` (the module header).
    fn slots_total_is_planned(c: &serve_gate::Clauses, dir: &Path) -> Result<bool, GateError> {
        let mut ok = true;
        let file = ref_model_path()?;
        let seat = SeatGate {
            model: c.model,
            kinds: c.kinds,
        };
        let ctx = c.ctx.to_string();
        let (status, _, text) = seat.plan(
            dir,
            "plan-two-slots",
            &file,
            &["--parallel", "2", "--ctx", &ctx],
        )?;
        let slot_ctx = c.ctx / 2;
        let want = format!(
            "parallel rule=slots slots=2 slot_ctx={slot_ctx} total={} from=flag pass=turns",
            c.ctx
        );
        let split = status.success() && text.lines().any(|l| l == want);
        println!(
            "two slots over ctx {}: the parallel line {want:?} {split}",
            c.ctx
        );
        if !split {
            println!("plan-two-slots stderr:\n{text}");
        }
        check(&mut ok, "slots_total_is_planned", split);
        Ok(ok)
    }

    /// The gate's end. The `PASSED` line names the slot clauses and the shared load and chat.
    fn finish(c: &serve_gate::Clauses, ok: bool) -> Result<(), GateError> {
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {} the seat plans two slots (pass=turns), two concurrent greedy requests \
             each equal their solo run, the second solo run a fresh server's own, loads at the \
             context it was asked for, answers a chat turn with its reasoning split at the \
             model's tags, and keeps the prefix a request shares with the slot \
             (live_slot_keeps_the_prefix)",
            c.gate
        );
        Ok(())
    }

    /// `bloomery-serve` of `args` under `<dir>/<name>/`, its plan held to `budget`
    /// (`BLOOMERY_CARD_BUDGET`: the plan takes `min(usable, budget)` of each card the budget
    /// names) and its split fixed for the run (`BLOOMERY_RESIDENCY=off`).
    fn spawn_at(
        c: &serve_gate::Clauses,
        dir: &Path,
        name: &str,
        args: &[&str],
        budget: &bloomery_levers::CardBudget,
    ) -> Result<(Served, PathBuf), GateError> {
        let d = dir.join(name);
        std::fs::create_dir_all(&d)?;
        let file = ref_model_path()?;
        let seat = SeatGate {
            model: c.model,
            kinds: c.kinds,
        };
        let (mut cmd, mut all) = seat.command(&file)?;
        cmd.env(bloomery_levers::CARD_BUDGET, budget.to_string());
        cmd.env(bloomery_levers::RESIDENCY, "off");
        all.extend(args.iter().copied().map(str::to_owned));
        let all: Vec<&str> = all.iter().map(String::as_str).collect();
        Ok((Served::spawn_cmd(cmd, &all, &d)?, d))
    }

    /// The address, or the stderr when the server never listens.
    fn listened(s: &mut Served, err_log: &Path, what: &str) -> Result<Option<String>, GateError> {
        match s.address(err_log, LISTEN_POLLS, Duration::from_secs(1)) {
            Ok(addr) => Ok(Some(addr)),
            Err(e) => {
                println!("the {what} server never listened: {e}");
                println!(
                    "server stderr:\n{}",
                    std::fs::read_to_string(err_log).unwrap_or_default()
                );
                Ok(None)
            }
        }
    }

    /// The CUDA ordinal of the stage card the clause's servers plan on: the stage device of the
    /// `plan` record of a `--plan` at the one-slot server's shape. `None` where the plan does not
    /// run or prints no record.
    fn stage_of(c: &serve_gate::Clauses, dir: &Path, ctx: &str) -> Result<Option<u32>, GateError> {
        let file = ref_model_path()?;
        let seat = SeatGate {
            model: c.model,
            kinds: c.kinds,
        };
        let (status, log, _) = seat.plan(dir, "plan-stage", &file, &["--ctx", ctx])?;
        if !status.success() {
            return Ok(None);
        }
        let Some(plan) = log.first(&record::PLAN)? else {
            println!("the stage card's plan printed no plan record");
            return Ok(None);
        };
        let stage = record::stage_ordinal(&plan)?;
        println!(
            "the servers plan on the stage card cuda{stage} of {:?}",
            plan.csv("devices")?
        );
        Ok(Some(stage))
    }

    /// One greedy completion of `text`, `cache_prompt` false.
    fn greedy_text(
        url: &dyn Fn(&str) -> String,
        d: &Path,
        name: &str,
        text: &str,
    ) -> Result<Answer, GateError> {
        let ids = tokenized(url, text)?;
        let (a, _) = greedy(url, ids, PREDICT, d, name, false)?;
        Ok(a)
    }

    /// What one greedy request answered: the ids, the stop, the slot it ran on, and each token's
    /// two best logits. The probabilities change no id: the row read for them is the row the id
    /// was taken from, so the ids are the ones the same request without them answers.
    struct Ran {
        prompt: Vec<u32>,
        tokens: Vec<u32>,
        stop: String,
        slot: Option<u64>,
        /// A token's two best by logit, the taken id first: `(id, logprob)`.
        top: Vec<[(u32, f64); 2]>,
        /// The server's `prompt_ms` plus `predicted_ms`: about how long the request held its
        /// slot, the window a request posted beside it has to arrive in to take the other one.
        held_ms: f64,
    }

    /// One greedy `/completion` of `prompt`, [`PREDICT`] tokens, `cache_prompt` false, each
    /// token's two best logits asked (`n_probs` 2), its body in `<dir>/<name>.json`. Refused by
    /// name when the answer carries no two best for a token it answered.
    fn greedy_probs(
        url: &dyn Fn(&str) -> String,
        d: &Path,
        name: &str,
        prompt: &[u32],
    ) -> Result<Ran, GateError> {
        let body = json!({
            "prompt": prompt, "n_predict": PREDICT, "temperature": 0, "return_tokens": true,
            "cache_prompt": false, "n_probs": 2,
        });
        let (st, text) = curl(&url("/completion"), Some(&body), false)?;
        std::fs::write(d.join(format!("{name}.json")), &text)?;
        let v = json_of("/completion", st, &text)?;
        let tokens = ids_of(&v["tokens"])?;
        let stop = stop_of(&v)?.to_owned();
        let slot = v["id_slot"].as_u64();
        let held_ms = held_ms(name, &v)?;
        let entries = v["completion_probabilities"]
            .as_array()
            .ok_or("a completion that asked carries completion_probabilities")?;
        let mut top = Vec::with_capacity(entries.len());
        for (k, e) in entries.iter().enumerate() {
            let two = e["top_logprobs"]
                .as_array()
                .and_then(|t| {
                    let (a, b) = (t.first()?, t.get(1)?);
                    Some([
                        (
                            u32::try_from(a["id"].as_u64()?).ok()?,
                            a["logprob"].as_f64()?,
                        ),
                        (
                            u32::try_from(b["id"].as_u64()?).ok()?,
                            b["logprob"].as_f64()?,
                        ),
                    ])
                })
                .ok_or(format!(
                    "{name} token {k}: its completion_probabilities entry carries no two best"
                ))?;
            top.push(two);
        }
        if top.len() != tokens.len() {
            return Err(format!(
                "{name}: {} completion_probabilities entries for {} answered tokens",
                top.len(),
                tokens.len()
            )
            .into());
        }
        println!("{name} completion tokens {tokens:?} stop {stop} with each token's two best");
        Ok(Ran {
            prompt: prompt.to_vec(),
            tokens,
            stop,
            slot,
            top,
            held_ms,
        })
    }

    /// The first step the two requests answer different ids at, each side's top id and its
    /// margin over its runner-up there, and whether the other side's id is that runner-up: what
    /// tells a near tie (each side's id the other's runner-up at a small margin) from a real
    /// divergence. `None` when they answer the same ids.
    fn first_step_apart(a: (&str, &Ran), b: (&str, &Ran)) -> Option<String> {
        let ((a_name, a), (b_name, b)) = (a, b);
        let k = match a.tokens.iter().zip(&b.tokens).position(|(x, y)| x != y) {
            Some(k) => k,
            None if a.tokens.len() == b.tokens.len() => return None,
            None => {
                return Some(format!(
                    "one answered a prefix of the other's {} ids",
                    a.tokens.len().min(b.tokens.len())
                ));
            }
        };
        let side = |name: &str, r: &Ran, other: u32| {
            let [(id, lp), (next, next_lp)] = r.top[k];
            format!(
                "{name}'s {id} over {next} by {:.3}{}",
                lp - next_lp,
                if next == other {
                    " (the other side's id its runner-up)"
                } else {
                    ""
                }
            )
        };
        Some(format!(
            "step {k}: {} against {}",
            side(a_name, a, b.tokens[k]),
            side(b_name, b, a.tokens[k])
        ))
    }

    /// The card/host split a `plan` record names: the experts on each side and the range of
    /// experts a held layer keeps. Equal splits answer equal ids; the card budget is not part
    /// of it, two plans reach one split from different budgets.
    #[derive(Debug, PartialEq, Eq)]
    struct Planned {
        card: u64,
        host: u64,
        n_l: String,
    }

    impl Planned {
        fn of(plan: &record::Fields) -> Result<Planned, GateError> {
            Ok(Planned {
                card: plan.u64("card_experts")?,
                host: plan.u64("host_experts")?,
                n_l: plan.word("n_l")?.to_owned(),
            })
        }
    }

    /// `two_slots_decode_as_one_slot_runs` (the module header). The one-slot server is stopped
    /// before the two-slot server starts: the gate holds one card.
    fn two_slots_decode_as_one_slot_runs(
        c: &serve_gate::Clauses,
        dir: &Path,
    ) -> Result<bool, GateError> {
        const NAME: &str = "two_slots_decode_as_one_slot_runs";
        let mut ok = true;
        let file = ref_model_path()?;
        let model = minimaxm3_folder::model_at(&file)?;
        let layers = usize::try_from(c.layers)?;
        let store_bytes = PlanInputs::describe(model.header())?
            .kv
            .bytes(0..layers, c.ctx);
        drop(model);

        let ctx = c.ctx.to_string();
        let Some(stage) = stage_of(c, dir, &ctx)? else {
            check(&mut ok, NAME, false);
            return Ok(ok);
        };
        let (budget_one, budget_two) = gate_budget(store_bytes, stage)?;
        let (mut one, d) = spawn_at(
            c,
            dir,
            "serve-one",
            &["--port", "0", "--ctx", &ctx],
            &budget_one,
        )?;
        let err_log = d.join("server.err");
        let Some(addr) = listened(&mut one, &err_log, "one-slot")? else {
            check(&mut ok, NAME, false);
            return Ok(ok);
        };
        let url = |p: &str| format!("http://{addr}{p}");
        let colours = greedy_text(&url, &d, "colours", ONE_TURN)?;
        let animals = greedy_probs(&url, &d, "animals", &tokenized(&url, OTHER_TURN)?)?;
        let records = server_log(&err_log, c.kinds)?;
        let load = records.one(c.load)?;
        let one_plan = Planned::of(&records.one(&record::PLAN)?)?;
        let resident_one = load.u64("resident_bytes")?;
        let one_shaped = load.u64("slots")? == 1 && load.u64("ctx")? == c.ctx;
        println!(
            "one-slot server at {addr}; load {:?}; resident {resident_one} B; slots 1 and ctx {} \
             {one_shaped}",
            load.line(),
            c.ctx
        );
        println!("one-slot server stopped: {}", one.stop()?);

        // The premise the sequential runs stand on: a second request answers the ids a fresh
        // sequence does. A third server, one slot at the same budget, answers `animals` first.
        let (mut fresh, df) = spawn_at(
            c,
            dir,
            "serve-fresh",
            &["--port", "0", "--ctx", &ctx],
            &budget_one,
        )?;
        let err_log = df.join("server.err");
        let Some(addr) = listened(&mut fresh, &err_log, "fresh one-slot")? else {
            check(&mut ok, NAME, false);
            return Ok(ok);
        };
        let url = |p: &str| format!("http://{addr}{p}");
        let animals_fresh =
            greedy_probs(&url, &df, "animals_fresh", &tokenized(&url, OTHER_TURN)?)?;
        println!(
            "animals first on a fresh one-slot server at {addr}: {:?}",
            animals_fresh.tokens
        );
        println!("fresh one-slot server stopped: {}", fresh.stop()?);
        let second_is_fresh = !animals_fresh.tokens.is_empty()
            && animals.tokens == animals_fresh.tokens
            && animals.stop == animals_fresh.stop;
        println!("the second request's ids equal a fresh server's {second_is_fresh}");
        if let Some(step) = first_step_apart(("sequential", &animals), ("fresh", &animals_fresh)) {
            println!("animals sequential against fresh: {step}");
        }
        check(
            &mut ok,
            "second_request_equals_a_fresh_server",
            second_is_fresh,
        );

        let total = (2 * c.ctx).to_string();
        let (mut two, d2) = spawn_at(
            c,
            dir,
            "serve-two",
            &["--port", "0", "--ctx", &total, "--parallel", "2"],
            &budget_two,
        )?;
        let err_log = d2.join("server.err");
        let Some(addr) = listened(&mut two, &err_log, "two-slot")? else {
            check(&mut ok, NAME, false);
            return Ok(ok);
        };
        let url = |p: &str| format!("http://{addr}{p}");
        let records = server_log(&err_log, c.kinds)?;
        let load = records.one(c.load)?;
        let listen = records.one(c.listening)?;
        let planned = Planned::of(&records.one(&record::PLAN)?)?;
        let resident_two = load.u64("resident_bytes")?;
        let stderr = std::fs::read_to_string(&err_log).unwrap_or_default();
        let line = format!(
            "parallel rule=slots slots=2 slot_ctx={} total={} from=flag pass=turns",
            c.ctx,
            2 * c.ctx
        );
        if planned != one_plan {
            println!(
                "the two servers plan different splits: the two-slot server's {planned:?} against \
                 the one-slot server's {one_plan:?}; two splits answer other ids, so no id is \
                 compared; two-slot load {:?}",
                load.line()
            );
            check(&mut ok, NAME, false);
            println!("two-slot server stopped: {}", two.stop()?);
            return Ok(ok);
        }
        let past = i128::from(resident_two) - i128::from(resident_one) - i128::from(store_bytes);
        let loaded = one_shaped
            && load.u64("slots")? == 2
            && load.u64("ctx")? == c.ctx
            && listen.u64("slots")? == 2
            && listen.u64("ctx")? == c.ctx
            && stderr.lines().any(|l| l == line)
            && past == 0;
        println!(
            "two-slot server at {addr}; plan {planned:?}, the one-slot server's split; load {:?}; \
             listening {:?}; parallel line {line:?}; resident {resident_two} B against the \
             one-slot server's {resident_one} B: past one sequence's stores {store_bytes} B, \
             {past} B (want 0): {loaded}",
            load.line(),
            listen.line()
        );
        // Both posted at once, each asking its tokens' two best logits.
        let ran = together(
            &addr,
            ("colours", &colours.prompt),
            ("animals", &animals.prompt),
            |url, name, prompt| greedy_probs(url, &d2, name, prompt),
        )?;
        let equal = !ran[0].tokens.is_empty()
            && ran[0].tokens == colours.tokens
            && ran[0].stop == colours.stop
            && !ran[1].tokens.is_empty()
            && ran[1].tokens == animals.tokens
            && ran[1].stop == animals.stop;
        let apart = matches!((ran[0].slot, ran[1].slot), (Some(a), Some(b)) if a != b);
        println!(
            "together: colours on slot {:?} (held {:.1} ms) and animals on slot {:?} (held {:.1} \
             ms); ids equal the one-slot server's {equal}; on different slots {apart}",
            ran[0].slot, ran[0].held_ms, ran[1].slot, ran[1].held_ms
        );
        if let Some(step) = first_step_apart(("together", &ran[1]), ("sequential", &animals)) {
            println!("animals together against sequential: {step}");
        }
        let swaps = metric(&url, "swaps_total")?;
        println!("swaps_total {swaps:?}");
        check(
            &mut ok,
            NAME,
            loaded && equal && apart && swaps.is_none_or(|v| v == 0.0),
        );
        println!("two-slot server stopped: {}", two.stop()?);
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let c = clauses()?;
        let dir = serve_gate::dir_of(&c)?;
        std::fs::create_dir_all(&dir)?;
        let two = gpu_census::census()?.len() >= 2;
        let mut ok = slots_total_is_planned(&c, &dir)?;
        ok &= place_bp_follows_the_cards(&c, &dir, two)?;
        ok &= residency_default_is_the_machine(&c, &dir)?;
        ok &= serve_gate::served(&c, &dir)?;
        ok &= two_slots_decode_as_one_slot_runs(&c, &dir)?;
        finish(&c, ok)
    }
}
