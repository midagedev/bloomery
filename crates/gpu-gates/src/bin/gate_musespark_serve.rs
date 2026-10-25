//! `gate_musespark_serve` — the musespark seat of `bloomery-serve` against the model it serves,
//! on the muse-spark file of the profile (`BLOOMERY_REF_MODEL`; the fixture under the fixture
//! tier): the seat starts, loads the file, refuses what its body cannot serve and answers one chat
//! request whose reasoning is split out of the recipient protocol and whose turn ends at the
//! model's own end token.
//!
//!     gate_musespark_serve --dir <dir>
//!
//! The seat serves one sequence (`--parallel` above 1 is refused by name) on one card, or on plan
//! (b′) where the census holds two (the A6000 the stage, the 3090 an expert tier), runs the
//! common residency machine over the routed experts its plan placed, streams the host experts
//! of a prompt call by the common rule (`BLOOMERY_XSTREAM`) and serves no draft: the file holds
//! none. Every server is `bloomery-serve --model musespark -m <file> …` beside this binary,
//! `BLOOMERY_REF_MODEL` removed from its environment. Logs per server in `<dir>/<name>/`
//! (`server.err`, the answer).
//!
//! The model writes no think tag. A turn is `to=<recipient><|message|>` messages: the generation
//! prompt ends `<|start|>assistant`, the model writes ` to=self<|message|>` its reasoning
//! `<|eom|>`, then ` to=user<|message|>` its answer `<|eot|>`, and the server's recipient scan
//! (`serve::museatem`) puts the `self` message in `reasoning_content` and the rest in `content`.
//! The shared clauses (`shared/serve_gate.rs`) are read with the protocol's markers where they
//! name a think span's tags ([`MARKERS`]): neither may be left in the message.
//!
//! Without a load:
//!
//! - `plan_names_one_slot`: `--ctx` [`CTX`] `--plan` exits 0 and its stderr carries `parallel
//!   rule=slots slots=1 slot_ctx=<CTX> total=<CTX> from=ctx pass=one`. FAIL-first: a seat that
//!   plans a second slot, or prints `pass=turns`, misses the line and the clause is red.
//! - `place_bp_follows_the_cards`: `--place bp --plan` where the census holds two cards exits 0
//!   and prints a `plan` record whose devices name the tier card (`tier0`), or, where the stage
//!   holds every routed expert (the fixture), exits non-zero naming the tier card that would hold
//!   none ([`EMPTY_TIER`]) with no `plan` record; where it holds one card, it exits non-zero, its
//!   stderr names the place word the placement refuses, and no `plan` record is printed.
//!   FAIL-first: a seat that plans its one-card machine under `bp` prints no tier card and no
//!   refusal, and the clause is red on two cards.
//! - `place_unset_follows_the_cards`: the `plan_names_one_slot` run's `place unset` record (no
//!   `--place`) where the census holds two cards carries the tier experts the plan of `bp` holds
//!   and the family's break-even and basis (`TIER_RULE`: 845 experts, the tier size whose host
//!   time saved covers the tier's fixed cost, docs/cards/musespark-2card.card), and names `bp`
//!   exactly when the tier holds at least the break-even (the real file's tier holds the 3090's
//!   room, ~1,820); where the stage holds every expert (the fixture) the plan of `bp` is refused,
//!   the record carries no tier experts and names `a` for that refusal; where the census holds
//!   one card it does not name `bp`. FAIL-first: a seat that keeps the untiered rule
//!   (`Place::choose_untiered`) never asks the plan, prints no tier experts and the clause is red
//!   on two cards.
//! - `residency_default_is_the_machine`: `--plan` at [`CTX`] with `BLOOMERY_RESIDENCY` removed
//!   from the environment prints the `residency lever` record ahead of the plan with a why the
//!   plan resolved (not `set`, not `fixed_place`): the serving word
//!   (`bloomery_levers::RESIDENCY_SERVING`) where it prints `place`, `off` where the plan left no
//!   room for it or holds no host expert to move onto the card, and the `residency host` record of
//!   the churn pool after the plan exactly where the word is not `off`. `--place gate` prints `off`
//!   with why `fixed_place` and no pool. `BLOOMERY_RESIDENCY=off` prints `off` with why `set` and
//!   no pool. FAIL-first: a seat whose residency returns no pick prints no lever and the clause is
//!   red; a seat that treats `gate`
//!   as a serving placement prints the serving word and the gate half is red; a seat that omits
//!   the lever from the ones it acts on refuses `off` by name and the off half is red.
//! - `stream_ring_is_planned`: `--place a --plan` at [`CTX`] three ways: `BLOOMERY_XSTREAM`
//!   removed, `off` and `admit`. The `off` and `admit` plans are the one plan (no ring kept out of
//!   the expert budget), and the unset plan holds fewer card experts than they do where they
//!   leave host experts, and never more (the common rule starts the stream on plan (a) with a
//!   residency, `shared/xstream_rule.rs`, and the plan keeps its ring's bytes out of the expert
//!   budget). A fixture's plan holds every expert on the card whatever the ring, so there the
//!   clause reads the three plans equal. FAIL-first: a seat that does not plan its ring prints
//!   the unset plan of the `off` plan's experts and the clause is red where host experts exist;
//!   a seat that plans the ring under `admit` prints three plans of one kind and the pair is
//!   red.
//! - `xstream_line_follows_the_rule`: after the server of `load_and_listen` below has run, its
//!   stderr carries one `xstream=` line, and the line is the rule's for the plan it loaded: `admit`
//!   where the placement holds an expert tier card, `split` where the plan holds host experts on
//!   plan (a), `off` beside no residency where it holds none (the unset residency is `off` with
//!   why `no_host_experts`, which the `residency lever` record names: the machine has no host
//!   expert to move onto the card, so the rule's residency-off branch ends the line). FAIL-first:
//!   a seat that resolves no word prints no line and the clause is red; a seat whose residency
//!   machine runs over the all-card plan resolves `mid-p0-s1` and the clause is red on the
//!   lever and on the line.
//! - `parallel_above_one_is_refused` (shared): `--parallel 2 --plan` exits non-zero, its stderr
//!   names the body's one sequence, and no `plan` record is printed. FAIL-first: a seat that takes
//!   a second slot (`Tiered::RESIDENT`) prints its plan and the clause is red.
//!
//! One server as a user starts it, `--port 0` and `--ctx` [`CTX`] (shared):
//!
//! - `load_and_listen`: it listens; the `load` record comes before the `listening` record; the
//!   load names `muse-spark`, one slot, the file's layer count and [`CTX`]; the listening record
//!   names one slot and [`CTX`]; `/health` answers 200. FAIL-first: a seat that loads another
//!   context than the one it was asked for names it in the load record and the clause is red. The
//!   `load` record's `prompt` word (`batch`: the prompt's passes through every layer) is printed,
//!   not pinned.
//! - `chat_answers_a_turn`: `/v1/chat/completions` of one user turn at temperature 0,
//!   `reasoning_effort` `low` and `max_tokens` 512 is a 200 whose `usage` counts a prompt and at
//!   most 512 completion tokens, whose message holds neither marker of the protocol, and whose
//!   `finish_reason` is `stop` or `length`, on either tier. FAIL-first: a server that scans the
//!   reply only when the request carries tools leaves ` to=self<|message|>…` in `content`, the
//!   marker is in the message and the clause is red (`serve`'s
//!   `hw_musespark_reply_splits_at_the_recipient_span` pins the split on any machine).
//! - `chat_splits_the_reasoning_at_the_models_tags`: the same answer holds content and no marker,
//!   and reasoning where the model wrote a `to=self` message: at the low effort the model may
//!   answer with none, so the clause does not require it and the answer's reasoning length is in
//!   the printed line. PIN(2026-10-23): the fixture cut keeps four of the model's layers, whose
//!   weights write no protocol, so this and the next clause are `Tag::FileBound` and the fixture
//!   tier leaves them to the real tier by name.
//! - `chat_ends_at_the_turn_end`: the same answer finishes with `stop`, and no message holds
//!   `<|eot|>`: the model closes its turn with that token, an end-of-generation id by name, so
//!   the server stops there and leaves it out of the text. FAIL-first: a vocabulary that lists
//!   only the file's `eos_token_id` (200001) lets the model run past its turn end, the token's
//!   text is in the message and the clause is red.
//! - `live_slot_keeps_the_prefix` (shared, `shared/serve_gate.rs`): a request keeps the prefix
//!   it shares with the slot the server holds — the seat's cut keeps every held position
//!   (`app::Keep`, crates/app/src/arch/musespark/mod.rs:81), so the shared prefix of another
//!   question on the same text is kept whole and its cached answer equals the uncached one on
//!   a server whose residency is off. FAIL-first: a seat whose keep query grants nothing keeps
//!   none of the shared prefix and the clause is red.

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_musespark_serve", gate::run())
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
        SeatGate, check, no_residency_host, residency_lever_is, residency_lever_resolved,
        server_log,
    };
    use bloomery_gpu_gates::{GateError, checks_failed, gpu_census, ref_model_path};
    use gguf::Split;
    use model::arch::musespark::place::PlanInputs;

    use crate::serve_gate;

    /// The slot's positions the server is asked for: past every prompt and answer here, and past
    /// the sliding window's 2048.
    const CTX: u64 = 4096;

    /// The two markers of the recipient protocol the message must not carry, read where the
    /// shared clauses name a think span's tags.
    const MARKERS: [&str; 2] = ["<|message|>", "<|eom|>"];

    /// The token that closes the model's turn, as its vocabulary spells it.
    const TURN_END: &str = "<|eot|>";

    /// The words of the refusal of the placement: the place word it refuses.
    const PLACED: &str = "--place bp";

    /// The chat request: no reasoning budget (no span to close), the lowest effort the template's
    /// map holds, and a message that need not carry reasoning.
    const ASK: serve_gate::Ask = serve_gate::Ask {
        budget: None,
        effort: Some("low"),
        reasons: false,
    };

    /// The seat's facts, as the shared clauses read them.
    fn clauses() -> Result<serve_gate::Clauses, GateError> {
        let file = ref_model_path()?;
        let split = Split::open(&file).map_err(|e| format!("open {}: {e}", file.display()))?;
        let layers = u64::try_from(PlanInputs::read(&split)?.model.layers)?;
        Ok(serve_gate::Clauses {
            gate: "gate_musespark_serve",
            model: "musespark",
            kinds: record::BLOOMERY_SERVE_MUSESPARK,
            load: &record::LOAD_MUSESPARK,
            listening: &record::LISTENING_MUSESPARK,
            arch: "muse-spark",
            layers,
            think: MARKERS,
            turn_end: TURN_END,
            ctx: CTX,
            // Every layer's store is a plane of positions, so the session's cut keeps every
            // held position (`app::Keep for Body`, crates/app/src/arch/musespark/mod.rs:81).
            cut: serve_gate::Cut::EveryPosition,
        })
    }

    fn seat(c: &serve_gate::Clauses) -> SeatGate {
        SeatGate {
            model: c.model,
            kinds: c.kinds,
        }
    }

    /// What the placement's refusal of a tier card with no expert says ([`place_bp_follows_the_cards`]).
    const EMPTY_TIER: &str = "holds no expert: the cards before it hold every routed expert";
    /// What the unset rule's why says when the plan of `bp` was refused.
    const TIER_REFUSED: &str = "the tier's plan refused";

    /// `plan_names_one_slot` and `place_unset_follows_the_cards` (the module header), on one
    /// `--plan` run; `two` whether the census holds two cards.
    fn plan_names_one_slot(
        c: &serve_gate::Clauses,
        dir: &Path,
        two: bool,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let file = ref_model_path()?;
        let ctx = c.ctx.to_string();
        let (status, log, text) = seat(c).plan(dir, "plan-one-slot", &file, &["--ctx", &ctx])?;
        let want = format!(
            "parallel rule=slots slots=1 slot_ctx={} total={} from=ctx pass=one",
            c.ctx, c.ctx
        );
        let one = status.success() && text.lines().any(|l| l == want);
        println!("ctx {}: the parallel line {want:?} {one}", c.ctx);
        if !one {
            println!("plan-one-slot stderr:\n{text}");
        }
        check(&mut ok, "plan_names_one_slot", one);
        let unset = log.one(&record::PLACE_UNSET)?;
        // The family's rule as the seat names it (`TIER_RULE`): the tier experts that keep the tier
        // card, and the card file that derives them.
        const BREAK_EVEN: u64 = 845;
        const BASIS: &str = "docs/cards/musespark-2card.card";
        let (place, tier) = (unset.word("place")?, unset.opt_u64("tier_experts")?);
        let (even, basis) = (unset.opt_u64("break_even")?, unset.opt_word("basis")?);
        // On two cards the rule asked the plan of bp for its tier: kept at the break-even or more;
        // the fixture, whose stage holds every expert, has that plan refused and runs `a`.
        let why = unset.text("why")?;
        let follows = if two {
            let ruled = match tier {
                Some(t) => (place == "bp") == (t >= BREAK_EVEN),
                None => place == "a" && why.contains(TIER_REFUSED),
            };
            ruled && even == Some(BREAK_EVEN) && basis == Some(BASIS)
        } else {
            place != "bp"
        };
        println!(
            "place unset on {} card(s): place={place} tier_experts={tier:?} break_even={even:?} \
             basis={basis:?} why={why:?} {follows}",
            if two { 2 } else { 1 },
        );
        check(&mut ok, "place_unset_follows_the_cards", follows);
        Ok(ok)
    }

    /// `place_bp_follows_the_cards` (the module header); `two` whether the census holds two cards.
    fn place_bp_follows_the_cards(
        c: &serve_gate::Clauses,
        dir: &Path,
        two: bool,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let file = ref_model_path()?;
        let (status, log, text) = seat(c).plan(dir, "place-bp", &file, &["--place", "bp"])?;
        let plan = log.first(&record::PLAN)?;
        let follows = if two {
            let devices = match &plan {
                Some(p) => p.csv("devices")?.iter().map(|d| (*d).to_owned()).collect(),
                None => Vec::new(),
            };
            let tiered = devices.iter().any(|d| d.starts_with("tier0"));
            let empty = !status.success() && text.contains(EMPTY_TIER) && plan.is_none();
            println!(
                "place-bp on two cards: exit 0 {}; plan devices {devices:?}, a tier card {tiered}; \
                 refused as a tier with no expert {empty}",
                status.success()
            );
            (status.success() && tiered) || empty
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

    /// `--plan` of `args` with `BLOOMERY_RESIDENCY` set to `residency` and `BLOOMERY_XSTREAM` to
    /// `xstream`, each removed when it is `None` ([`SeatGate::plan_under`]).
    fn plan_under(
        seat: &SeatGate,
        dir: &Path,
        name: &str,
        file: &Path,
        args: &[&str],
        (residency, xstream): (Option<&str>, Option<&str>),
    ) -> Result<(ExitStatus, record::Log, String), GateError> {
        let levers = [
            (bloomery_levers::RESIDENCY, residency),
            (bloomery_levers::XSTREAM, xstream),
        ];
        seat.plan_under((dir, name), file, args, &levers)
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
        let sg = seat(c);
        let ctx = c.ctx.to_string();
        let (serving_status, serving, _) = plan_under(
            &sg,
            dir,
            "plan-residency",
            &file,
            &["--ctx", &ctx],
            (None, None),
        )?;
        let (gate_status, gate, _) = plan_under(
            &sg,
            dir,
            "plan-residency-gate",
            &file,
            &["--ctx", &ctx, "--place", "gate"],
            (None, None),
        )?;
        let (off_status, off, _) = plan_under(
            &sg,
            dir,
            "plan-residency-off",
            &file,
            &["--ctx", &ctx],
            (Some("off"), None),
        )?;
        let exits = serving_status.success() && gate_status.success() && off_status.success();
        let serving_ok = residency_lever_resolved(&serving)?;
        let gate_ok = residency_lever_is(&gate, "off", "fixed_place")? && no_residency_host(&gate)?;
        let off_ok = residency_lever_is(&off, "off", "set")? && no_residency_host(&off)?;
        println!(
            "residency default: exit 0 {exits}; unset lever {:?} is resolved, ahead of the plan, \
             with its churn pool after it exactly where it is not off {serving_ok}; --place gate \
             {:?} is off (fixed_place) with no pool {gate_ok}; set off {:?} is off (set) with no \
             pool {off_ok}",
            lever_line(&serving)?,
            lever_line(&gate)?,
            lever_line(&off)?
        );
        check(
            &mut ok,
            "residency_default_is_the_machine",
            exits && serving_ok && gate_ok && off_ok,
        );
        Ok(ok)
    }

    /// The card and host experts of a plan record.
    fn experts(log: &record::Log) -> Result<Option<(u64, u64)>, GateError> {
        Ok(match log.first(&record::PLAN)? {
            Some(p) => Some((p.u64("card_experts")?, p.u64("host_experts")?)),
            None => None,
        })
    }

    /// `stream_ring_is_planned` (the module header).
    fn stream_ring_is_planned(c: &serve_gate::Clauses, dir: &Path) -> Result<bool, GateError> {
        let mut ok = true;
        let file = ref_model_path()?;
        let sg = seat(c);
        let ctx = c.ctx.to_string();
        let args = ["--ctx", ctx.as_str(), "--place", "a"];
        let (unset_status, unset, _) =
            plan_under(&sg, dir, "plan-xstream", &file, &args, (None, None))?;
        let (off_status, off, _) = plan_under(
            &sg,
            dir,
            "plan-xstream-off",
            &file,
            &args,
            (None, Some("off")),
        )?;
        let (admit_status, admit, _) = plan_under(
            &sg,
            dir,
            "plan-xstream-admit",
            &file,
            &args,
            (None, Some("admit")),
        )?;
        let exits = unset_status.success() && off_status.success() && admit_status.success();
        let (Some((u_card, u_host)), Some((o_card, o_host)), Some((a_card, a_host))) =
            (experts(&unset)?, experts(&off)?, experts(&admit)?)
        else {
            println!("a --plan run printed no plan record FAIL");
            check(&mut ok, "stream_ring_is_planned", false);
            return Ok(ok);
        };
        let pair = (o_card, o_host) == (a_card, a_host);
        // The ring's bytes leave the expert budget: fewer card experts where the plan holds host
        // experts, never more, and the host holds what the card does not.
        let ring = u_card <= o_card && (o_host == 0 || u_card < o_card) && u_host >= o_host;
        println!(
            "xstream plans on --place a: unset card {u_card} host {u_host}; off card {o_card} host \
             {o_host}; admit card {a_card} host {a_host}; exit 0 {exits}; off = admit {pair}; \
             the unset plan keeps the ring out ({} host experts) {ring}",
            if o_host == 0 { "no" } else { "the plan holds" }
        );
        check(&mut ok, "stream_ring_is_planned", exits && pair && ring);
        Ok(ok)
    }

    /// `xstream_line_follows_the_rule` (the module header), on the stderr of the server
    /// [`serve_gate::served_with`] ran in `<dir>/serve`.
    fn xstream_line_follows_the_rule(
        c: &serve_gate::Clauses,
        dir: &Path,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let err = dir.join("serve").join("server.err");
        let text =
            std::fs::read_to_string(&err).map_err(|e| format!("read {}: {e}", err.display()))?;
        let line = text.lines().find(|l| l.starts_with("xstream="));
        let log = server_log(&err, c.kinds)?;
        let plan = log.first(&record::PLAN)?;
        let (Some(line), Some(plan)) = (line, plan) else {
            println!("the server's stderr holds no xstream= line or no plan record FAIL");
            check(&mut ok, "xstream_line_follows_the_rule", false);
            return Ok(ok);
        };
        let tiered = plan.csv("devices")?.iter().any(|d| d.starts_with("tier0"));
        let host = plan.u64("host_experts")?;
        let follows = if tiered {
            line.starts_with("xstream=admit")
        } else if host > 0 {
            line.starts_with("xstream=split")
        } else {
            let lever = log.first(&record::RESIDENCY_LEVER)?;
            let off_for_it = match &lever {
                Some(l) => l.word("residency")? == "off" && l.word("why")? == "no_host_experts",
                None => false,
            };
            off_for_it
                && line.starts_with("xstream=off")
                && line.ends_with("(unset: no residency or not --place a)")
        };
        println!(
            "the load's {line:?}: the plan holds {host} host experts{}: the rule's word {follows}",
            if tiered { " beside an expert tier" } else { "" }
        );
        check(&mut ok, "xstream_line_follows_the_rule", follows);
        Ok(ok)
    }

    /// The gate's end. The `PASSED` line names the clauses.
    fn finish(ok: bool) -> Result<(), GateError> {
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: gate_musespark_serve the seat plans one slot, plans the expert tier card where \
             two cards are in view and refuses it where one is, refuses a second slot, runs the \
             residency machine and plans the stream's ring by the common rule, loads at the \
             context it was asked for, answers a \
             chat turn with its reasoning split out of the recipient protocol and its turn ended \
             at the model's own end token, and keeps the prefix a request shares with the slot \
             (live_slot_keeps_the_prefix)"
        );
        Ok(())
    }

    pub fn run() -> Result<(), GateError> {
        let c = clauses()?;
        let dir = serve_gate::dir_of(&c)?;
        std::fs::create_dir_all(&dir)?;
        let two = gpu_census::census()?.len() >= 2;
        let mut ok = plan_names_one_slot(&c, &dir, two)?;
        ok &= place_bp_follows_the_cards(&c, &dir, two)?;
        ok &= residency_default_is_the_machine(&c, &dir)?;
        ok &= stream_ring_is_planned(&c, &dir)?;
        ok &= serve_gate::parallel_above_one_is_refused(&c, &dir)?;
        ok &= serve_gate::served_with(&c, &dir, ASK)?;
        ok &= xstream_line_follows_the_rule(&c, &dir)?;
        finish(ok)
    }
}
