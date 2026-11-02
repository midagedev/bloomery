//! `gate_mistrallarge4_serve` — the mistrallarge4 seat of `bloomery-serve` against the model it
//! serves, on the mistral-large4 file of the profile (`BLOOMERY_REF_MODEL`; the fixture under the
//! fixture tier): the seat plans its slots and its expert tier card, refuses what its body cannot
//! serve, loads the file, answers a chat turn with its reasoning split at the file's `[THINK]`
//! tags, keeps each resident sequence's rows apart, saves and restores a slot's state, parses a
//! tool call by the file's template and answers an image request.
//!
//!     gate_mistrallarge4_serve --dir <dir> [--mmproj <file>]
//!
//! `--mmproj` is the Pixtral projector of the file's repo; without it the image clauses are
//! skipped by name. Every server is `bloomery-serve --model mistrallarge4 -m <file> …` beside this
//! binary, `BLOOMERY_REF_MODEL` removed from its environment. Logs per server in `<dir>/<name>/`
//! (`server.err`, the answers).
//!
//! Without a load:
//!
//! - `plan_names_the_slots`: `--ctx` 2·[`CTX`] `--parallel 2 --plan` exits 0 and its stderr carries
//!   `parallel rule=slots slots=2 slot_ctx=<CTX> total=<2·CTX> from=… pass=turns`, and `--ctx`
//!   [`CTX`] `--plan` carries `slots=1 … pass=one`. FAIL-first: a seat that serves several slots
//!   in one pass (`Tiered::ONE_PASS`) prints `pass=one` for two and the clause is red.
//! - `place_bp_follows_the_cards`: `--place bp --plan` where the census holds two cards exits 0 and
//!   prints a `plan` record whose devices name the tier card (`tier0`), or, where the stage holds
//!   every routed expert (the fixture), exits non-zero naming the tier card that would hold none
//!   ([`EMPTY_TIER`]) with no `plan` record; where it holds one card, it exits non-zero, its stderr
//!   names the place word the placement refuses, and no `plan` record is printed. FAIL-first: a
//!   seat whose rule serves no tier card (`TierRule::UNTIERED`) refuses `bp` on two cards by name
//!   and the clause is red.
//! - `place_unset_runs_a_until_a_sitting_decides`: the `place unset` record (no `--place`) carries
//!   no break-even where the census holds two cards, and names `a`: no sitting has shown the
//!   tier not slower; where it holds one card it does not name `bp`. FAIL-first: a rule with a
//!   break-even prints it and the clause is red on two cards.
//! - `residency_lever_is_honoured`: `--plan` at [`CTX`] with `BLOOMERY_RESIDENCY=mid-p0-s1` exits 0
//!   and prints the `residency lever` record at that word with why `set`, ahead of the `plan`
//!   record, and the churn pool's `residency host` record after it; with `BLOOMERY_RESIDENCY=off`
//!   it prints `off` with why `set` and no pool. With the lever removed from the environment it
//!   prints a word the plan resolved (its why neither `set` nor `fixed_place`; the serving word
//!   where it is `place`, `off` where the plan left no room or no host expert), ahead of the plan,
//!   with the pool after the plan exactly where the word is not `off`
//!   (`serve_client::residency_lever_resolved`); `--place gate` prints `off` with why
//!   `fixed_place` and no pool. FAIL-first: a seat whose prompt feed is steps
//!   (`Tiered::prefill_steps` true) refuses the `mid-p0-s1` word by name beside `--prefill steps`,
//!   exits non-zero with no `plan` record, and the clause is red.
//! - `xstream_lever_is_honoured`: `--place a --plan` at [`CTX`] with `BLOOMERY_XSTREAM` removed,
//!   `off`, `admit` and `split`, each exiting 0 with a `plan` record. The `off` and `admit` plans
//!   are the one plan (no ring kept out of the expert budget); the `split` plan, and the unset one
//!   (the common rule starts the stream at `split` on plan (a) with a residency,
//!   `shared/xstream_rule.rs`), hold fewer card experts than they do where the plan leaves host
//!   experts, never more, and the host holds what the card does not. A fixture's plan holds every
//!   expert on the card whatever the ring, so there the four plans read equal. Beside an expert
//!   tier card (`--place bp`) `admit` and `split` plan too: where the census holds two cards, a
//!   `plan` record whose devices name the tier card or, where the stage holds every routed expert
//!   (the fixture), the refusal of the tier card that would hold none ([`EMPTY_TIER`]); where it
//!   holds one card, the placement's refusal. FAIL-first: a seat that leaves the stream out of its
//!   levers refuses `off` by name, exits non-zero with no `plan` record and the clause is red; a
//!   seat that plans no ring prints the `split` plan of the `off` plan's experts and the clause is
//!   red where the plan leaves host experts.
//! - `mmproj_is_held_before_the_plan`: `--mmproj <missing file> --plan` exits non-zero naming the
//!   file, and `--image-max-tokens 64 --plan` (no projector to bound) exits non-zero naming the
//!   flag, each with no `plan` record. FAIL-first: a seat that takes no image flag is read as an
//!   unknown argument, which does not name the file, and the clause is red.
//!
//! One server as a user starts it, `--port 0` and `--ctx` [`CTX`] (shared, `shared/serve_gate.rs`):
//! `load_and_listen`, `chat_answers_a_turn`, `chat_splits_the_reasoning_at_the_models_tags` and
//! `chat_ends_at_the_turn_end` (the last two are `Tag::FileBound`), `live_slot_keeps_the_prefix` and
//! `saved_state_returns`, with the `Ask` of a think-span model (the template's default effort, the
//! reasoning budget); the message holds neither of the file's reasoning tags and the turn ends at
//! `</s>`. FAIL-first of the tags: a seat that leaves the file's `[THINK]` pair to the server's
//! `<think>` default leaves the model's span in the message and the split clause is red. This
//! server runs at the residency default, so where the plan keeps the machine the same-answer
//! halves of the last two clauses are skipped by name (they run only where the server's
//! `residency lever` record says `off`). Its stderr is read for one more clause:
//!
//! - `xstream_line_follows_the_rule`: the server's stderr carries one `xstream=` line, and the
//!   line is the rule's for the plan it loaded: `off` where its `residency lever` record is
//!   `off`, else `admit` where the placement holds an expert tier card, `split` where the plan
//!   holds host experts on plan (a), and `off` where it holds none. FAIL-first: a seat that never
//!   resolves the stream prints no line and the clause is red.
//! - `load_holds_the_plan`: the server's `load` record names the card and host experts of its
//!   `plan` record, where the plan holds no expert tier card: the run's plan and the open's are
//!   one, over the same inputs (the prompt group's reserve and the ring's). FAIL-first: a run that
//!   plans the ring on inputs priced for another prompt group than the open's prints a plan the
//!   load does not hold, and the clause is red.
//!
//! One server of two slots (`--parallel 2`, `--ctx` 2·[`CTX`], `--slot-save-path <dir>/save`),
//! with `BLOOMERY_RESIDENCY=off`: the residency machine moves experts between the tiers by the
//! requests' heat, so a request after another conversation, or on the other slot, would run on
//! another split than a fresh sequence, and a near tie answers other ids; and with
//! `BLOOMERY_PREFILL_GROUP=1`, which writes the bits of the default group (every group writes the
//! same), so the slot clauses compare as before:
//!
//! - `two_slots_load`: the `load` and `listening` records name two slots and a slot's [`CTX`].
//!   FAIL-first: a seat that prints one slot in its load record is red.
//! - `prefill_group_is_honoured`: the `load` record's `group` of the server above (the lever
//!   removed from its environment) is the row's default ([`bloomery_levers::PREFILL_GROUP_DEFAULT`],
//!   2) and this server's is 1: the group the body holds after the open (`Body::prefill_group`),
//!   not the one the seat was asked for. FAIL-first: a seat that does not set the body's group
//!   after the capture leaves it at the 1 the load made, and the default server's record reads 1.
//! - `slots_keep_their_own_rows`: prompt P, then P plus its answer on P's slot with no request
//!   between (the truth), then an unrelated prompt Q on the other slot (`id_slot` differs); P plus
//!   its answer then returns to its slot keeping all but its last id (`timings.cache_n`) and
//!   answers the truth's ids, which the same calls over the same rows write bit for bit. FAIL-first:
//!   a seat whose select exchanges nothing (`Slots::swap_seq`) lets Q write over P's rows and the
//!   continuation's ids differ.
//! - `slot_file_restores_equal_bits`: a prompt on a slot, its state saved to a file, the prompt
//!   plus its answer on that slot (the truth: the rows the file holds, then the same calls), that
//!   slot erased and the file restored into the other slot (`n_restored` the positions saved; the
//!   other slot holds other prompts' rows), then the prompt plus its answer, which that other slot
//!   takes (`id_slot`), keeps all but its last id and answers the truth's ids. FAIL-first: a save
//!   that leaves a layer's rows out restores stale rows into the other slot and the ids differ.
//!
//! Both clauses also run the continuation uncached and, in the real tier, want its ids. The prompt
//! batch is the steps' within a band and not bit for bit (`gate_mistrallarge4_prefill`), so an
//! uncached request writes other bits than the steps that answered before it; the real file's
//! top-1 margins hold its greedy ids over that band, and the fixture's near ties do not, so the
//! fixture tier prints those ids and judges the truth's alone.
//! - `tool_calls_follow_the_template` (`Tag::FileBound`: the file's template teaches the markup
//!   and its model writes it): a chat request with a `get_weather` tool is a 200 whose `finish_reason`
//!   is `tool_calls` with one `get_weather` call whose `city` argument is a string. FAIL-first: a
//!   server that cannot read the template's `[TOOL_CALLS]` markup answers 501 and the clause is red.
//! - `image_request_is_served` (with `--mmproj`; `Tag::FileBound`: the projector is the real file's, its aligner rows the real
//!   embedding wide, and a server of the fixture's narrower one refuses it at the start, so the fixture tier starts the
//!   server without it and defers the clause): the `vision_seat` and `vision` records are
//!   printed, and a chat request holding a 448 × 448 image answers 200 with completion tokens whose
//!   prompt counts at least the image's [`IMAGE_ROWS`] rows (16 × 16 tokens and the 15 break rows
//!   between the grid rows). The span's rows are staged by the batched walk in place of their
//!   positions' embeddings, one residency pass a call; the clause's line prints the request's
//!   `prompt_ms`. FAIL-first: an encode that keeps the cells' rows alone refuses the span
//!   (`kinds.len() != span_len`) and the clause is red; a walk that stages one row fewer than the
//!   span holds ends the call with a named error (`a call that staged … media rows`) and the
//!   clause is red.

#[cfg(all(feature = "deepseek41", feature = "mistrallarge4"))]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_mistrallarge4_serve", gate::run())
}

#[cfg(not(all(feature = "deepseek41", feature = "mistrallarge4")))]
fn main() -> std::process::ExitCode {
    eprintln!("gate_mistrallarge4_serve needs the deepseek41 and mistrallarge4 features");
    std::process::ExitCode::from(64)
}

#[cfg(all(feature = "deepseek41", feature = "mistrallarge4"))]
#[allow(
    dead_code,
    reason = "the shared clauses serve several gates; each takes the helpers it reads"
)]
#[path = "shared/serve_gate.rs"]
mod serve_gate;

#[cfg(all(feature = "deepseek41", feature = "mistrallarge4"))]
mod gate {
    use std::path::{Path, PathBuf};
    use std::process::ExitStatus;
    use std::time::Duration;

    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::serve_client::{
        SeatGate, Served, base64, chatted, check, curl, greedy, no_residency_host,
        residency_lever_is, residency_lever_resolved, server_log, slot_action, tokenized,
        tool_chat,
    };
    use bloomery_gpu_gates::tier::{self, Tag};
    use bloomery_gpu_gates::{GateError, checks_failed, gpu_census, ref_model_path};
    use gguf::Split;
    use model::arch::mistrallarge4::place::PlanInputs;
    use serde_json::{Value, json};

    use crate::serve_gate;

    /// A slot's positions the server is asked for: past every prompt and answer here.
    const CTX: u64 = 4096;

    /// The two tags of the file's reasoning span, which the message must not carry.
    const THINK: [&str; 2] = ["[THINK]", "[/THINK]"];

    /// The token that closes the model's turn, as its vocabulary spells it.
    const TURN_END: &str = "</s>";

    /// The words of the refusal of the placement: the place word it refuses.
    const PLACED: &str = "--place bp";

    /// What the placement's refusal of a tier card with no expert says.
    const EMPTY_TIER: &str = "holds no expert: the cards before it hold every routed expert";

    /// A residency word that fits a plan whose fewest card experts a layer is two or more: no
    /// pinned expert, one spare and one that moves.
    const MID: &str = "mid-p0-s1";

    /// The chat request: the template's default effort (`high`, which reasons in a `[THINK]` span)
    /// with the reasoning budget a think-span model's request carries, so the span closes inside
    /// the answer's tokens; the message need not carry reasoning, but may carry no think tag.
    const ASK: serve_gate::Ask = serve_gate::Ask {
        reasons: false,
        ..serve_gate::Ask::THINK_SPAN
    };

    /// Polls of a second that bound the wait for a server to listen: a cold load of the whole
    /// host set.
    const LISTEN_POLLS: usize = 1000;

    /// The tokens the slot clauses' requests answer.
    const N: usize = 8;

    /// The prompts of the slot clauses, three conversations that share no first id.
    const P_TEXT: &str = "The capital of France is Paris, and the capital of Italy is";
    const Q_TEXT: &str = "Zebras run across the open plain while the lions watch from the";
    const R_TEXT: &str = "Water boils at one hundred degrees Celsius at sea level, but on a";

    /// The tool-call request's reply tokens and the think span's share of them.
    const TOOL_TOKENS: u64 = 192;
    const TOOL_THINK_BUDGET: u64 = 16;

    /// The slot file the save and restore clause writes under `--slot-save-path`.
    const SLOT_FILE: &str = "ml4-slot.bin";

    /// The checked-in 448 × 448 image of the image clause, from this crate's root, and the rows
    /// its span takes: 16 × 16 merged tokens and the 15 break rows between its 16 grid rows.
    const IMAGE: &str = "../../tools/ref/clefvis/images/free-448x448.png";
    const IMAGE_ROWS: u64 = 16 * 16 + 15;
    const IMAGE_ASK: &str = "Describe this image in one short sentence.";

    /// The gate's flags.
    struct Args {
        dir: PathBuf,
        mmproj: Option<PathBuf>,
    }

    fn args() -> Result<Args, GateError> {
        const USAGE: &str = "usage: gate_mistrallarge4_serve --dir <dir> [--mmproj <file>]";
        let (mut dir, mut mmproj) = (None, None);
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} needs a value: {USAGE}"))?;
            match flag.as_str() {
                "--dir" => dir = Some(PathBuf::from(v)),
                "--mmproj" => mmproj = Some(PathBuf::from(v)),
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        Ok(Args {
            dir: dir.ok_or(USAGE)?,
            mmproj,
        })
    }

    /// The seat's facts, as the shared clauses read them.
    fn clauses() -> Result<serve_gate::Clauses, GateError> {
        let file = ref_model_path()?;
        let split = Split::open(&file).map_err(|e| format!("open {}: {e}", file.display()))?;
        let layers = u64::try_from(PlanInputs::describe(&split)?.model.layers)?;
        Ok(serve_gate::Clauses {
            gate: "gate_mistrallarge4_serve",
            model: "mistrallarge4",
            kinds: record::BLOOMERY_SERVE_MISTRALLARGE4,
            load: &record::LOAD_MISTRALLARGE4,
            listening: &record::LISTENING_MISTRALLARGE4,
            arch: "mistral-large4",
            layers,
            think: THINK,
            turn_end: TURN_END,
            ctx: CTX,
            // Every layer's store is a plane of positions, so the session's cut keeps every held
            // position (`app::Keep for Body`, crates/app/src/arch/mistrallarge4/mod.rs).
            cut: serve_gate::Cut::EveryPosition,
        })
    }

    fn seat(c: &serve_gate::Clauses) -> SeatGate {
        SeatGate {
            model: c.model,
            kinds: c.kinds,
        }
    }

    /// `plan_names_the_slots` (the module header).
    fn plan_names_the_slots(c: &serve_gate::Clauses, dir: &Path) -> Result<bool, GateError> {
        let mut ok = true;
        let file = ref_model_path()?;
        let total = (2 * c.ctx).to_string();
        let (status, _, text) = seat(c).plan(
            dir,
            "plan-two-slots",
            &file,
            &["--ctx", &total, "--parallel", "2"],
        )?;
        let two = status.success()
            && text.lines().any(|l| {
                l.starts_with(&format!(
                    "parallel rule=slots slots=2 slot_ctx={} total={total} ",
                    c.ctx
                )) && l.ends_with(" pass=turns")
            });
        let ctx = c.ctx.to_string();
        let (status, _, text) = seat(c).plan(dir, "plan-one-slot", &file, &["--ctx", &ctx])?;
        let one = status.success()
            && text.lines().any(|l| {
                l.starts_with(&format!(
                    "parallel rule=slots slots=1 slot_ctx={} total={} ",
                    c.ctx, c.ctx
                )) && l.ends_with(" pass=one")
            });
        println!(
            "ctx {total} --parallel 2: two slots, pass=turns {two}; ctx {ctx}: one slot, pass=one {one}"
        );
        if !(two && one) {
            println!("plan-two-slots stderr:\n{text}");
        }
        check(&mut ok, "plan_names_the_slots", two && one);
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

    /// `place_unset_runs_a_until_a_sitting_decides` (the module header).
    fn place_unset_runs_a(
        c: &serve_gate::Clauses,
        dir: &Path,
        two: bool,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let file = ref_model_path()?;
        let (_, log, _) = seat(c).plan(dir, "place-unset", &file, &[])?;
        let unset = log.one(&record::PLACE_UNSET)?;
        let (place, even) = (unset.word("place")?, unset.opt_u64("break_even")?);
        let follows = if two {
            place == "a" && even.is_none()
        } else {
            place != "bp"
        };
        println!(
            "place unset on {} card(s): place={place} break_even={even:?} why={:?} {follows}",
            if two { 2 } else { 1 },
            unset.text("why")?
        );
        check(
            &mut ok,
            "place_unset_runs_a_until_a_sitting_decides",
            follows,
        );
        Ok(ok)
    }

    /// `--plan` of `args` with `BLOOMERY_RESIDENCY` set to `word`, or removed from the environment
    /// when it is `None` ([`SeatGate::plan_under`]).
    fn plan_residency(
        c: &serve_gate::Clauses,
        dir: &Path,
        name: &str,
        args: &[&str],
        word: Option<&str>,
    ) -> Result<(ExitStatus, record::Log, String), GateError> {
        let file = ref_model_path()?;
        seat(c).plan_under(
            (dir, name),
            &file,
            args,
            &[(bloomery_levers::RESIDENCY, word)],
        )
    }

    /// The `residency lever` record, then the `plan` record, then the churn pool's `residency
    /// host` record.
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

    /// The rendered `residency lever` line, or a named absence.
    fn lever_line(log: &record::Log) -> Result<String, GateError> {
        Ok(match log.first(&record::RESIDENCY_LEVER)? {
            Some(l) => l.line().to_owned(),
            None => "(no residency lever record)".to_owned(),
        })
    }

    /// `residency_lever_is_honoured` (the module header).
    fn residency_lever_is_honoured(c: &serve_gate::Clauses, dir: &Path) -> Result<bool, GateError> {
        let mut ok = true;
        let ctx = c.ctx.to_string();
        let args = ["--ctx", ctx.as_str()];
        let (mid_status, mid, _) = plan_residency(c, dir, "plan-residency-mid", &args, Some(MID))?;
        let (off_status, off, _) =
            plan_residency(c, dir, "plan-residency-off", &args, Some("off"))?;
        let (unset_status, unset, _) = plan_residency(c, dir, "plan-residency", &args, None)?;
        let (gate_status, gate, _) = plan_residency(
            c,
            dir,
            "plan-residency-gate",
            &["--ctx", ctx.as_str(), "--place", "gate"],
            None,
        )?;
        let mid_ok =
            mid_status.success() && residency_lever_is(&mid, MID, "set")? && hosted_in_order(&mid)?;
        let off_ok = off_status.success()
            && residency_lever_is(&off, "off", "set")?
            && no_residency_host(&off)?;
        let unset_ok = unset_status.success() && residency_lever_resolved(&unset)?;
        let gate_ok = gate_status.success()
            && residency_lever_is(&gate, "off", "fixed_place")?
            && no_residency_host(&gate)?;
        println!(
            "residency: {MID} {:?} with the pool after the plan {mid_ok}; off {:?} with no pool \
             {off_ok}; unset {:?} resolved by the plan {unset_ok}; --place gate {:?} is off \
             (fixed_place) with no pool {gate_ok}",
            lever_line(&mid)?,
            lever_line(&off)?,
            lever_line(&unset)?,
            lever_line(&gate)?,
        );
        check(
            &mut ok,
            "residency_lever_is_honoured",
            mid_ok && off_ok && unset_ok && gate_ok,
        );
        Ok(ok)
    }

    /// `--plan` of `args` with `BLOOMERY_XSTREAM` set to `word`, or removed from the environment
    /// when it is `None` ([`SeatGate::plan_under`]).
    fn plan_xstream(
        c: &serve_gate::Clauses,
        dir: &Path,
        name: &str,
        args: &[&str],
        word: Option<&str>,
    ) -> Result<(ExitStatus, record::Log, String), GateError> {
        let file = ref_model_path()?;
        seat(c).plan_under(
            (dir, name),
            &file,
            args,
            &[(bloomery_levers::XSTREAM, word)],
        )
    }

    /// The card and host experts of a plan record.
    fn experts(log: &record::Log) -> Result<Option<(u64, u64)>, GateError> {
        Ok(match log.first(&record::PLAN)? {
            Some(p) => Some((p.u64("card_experts")?, p.u64("host_experts")?)),
            None => None,
        })
    }

    /// Whether a `--place bp --plan` run ended as the placement does: where the census holds two
    /// cards, a `plan` record whose devices name the tier card, or the refusal of a tier card that
    /// would hold no expert; where it holds one card, the refusal naming the place word.
    fn bp_planned(
        two: bool,
        status: ExitStatus,
        log: &record::Log,
        text: &str,
    ) -> Result<bool, GateError> {
        let plan = log.first(&record::PLAN)?;
        Ok(if two {
            let tiered = match &plan {
                Some(p) => p.csv("devices")?.iter().any(|d| d.starts_with("tier0")),
                None => false,
            };
            (status.success() && tiered)
                || (!status.success() && text.contains(EMPTY_TIER) && plan.is_none())
        } else {
            !status.success() && text.contains(PLACED) && plan.is_none()
        })
    }

    /// `xstream_lever_is_honoured` (the module header); `two` whether the census holds two cards.
    fn xstream_lever_is_honoured(
        c: &serve_gate::Clauses,
        dir: &Path,
        two: bool,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let ctx = c.ctx.to_string();
        let on_a = ["--ctx", ctx.as_str(), "--place", "a"];
        let (unset_status, unset, _) = plan_xstream(c, dir, "plan-xstream", &on_a, None)?;
        let (off_status, off, _) = plan_xstream(c, dir, "plan-xstream-off", &on_a, Some("off"))?;
        let (admit_status, admit, _) =
            plan_xstream(c, dir, "plan-xstream-admit", &on_a, Some("admit"))?;
        let (split_status, split, _) =
            plan_xstream(c, dir, "plan-xstream-split", &on_a, Some("split"))?;
        let exits = unset_status.success()
            && off_status.success()
            && admit_status.success()
            && split_status.success();
        let (Some((u_card, u_host)), Some((o_card, o_host)), Some((a_card, a_host))) =
            (experts(&unset)?, experts(&off)?, experts(&admit)?)
        else {
            println!("a --plan run printed no plan record FAIL");
            check(&mut ok, "xstream_lever_is_honoured", false);
            return Ok(ok);
        };
        let Some((s_card, s_host)) = experts(&split)? else {
            println!("the split --plan run printed no plan record FAIL");
            check(&mut ok, "xstream_lever_is_honoured", false);
            return Ok(ok);
        };
        let pair = (o_card, o_host) == (a_card, a_host);
        // The ring's bytes leave the expert budget: fewer card experts where the plan holds host
        // experts, never more, and the host holds what the card does not.
        let kept_out = |card: u64, host: u64| {
            card <= o_card && (o_host == 0 || card < o_card) && host >= o_host
        };
        let ring = kept_out(u_card, u_host) && kept_out(s_card, s_host);
        // Beside the expert tier card the stream plans as it does on the stage card alone.
        let args = ["--ctx", ctx.as_str(), "--place", "bp"];
        let mut beside = true;
        for word in ["admit", "split"] {
            let name = format!("plan-xstream-bp-{word}");
            let (status, log, text) = plan_xstream(c, dir, &name, &args, Some(word))?;
            beside &= bp_planned(two, status, &log, &text)?;
        }
        println!(
            "xstream plans on --place a: unset card {u_card} host {u_host}; off card {o_card} \
             host {o_host}; admit card {a_card} host {a_host}; split card {s_card} host \
             {s_host}; exit 0 {exits}; off = admit {pair}; the unset and split plans keep the \
             ring out ({} host experts) {ring}; admit and split beside the tier card on --place \
             bp {beside}",
            if o_host == 0 { "no" } else { "the plan holds" }
        );
        check(
            &mut ok,
            "xstream_lever_is_honoured",
            exits && pair && ring && beside,
        );
        Ok(ok)
    }

    /// `load_holds_the_plan` (the module header), on the stderr of the server
    /// [`serve_gate::served_with`] ran in `<dir>/serve`.
    fn load_holds_the_plan(c: &serve_gate::Clauses, dir: &Path) -> Result<bool, GateError> {
        let mut ok = true;
        let log = server_log(&dir.join("serve").join("server.err"), c.kinds)?;
        let Some(plan) = log.first(&record::PLAN)? else {
            println!("the server's stderr holds no plan record FAIL");
            check(&mut ok, "load_holds_the_plan", false);
            return Ok(ok);
        };
        let load = log.one(c.load)?;
        if plan.csv("devices")?.iter().any(|d| d.starts_with("tier0")) {
            println!("load_holds_the_plan: skipped, the plan holds an expert tier card");
            return Ok(ok);
        }
        let planned = (plan.u64("card_experts")?, plan.u64("host_experts")?);
        let loaded = (load.u64("card_experts")?, load.u64("host_experts")?);
        println!(
            "the plan's card and host experts {planned:?}, the load's {loaded:?}: the load holds \
             the plan {}",
            planned == loaded
        );
        check(&mut ok, "load_holds_the_plan", planned == loaded);
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
        let (Some(line), Some(plan), Some(lever)) = (
            line,
            log.first(&record::PLAN)?,
            log.first(&record::RESIDENCY_LEVER)?,
        ) else {
            println!(
                "the server's stderr holds no xstream= line, plan record or residency lever \
                 record FAIL"
            );
            check(&mut ok, "xstream_line_follows_the_rule", false);
            return Ok(ok);
        };
        let tiered = plan.csv("devices")?.iter().any(|d| d.starts_with("tier0"));
        let host = plan.u64("host_experts")?;
        let residency = lever.word("residency")?;
        let want = if residency == "off" {
            "off"
        } else if tiered {
            "admit"
        } else if host > 0 {
            "split"
        } else {
            "off"
        };
        let word = line["xstream=".len()..].split(' ').next().unwrap_or("");
        println!(
            "the load's {line:?}: the residency is {residency}, the plan holds {host} host \
             experts{}: the rule's word is {want} {}",
            if tiered { " beside an expert tier" } else { "" },
            word == want
        );
        check(&mut ok, "xstream_line_follows_the_rule", word == want);
        Ok(ok)
    }

    /// `mmproj_is_held_before_the_plan` (the module header).
    fn mmproj_is_held_before_the_plan(
        c: &serve_gate::Clauses,
        dir: &Path,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let file = ref_model_path()?;
        let missing = dir.join("no-such-mmproj.gguf");
        let missing = missing
            .to_str()
            .ok_or("the gate's directory is not UTF-8")?;
        let (status, log, text) =
            seat(c).plan(dir, "plan-mmproj", &file, &["--mmproj", missing])?;
        let file_named =
            !status.success() && text.contains(missing) && log.first(&record::PLAN)?.is_none();
        let (status, log, text) = seat(c).plan(
            dir,
            "plan-image-limit",
            &file,
            &["--image-max-tokens", "64"],
        )?;
        let limit_named = !status.success()
            && text.contains("--image-max-tokens")
            && log.first(&record::PLAN)?.is_none();
        println!(
            "--mmproj of a missing file: refused naming it {file_named}; --image-max-tokens with \
             no projector: refused naming the flag {limit_named}"
        );
        check(
            &mut ok,
            "mmproj_is_held_before_the_plan",
            file_named && limit_named,
        );
        Ok(ok)
    }

    /// A server of two slots over the file, `--slot-save-path <dir>/save`, and `mmproj` when the
    /// gate was given one: its address, the handle and its stderr log.
    fn two_slot_server(
        c: &serve_gate::Clauses,
        dir: &Path,
        mmproj: Option<&Path>,
    ) -> Result<(Served, String, PathBuf), GateError> {
        let d = dir.join("two-slots");
        let save = d.join("save");
        std::fs::create_dir_all(&save)?;
        let file = ref_model_path()?;
        let (mut cmd, mut all) = seat(c).command(&file)?;
        cmd.env(bloomery_levers::RESIDENCY, "off");
        cmd.env(bloomery_levers::PREFILL_GROUP, "1");
        let total = (2 * c.ctx).to_string();
        let save = save
            .to_str()
            .ok_or("the gate's directory is not UTF-8")?
            .to_owned();
        for a in [
            "--port",
            "0",
            "--ctx",
            &total,
            "--parallel",
            "2",
            "--slot-save-path",
            &save,
        ] {
            all.push(a.to_owned());
        }
        if let Some(m) = mmproj {
            all.push("--mmproj".to_owned());
            all.push(
                m.to_str()
                    .ok_or("the --mmproj path is not UTF-8")?
                    .to_owned(),
            );
        }
        let all: Vec<&str> = all.iter().map(String::as_str).collect();
        let mut s = Served::spawn_cmd(cmd, &all, &d)?;
        let err_log = d.join("server.err");
        let addr = s.address(&err_log, LISTEN_POLLS, Duration::from_secs(1))?;
        Ok((s, addr, err_log))
    }

    /// The slot a `/completion` answer ran on.
    fn slot_of(v: &Value) -> Result<u64, GateError> {
        Ok(v["id_slot"]
            .as_u64()
            .ok_or("a /completion answer without id_slot")?)
    }

    /// `two_slots_load` (the module header).
    fn two_slots_load(c: &serve_gate::Clauses, err_log: &Path) -> Result<bool, GateError> {
        let mut ok = true;
        let log = server_log(err_log, c.kinds)?;
        let (load, listen) = (log.one(c.load)?, log.one(c.listening)?);
        let held = load.u64("slots")? == 2
            && load.u64("ctx")? == c.ctx
            && listen.u64("slots")? == 2
            && listen.u64("ctx")? == c.ctx;
        println!(
            "two slots: load {:?}; listening {:?} {held}",
            load.line(),
            listen.line()
        );
        check(&mut ok, "two_slots_load", held);
        Ok(ok)
    }

    /// `prefill_group_is_honoured` (the module header): the `load` record of the server
    /// [`serve_gate::served_with`] ran in `<dir>/serve` with the lever unset, and of the server of
    /// two slots at `err_log`, which ran it at 1.
    fn prefill_group_is_honoured(
        c: &serve_gate::Clauses,
        dir: &Path,
        err_log: &Path,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let default = server_log(&dir.join("serve").join("server.err"), c.kinds)?
            .one(c.load)?
            .u64("group")?;
        let one = server_log(err_log, c.kinds)?.one(c.load)?.u64("group")?;
        let held = default == bloomery_levers::PREFILL_GROUP_DEFAULT && one == 1;
        println!(
            "prefill group: the lever unset loads at group {default} (the row's default is {}), \
             set to 1 it loads at group {one} {held}",
            bloomery_levers::PREFILL_GROUP_DEFAULT
        );
        check(&mut ok, "prefill_group_is_honoured", held);
        Ok(ok)
    }

    /// The first id of each of `ids` must differ: three conversations that share no prefix.
    fn unrelated(ids: &[&[u32]]) -> Result<(), GateError> {
        let firsts: Vec<Option<&u32>> = ids.iter().map(|i| i.first()).collect();
        for (i, a) in firsts.iter().enumerate() {
            if a.is_none() || firsts[..i].contains(a) {
                return Err(format!(
                    "the slot clauses' prompts start with ids {firsts:?}: they must share no first id"
                )
                .into());
            }
        }
        Ok(())
    }

    /// `slots_keep_their_own_rows` and `slot_file_restores_equal_bits` (the module header), on the
    /// server of two slots at `url`; `real` judges the uncached ids as well.
    fn slot_clauses(
        url: &dyn Fn(&str) -> String,
        dir: &Path,
        real: bool,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let (p, q, r) = (
            tokenized(url, P_TEXT)?,
            tokenized(url, Q_TEXT)?,
            tokenized(url, R_TEXT)?,
        );
        unrelated(&[p.as_slice(), q.as_slice(), r.as_slice()])?;
        let d = dir.join("two-slots");
        // P, then P plus its answer on its slot with nothing between (the truth), then Q on the
        // other slot, then P plus its answer back on its own slot.
        let (a1, v1) = greedy(url, p.clone(), N, &d, "slot_p", true)?;
        let cont: Vec<u32> = p.iter().chain(&a1.tokens).copied().collect();
        let kept = (cont.len() - 1) as u64;
        let (alone, va) = greedy(url, cont.clone(), N, &d, "slot_p_alone", true)?;
        let (_, v2) = greedy(url, q, N, &d, "slot_q", true)?;
        let (back, vb) = greedy(url, cont.clone(), N, &d, "slot_p_back", true)?;
        let (fresh, _) = greedy(url, cont.clone(), N, &d, "slot_p_fresh", false)?;
        let apart = slot_of(&v1)? != slot_of(&v2)?;
        let returned = slot_of(&va)? == slot_of(&v1)?
            && alone.cache_n == kept
            && slot_of(&vb)? == slot_of(&v1)?
            && back.cache_n == kept
            && back.tokens == alone.tokens;
        let uncached = back.tokens == fresh.tokens;
        println!(
            "slots: P on slot {} and Q on slot {} apart {apart}; P back on slot {} kept {} of {} ids, \
             ids {:?} against the truth's {:?} (P plus its answer before Q, kept {}) {returned}; the \
             uncached {:?} {uncached}{}",
            slot_of(&v1)?,
            slot_of(&v2)?,
            slot_of(&vb)?,
            back.cache_n,
            cont.len(),
            back.tokens,
            alone.tokens,
            alone.cache_n,
            fresh.tokens,
            judged(real)
        );
        check(
            &mut ok,
            "slots_keep_their_own_rows",
            apart && returned && (uncached || !real),
        );

        // R on a slot, its state saved, R plus its answer on that slot (the truth), that slot
        // erased, the file restored into the other slot (which holds other prompts' rows, so a
        // restore that wrote none reads as other ids), then R plus its answer, which the slot with
        // the restored state takes.
        let (g, vr) = greedy(url, r.clone(), N, &d, "file_r", true)?;
        let slot = slot_of(&vr)?;
        if slot > 1 {
            return Err(format!("the two-slot server ran a request on slot {slot}").into());
        }
        let other = u64::from(slot == 0);
        let file = json!({ "filename": SLOT_FILE });
        let cont: Vec<u32> = r.iter().chain(&g.tokens).copied().collect();
        let positions = (cont.len() - 1) as u64;
        let (st_save, saved) = slot_action(url, slot, "save", &file)?;
        let (alone, va) = greedy(url, cont.clone(), N, &d, "file_r_alone", true)?;
        let (st_erase, _) = slot_action(url, slot, "erase", &json!({}))?;
        let (st_restore, restored) = slot_action(url, other, "restore", &file)?;
        let (back, vback) = greedy(url, cont.clone(), N, &d, "file_r_back", true)?;
        let (fresh, _) = greedy(url, cont.clone(), N, &d, "file_r_fresh", false)?;
        let held = st_save == 200
            && saved["n_saved"] == json!(positions)
            && slot_of(&va)? == slot
            && alone.cache_n == positions
            && st_erase == 200
            && st_restore == 200
            && restored["n_restored"] == json!(positions)
            && restored["n_read"] == saved["n_written"]
            && slot_of(&vback)? == other
            && back.cache_n == positions
            && back.tokens == alone.tokens;
        let uncached = back.tokens == fresh.tokens;
        println!(
            "slot file: {positions} positions saved from slot {slot} and restored into slot {other}; \
             the continuation on slot {} kept {} ids with {:?} against the truth's {:?} (on slot {} \
             before the erase, kept {}) {held}; the uncached {:?} {uncached}{}",
            slot_of(&vback)?,
            back.cache_n,
            back.tokens,
            alone.tokens,
            slot_of(&va)?,
            alone.cache_n,
            fresh.tokens,
            judged(real)
        );
        check(
            &mut ok,
            "slot_file_restores_equal_bits",
            held && (uncached || !real),
        );
        Ok(ok)
    }

    /// How a slot clause's line names the uncached comparison: judged in the real tier, printed in
    /// the fixture tier (the module header).
    fn judged(real: bool) -> &'static str {
        if real {
            ""
        } else {
            " (printed, not judged: the fixture's near ties part the ids under the batch's band)"
        }
    }

    /// `tool_calls_follow_the_template` (the module header).
    fn tool_calls_follow_the_template(
        url: &dyn Fn(&str) -> String,
        dir: &Path,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        if !tier::run_clause("tool_calls_follow_the_template", Tag::FileBound)? {
            return Ok(ok);
        }
        // The template's `low` effort and a reasoning budget keep the think span short, so the
        // call fits the tokens.
        let mut body = tool_chat(TOOL_TOKENS);
        body["reasoning_effort"] = json!("low");
        body["reasoning_budget"] = json!(TOOL_THINK_BUDGET);
        let answer = chatted(url, &dir.join("two-slots"), "tools", &body);
        let held = match answer {
            Ok(a) => {
                let choice = &a["choices"][0];
                let calls = choice["message"]["tool_calls"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default();
                let function = calls.first().map(|c| &c["function"]);
                let name = function.and_then(|f| f["name"].as_str());
                let city: Option<Value> = function
                    .and_then(|f| f["arguments"].as_str())
                    .and_then(|a| serde_json::from_str(a).ok());
                println!(
                    "tool call: finish {:?}, {} call(s), function {name:?}, arguments {city:?}",
                    choice["finish_reason"],
                    calls.len()
                );
                choice["finish_reason"] == "tool_calls"
                    && calls.len() == 1
                    && name == Some("get_weather")
                    && city.is_some_and(|c| c["city"].is_string())
            }
            Err(e) => {
                println!("tool call: {e}");
                false
            }
        };
        check(&mut ok, "tool_calls_follow_the_template", held);
        Ok(ok)
    }

    /// `image_request_is_served` (the module header), on the server of two slots at `url` that
    /// was started with `--mmproj`.
    fn image_request_is_served(
        c: &serve_gate::Clauses,
        url: &dyn Fn(&str) -> String,
        err_log: &Path,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let log = server_log(err_log, c.kinds)?;
        let records =
            log.first(&record::VISION)?.is_some() && log.first(&record::VISION_SEAT)?.is_some();
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(IMAGE);
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let body = json!({
            "messages": [{ "role": "user", "content": [
                { "type": "image_url", "image_url": {
                    "url": format!("data:image/png;base64,{}", base64(&bytes)) } },
                { "type": "text", "text": IMAGE_ASK },
            ] }],
            "temperature": 0, "max_tokens": N, "cache_prompt": false,
        });
        let (st, text) = curl(&url("/v1/chat/completions"), Some(&body), false)?;
        let reply: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        let prompt = reply["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
        let completion = reply["usage"]["completion_tokens"].as_u64().unwrap_or(0);
        let prompt_ms = reply["timings"]["prompt_ms"].as_f64();
        println!(
            "image chat: HTTP {st}, prompt tokens {prompt} (the image's span is {IMAGE_ROWS}) in \
             {prompt_ms:?} ms, {completion} completion tokens; vision and vision_seat records \
             {records}"
        );
        if st != 200 {
            println!("image chat answer: {text}");
        }
        check(
            &mut ok,
            "image_request_is_served",
            records && st == 200 && prompt >= IMAGE_ROWS && completion > 0,
        );
        Ok(ok)
    }

    /// The clauses of the server of two slots.
    fn two_slot_clauses(
        c: &serve_gate::Clauses,
        dir: &Path,
        mmproj: Option<&Path>,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let image = tier::run_clause("image_request_is_served", Tag::FileBound)?;
        let mmproj = mmproj.filter(|_| image);
        let (mut s, addr, err_log) = two_slot_server(c, dir, mmproj)?;
        let url = |p: &str| format!("http://{addr}{p}");
        ok &= two_slots_load(c, &err_log)?;
        ok &= prefill_group_is_honoured(c, dir, &err_log)?;
        ok &= slot_clauses(&url, dir, tier::Tier::from_env()? == tier::Tier::Real)?;
        ok &= tool_calls_follow_the_template(&url, dir)?;
        if mmproj.is_some() {
            ok &= image_request_is_served(c, &url, &err_log)?;
        } else if image {
            println!("image_request_is_served: skipped, no --mmproj");
        }
        println!("two-slot server stopped: {}", s.stop()?);
        Ok(ok)
    }

    /// The gate's end. The `PASSED` line names the clauses.
    fn finish(ok: bool) -> Result<(), GateError> {
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: gate_mistrallarge4_serve the seat plans its slots (a round of several is a \
             select and a step each), plans the expert tier card where two cards are in view, \
             honours BLOOMERY_RESIDENCY (a set word as given, the serving word resolved by the \
             plan unset, off on gate) and BLOOMERY_XSTREAM (the ring planned out of the expert \
             budget, the word resolved by the common rule at the open), holds --mmproj to its \
             file before the plan, loads at the context it was asked for and the prompt group \
             the lever names, answers a chat turn with its reasoning split at the file's tags, \
             keeps each slot's rows apart, restores a slot file bit for bit, parses a tool call \
             by the file's template and serves an image request"
        );
        Ok(())
    }

    pub fn run() -> Result<(), GateError> {
        let Args { dir, mmproj } = args()?;
        let c = clauses()?;
        std::fs::create_dir_all(&dir)?;
        let two = gpu_census::census()?.len() >= 2;
        let mut ok = plan_names_the_slots(&c, &dir)?;
        ok &= place_bp_follows_the_cards(&c, &dir, two)?;
        ok &= place_unset_runs_a(&c, &dir, two)?;
        ok &= residency_lever_is_honoured(&c, &dir)?;
        ok &= xstream_lever_is_honoured(&c, &dir, two)?;
        ok &= mmproj_is_held_before_the_plan(&c, &dir)?;
        ok &= serve_gate::served_with(&c, &dir, ASK)?;
        ok &= xstream_line_follows_the_rule(&c, &dir)?;
        ok &= load_holds_the_plan(&c, &dir)?;
        ok &= two_slot_clauses(&c, &dir, mmproj.as_deref())?;
        finish(ok)
    }
}
