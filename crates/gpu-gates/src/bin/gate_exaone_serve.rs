//! `gate_exaone_serve` — the exaone-moe seat of `bloomery-serve` (the flint seat, whose file's
//! router row picks K-EXAONE's body) against the model it serves, on the fixture GGUF of the
//! profile (`BLOOMERY_REF_MODEL`): the seat starts, plans its two resident slots, loads the file
//! on the placed body, answers one chat request with its reasoning split at the model's own
//! `<think>` span and keeps the prefix a request shares with the slot by the body's own rule.
//!
//!     gate_exaone_serve --dir <dir> [--tier]
//!
//! The seat serves resident sequences, a select and a step each (`pass=turns`). Every
//! server is `bloomery-serve --model flint -m <file>` beside this binary, `BLOOMERY_REF_MODEL`
//! removed from its environment. Logs per server in `<dir>/<name>/` (`server.err`, the answer).
//! `--tier` (`just gate-gpu-exaone-twocard`, both cards in view) runs the two tier clauses alone,
//! `place_bp_follows_the_cards` and `mtp_survives_a_cut_on_the_tier`, and is refused by name
//! where the census holds one card.
//!
//! The shared clauses are `shared/serve_gate.rs`'s: `plan_names_the_slots` (the body's
//! `Sequences` are `app::arch::flint`'s resident ones, so `--parallel 2` is served, not refused),
//! `load_and_listen` (the load record names arch `exaone-moe`, the trunk's six layers, one slot),
//! `chat_answers_a_turn`, `chat_splits_the_reasoning_at_the_models_tags` (the file's own template
//! opens `<think>` in the generation prompt and the model closes it; the request carries a
//! reasoning budget so the span closes even if the cut model never writes the close itself),
//! `chat_ends_at_the_turn_end` (left by name: the six-layer cut never writes the turn end inside
//! the tokens a request asks for — the model's own turn end is `<|endofturn|>`, eos 53, and the
//! clause is the real model's when its file lands), `live_slot_keeps_the_prefix`
//! (the body's ring takes a shared prefix back only where the sequence stands:
//! [`serve_gate::Cut::WhereItStands`]) and `saved_state_returns` (skipped by name while the seat
//! prints no prompt-cache line).
//!
//! - `place_bp_follows_the_cards`: `--place bp --plan` where the census holds two cards exits 0 and
//!   prints a `plan` record whose devices name the tier card (`tier0`), or, where the stage holds
//!   every routed expert (the fixture), exits non-zero naming the tier card that would hold none
//!   ([`EMPTY_TIER`]) with no `plan` record; where it holds one card it exits non-zero, its stderr
//!   names the place word the placement refuses, and no `plan` record is printed. On two cards
//!   the clause `place_bp_holds_tier_experts` follows it: the plan at a card budget of the stage's
//!   floor and [`HOST_EXTRA`] exits 0, names the tier card, holds experts on the stage card, and
//!   its stage and host experts leave the rest of plan (a)'s at that budget to the tier card.
//!   FAIL-first: a seat whose rule serves no tier card (`TierRule::UNTIERED`) refuses `bp` on two
//!   cards with "the flint body serves 0 at most" and no tier card, and both clauses are red.
//! - `mtp_survives_a_cut_on_the_tier` (`--tier`, both cards in view): the clause below on
//!   `--place bp`, under the card budget of the stage's floor and [`HOST_EXTRA`] (a part of the
//!   routed experts on the stage card, the tier card holding what it has room for, the host the
//!   rest), residency `off`, the drafted width fixed: the plain and the drafted `bp` servers and
//!   a reference on plan (a) at the same budget. The `bp` plans name the tier card and the
//!   reference's none; the drafted `bp` load's ids are the plain `bp` load's bit for bit (the
//!   draft's verify pass of two rows crosses the tier's handoff as the step's one row does),
//!   the draft proposes in both requests and joins the second at the kept position; the plain
//!   `bp` load's first id of each request is the reference's, and the routed picks its host
//!   reads through the file's mapping fall against the reference's by the ratio of the plans'
//!   host experts (within [`HOST_SLACK`]): the experts the tier plan takes off the host are
//!   served by the cards. The tier card's arithmetic is the card's, the reference's the host's,
//!   so the ids are not held further. FAIL-first, shown: a Flint tier whose page rows differ from
//!   the body's open (`LegFamily::ROWS` 1 against `PASS_ROWS` 2) refuses the `bp` load by name
//!   (`the node tier's shape is not this file's`), so the server never listens and the clause is
//!   red.
//! - `tools_request_is_served`: a chat request that carries `tools` is answered with a completion
//!   (HTTP 200, `choices`), never an error: the file's own template renders the tool-declare
//!   section (`<tool_declare|>` and the `<tool>` JSON blocks), and the description row binds the
//!   QwenXml parser (`ToolFormat::of_template` of the template's `<tool_call>` markup), where the
//!   Kimi K3 seat refuses the same request by name. FAIL-first: a seat that leaves the tools out
//!   of the prompt, or one that answers the request with the unparsed-format refusal, is red.
//! - `tool_call_round_trips`: the file's chat template round-trips one tool call — a server over a
//!   scripted engine whose every reply is the template's own `<tool_call>` markup
//!   (`serve_client::scripted_call`, the qwen3 serve gate's method: the clause reads nothing the
//!   model chose to reply) answers one tool turn with `finish_reason` `tool_calls` and one call of
//!   the listed function with the arguments the script gave. FAIL-first: a scanner that drops a
//!   call's parameters answers `{}` and the clause is red.
//! - `mtp_serves_the_plain_ids`: one server of the plain path and one under `BLOOMERY_DRAFT=mtp`
//!   (the file's MTP layer as the draft, `--parallel 1`) answer the same greedy `/completion` of
//!   [`MTP_IDS`] ids with the same [`MTP_N`] ids and stop; the drafted server prints its
//!   `load draft=mtp` record, the plain one none, and the drafted answer carries draft counts
//!   (`timings.draft_n`, `draft_n_accepted`, printed: the acceptance is a reading on this
//!   cut file); `BLOOMERY_DRAFT=mtp --parallel 2` is refused by name before the load. FAIL-first:
//!   a seat that drafts through a draft that moves the target (a token differs), one that never
//!   opens the draft (no record, no counts), and one that serves two slots under the draft
//!   (listens) are red.
//! - `mtp_survives_a_cut_on_the_host_tier`, `mtp_survives_a_cut_with_residency` and
//!   `mtp_survives_a_cut_at_the_default_width`: the draft beside a prefix cut, on the host tier. A plain server and one under `BLOOMERY_DRAFT=mtp
//!   BLOOMERY_MTP_WIDTH=fixed` (every pass verifies a whole proposal, so the draft counts do not
//!   follow the width chooser's reading of the pass wall) take a card budget of their own floor
//!   (read off the refusal of a budget below it: the draft's bytes are in its floor) and
//!   [`HOST_EXTRA`], so each holds the same routed experts on the card (about a quarter) and the
//!   rest on the host, and every token's top eight reach the host tier: the step's chain, the verify
//!   pass's pair chain. Two chats that share the system message follow each other in the slot; the
//!   second keeps the shared prefix, the drafted load one id fewer than the plain (the prompt call
//!   feeds the id the draft's entry at the first fed position reads the target's hidden row of),
//!   the draft joins the call at that position (`mtp prompt` ... `skipped=none`) and proposes in
//!   both requests (`timings.draft_n > 0`), the plain load in neither. The first runs under
//!   `BLOOMERY_RESIDENCY=off`, where no expert moves and the drafted load's ids are the plain
//!   load's, bit for bit (the host tier's pair chain sums a row as the step's one-column call
//!   does). The second takes the word the plan resolves (`mid-p0-s1`): the swap machine and its
//!   tally run beside the verify pass of two rows, and the drafted load's ids are the plain load's
//!   or part from them at a near tie ([`NEAR_TIE`]: the two loads' experts flip at different
//!   passes). FAIL-first: a seat that keeps one more id than it can join the draft at, or turns
//!   the draft off at a cut, answers the second request with no draft and no join record; a
//!   residency tally that counts one row (`PASS_ROWS` 1 at the load) fails the second clause's
//!   first drafted request, where the first clause, which runs no tally, stays green. The third
//!   runs the first at the default width chooser, the defaults twin of the `fixed` pin: the
//!   chooser's gate follows the pass wall, so a request's draft counts are not held there (the
//!   fixture's draft keeps none, and the chooser may close its gate), and the rest of the clause
//!   is: the join record, the kept prefix and the plain load's ids.

#[cfg(all(feature = "deepseek41", feature = "flint"))]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_exaone_serve", gate::run())
}

#[cfg(all(feature = "deepseek41", feature = "flint"))]
#[allow(
    dead_code,
    reason = "the shared clauses serve several gates; each takes the helpers it reads"
)]
#[path = "shared/serve_gate.rs"]
mod serve_gate;

#[cfg(all(feature = "deepseek41", feature = "flint"))]
mod gate {
    use std::path::Path;
    use std::time::Duration;

    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::serve_client::{
        ChatReply, SeatGate, Served, answered, check, completion, curl, draft_counts, greedy_chat,
        ids_of, json_of, server_log, stop_of,
    };
    use bloomery_gpu_gates::{GateError, checks_failed, gpu_census, ref_model_path};
    use gguf::Split;
    use serde_json::{Value, json};

    use crate::serve_gate;

    /// The slot's positions the server is asked for: the oracle's own bound, the patched tree's
    /// context the fixture GGUF's 4096-window layer keeps.
    const CTX: u64 = 4096;

    /// The tags of the model's think span, as its vocabulary spells them.
    const THINK_TAGS: [&str; 2] = ["<think>", "</think>"];

    /// Polls of a second that bound the wait for a server to listen.
    const LISTEN_POLLS: usize = 1000;

    /// The chat request the seat's turn is asked as: a reasoning budget, so the span closes even
    /// if the cut model never writes the close itself, and the message holds reasoning.
    const ASK: serve_gate::Ask = serve_gate::Ask::THINK_SPAN;

    /// `tools_request_is_served` (the module header): a server of its own at [`CTX`].
    fn tools_request_is_served(dir: &Path) -> Result<bool, GateError> {
        let mut ok = true;
        let d = dir.join("tools");
        std::fs::create_dir_all(&d)?;
        let file = ref_model_path()?;
        let (cmd, mut all) = SeatGate {
            model: "flint",
            kinds: record::BLOOMERY_SERVE_HYBRID,
        }
        .command(&file)?;
        all.extend(
            ["--port", "0", "--ctx", &CTX.to_string()]
                .iter()
                .map(|a| (*a).to_owned()),
        );
        let all: Vec<&str> = all.iter().map(String::as_str).collect();
        let mut s = Served::spawn_cmd(cmd, &all, &d)?;
        let err_log = d.join("server.err");
        let addr = match s.address(&err_log, LISTEN_POLLS, Duration::from_secs(1)) {
            Ok(addr) => addr,
            Err(e) => {
                println!("tools: the server never listened: {e}");
                check(&mut ok, "tools_request_is_served", false);
                return Ok(ok);
            }
        };
        let url = format!("http://{addr}/v1/chat/completions");
        let turn = json!([{ "role": "user", "content": "What is the weather in Seoul?" }]);
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "The weather of a city.",
                "parameters": {
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"],
                },
            },
        }]);
        let with_tools = json!({
            "messages": turn, "tools": tools, "temperature": 0, "max_tokens": 8, "stream": false,
        });
        let (status, text) = curl(&url, Some(&with_tools), false)?;
        std::fs::write(d.join("tools.json"), &text)?;
        let answer: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        let message = answer["error"]["message"].as_str().unwrap_or("");
        let served = status == 200 && answer.get("choices").is_some();
        // The seat still answers a plain turn after it.
        let plain = json!({
            "messages": turn, "temperature": 0, "max_tokens": 8, "stream": false,
        });
        let (after, after_text) = curl(&url, Some(&plain), false)?;
        std::fs::write(d.join("after.json"), &after_text)?;
        let usable = after == 200;
        println!(
            "tools: HTTP {status} (a completion, no error {served}; {message:?}); a plain turn \
             after it: HTTP {after} {usable}; {}",
            s.stop()?
        );
        check(&mut ok, "tools_request_is_served", served && usable);
        Ok(ok)
    }

    /// The ids [`mtp_serves_the_plain_ids`] asks for a completion of, and how many it asks.
    const MTP_IDS: u32 = 17;
    const MTP_N: usize = 24;

    /// A server of the seat on the profile's file in `<dir>/<name>/` at [`CTX`], `BLOOMERY_DRAFT`
    /// set to `draft` when given, `--parallel parallel`, `--place` `place` when given, each of
    /// `levers` set beside them: it and its stderr's log, or the error of a server that exited
    /// before it listened (its log in the text).
    fn mtp_server(
        dir: &Path,
        name: &str,
        draft: Option<&str>,
        parallel: u32,
        levers: &[(&str, &str)],
        place: Option<&str>,
    ) -> Result<(Served, String, std::path::PathBuf), GateError> {
        let d = dir.join(name);
        std::fs::create_dir_all(&d)?;
        let file = ref_model_path()?;
        let (mut cmd, mut all) = SeatGate {
            model: "flint",
            kinds: record::BLOOMERY_SERVE_HYBRID,
        }
        .command(&file)?;
        if let Some(word) = draft {
            cmd.env("BLOOMERY_DRAFT", word);
        }
        for &(lever, value) in levers {
            cmd.env(lever, value);
        }
        all.extend(
            [
                "--port",
                "0",
                "--ctx",
                &CTX.to_string(),
                "--parallel",
                &parallel.to_string(),
            ]
            .iter()
            .map(|a| (*a).to_owned()),
        );
        if let Some(word) = place {
            all.extend(["--place".to_owned(), word.to_owned()]);
        }
        let all: Vec<&str> = all.iter().map(String::as_str).collect();
        let mut s = Served::spawn_cmd(cmd, &all, &d)?;
        let err_log = d.join("server.err");
        let addr = s.address(&err_log, LISTEN_POLLS, Duration::from_secs(1))?;
        Ok((s, addr, err_log))
    }

    /// `mtp_serves_the_plain_ids` (the module header).
    fn mtp_serves_the_plain_ids(dir: &Path) -> Result<bool, GateError> {
        const CLAUSE: &str = "mtp_serves_the_plain_ids";
        let mut ok = true;
        let ids: Vec<u32> = (1..=MTP_IDS).collect();
        let mut answers = Vec::new();
        for (name, draft) in [("mtp-off", None), ("mtp-on", Some("mtp"))] {
            let (mut s, addr, err_log) = match mtp_server(dir, name, draft, 1, &[], None) {
                Ok(served) => served,
                Err(e) => {
                    println!("{CLAUSE}: the {name} server never listened: {e}");
                    check(&mut ok, CLAUSE, false);
                    return Ok(ok);
                }
            };
            let url = move |route: &str| format!("http://{addr}{route}");
            let answer = completion(&url, &ids, MTP_N, false)?;
            let tokens = ids_of(&answer["tokens"])?;
            let stop = stop_of(&answer)?.to_owned();
            let counts = draft_counts(&answer)?;
            let log = std::fs::read_to_string(&err_log)?;
            let record = log.lines().any(|l| l.starts_with("load draft=mtp"));
            println!(
                "{CLAUSE} {name}: {} ids, stop {stop}, drafts proposed {} kept {}, load draft=mtp \
                 record {record}; {}",
                tokens.len(),
                counts.0,
                counts.1,
                s.stop()?
            );
            answers.push((tokens, stop, counts, record));
        }
        let (off, on) = (&answers[0], &answers[1]);
        let same = off.0 == on.0 && off.1 == on.1 && off.0.len() == MTP_N;
        let drafted = on.3 && !off.3 && on.2.0 > 0 && off.2 == (0, 0);
        println!(
            "{CLAUSE}: the drafted server's {MTP_N} ids and stop are the plain server's {same}; \
             the record and the counts are the drafted server's alone {drafted}"
        );
        check(&mut ok, CLAUSE, same && drafted);
        // Two slots beside the draft: refused by name before the load, so it never listens.
        let refused = match mtp_server(dir, "mtp-two", Some("mtp"), 2, &[], None) {
            Ok((mut s, ..)) => {
                let _ = s.stop();
                println!("{CLAUSE}: --parallel 2 under BLOOMERY_DRAFT=mtp listened");
                false
            }
            Err(e) => {
                let text = e.to_string();
                println!("{CLAUSE}: --parallel 2 under BLOOMERY_DRAFT=mtp: {text}");
                text.contains("BLOOMERY_DRAFT=mtp") && text.contains("drafts one slot")
            }
        };
        check(&mut ok, "mtp_refuses_two_slots", refused);
        Ok(ok)
    }

    /// The card budget a host-tier load takes past its own floor, in bytes: about a quarter of the
    /// fixture's routed experts (four layers of 256) sit on the card, the rest on the host, so
    /// every token's top eight reach the host tier.
    const HOST_EXTRA: u64 = 1_400_000_000;

    /// The system message and the two user turns [`mtp_survives_a_cut_on_the_host_tier`] asks:
    /// the template renders the system message first, so the two prompts share its ids.
    const SYSTEM: &str = "You are a careful assistant. Answer in short plain sentences.";
    const USERS: [&str; 2] = [
        "What is the capital of France?",
        "Name a prime number above ten.",
    ];

    /// The tokens each chat of that clause asks for at most.
    const CHAT_N: usize = 24;

    /// The gap, in logprob, below which a drafted load's id may part from the plain load's under
    /// the residency the plan resolves: the two loads' experts flip at different passes, so a
    /// token can land on either side of a near tie.
    // PIN(2026-11-13): 1.0 = 2 sqrt(2) sigma, sigma 0.35 logits (`prompts::SIGMA_LOGIT_V41`, the
    // value every family carries until its own is measured), the gap of two ids' logits between
    // two implementations at two sigma of their difference; provisional, read on this seat's
    // first gaps, which the clause prints.
    const NEAR_TIE: f64 = 1.0;

    /// The factor the host's picks may exceed the ratio of its experts' counts by between two
    /// plans of one prompt: the router's picks follow the ids' hotness, not their count.
    const HOST_SLACK: u64 = 2;

    /// What one `--plan` of a host-tier load holds: the budget it took, its routed experts on the
    /// stage card and on the host, and whether it names an expert tier card (`tier0` among its
    /// devices).
    struct HostPlan {
        budget: u64,
        card: u64,
        host: u64,
        tiered: bool,
    }

    /// The floor a budget refusal names: `... holds <budget> of the <floor> bytes that takes: ...`.
    fn floor_of(text: &str) -> Result<u64, GateError> {
        text.split(" bytes that takes")
            .next()
            .and_then(|head| head.rsplit(" of the ").next())
            .and_then(|n| n.trim().parse().ok())
            .ok_or_else(|| format!("the refusal names no floor: {text}").into())
    }

    /// The `--plan` of the load (the drafted one when `draft`) under a card budget of its own
    /// floor and [`HOST_EXTRA`], on `--place` `place` when given: the floor is the one a budget
    /// below it names, so a plain and a drafted load hold the same experts on the card (the
    /// draft's bytes are in the floor).
    fn host_plan(
        dir: &Path,
        name: &str,
        draft: bool,
        residency: Option<&str>,
        place: Option<&str>,
    ) -> Result<HostPlan, GateError> {
        let seat = SeatGate {
            model: "flint",
            kinds: record::BLOOMERY_SERVE_HYBRID,
        };
        let file = ref_model_path()?;
        let ctx = CTX.to_string();
        let mut args = vec!["--port", "0", "--ctx", ctx.as_str(), "--parallel", "1"];
        if let Some(word) = place {
            args.extend(["--place", word]);
        }
        let draft = draft.then_some("mtp");
        let (status, _, text) = seat.plan_under(
            (dir, &format!("{name}-floor")),
            &file,
            &args,
            &[
                (bloomery_levers::DRAFT, draft),
                (bloomery_levers::CARD_BUDGET, Some("1M")),
                (bloomery_levers::RESIDENCY, residency),
            ],
        )?;
        if status.success() {
            return Err("a card budget of 1 MiB planned a load: the floor is not read".into());
        }
        let budget = floor_of(&text)? + HOST_EXTRA;
        let (status, log, text) = seat.plan_under(
            (dir, name),
            &file,
            &args,
            &[
                (bloomery_levers::DRAFT, draft),
                (bloomery_levers::CARD_BUDGET, Some(&budget.to_string())),
                (bloomery_levers::RESIDENCY, residency),
            ],
        )?;
        if !status.success() {
            return Err(format!("the plan at a card budget of {budget} B: {text}").into());
        }
        let plan = log
            .first(&record::PLAN)?
            .ok_or("--plan printed no plan line")?;
        Ok(HostPlan {
            budget,
            card: plan.u64("card_experts")?,
            host: plan.u64("host_experts")?,
            tiered: plan.csv("devices")?.iter().any(|d| d.starts_with("tier0")),
        })
    }

    /// What one chat of [`mtp_survives_a_cut_on_the_host_tier`] answered: the reply's ids with each
    /// step's runner-up, the draft counts and the positions the request kept.
    struct Turn {
        reply: ChatReply,
        draft: (u64, u64),
        kept: u64,
    }

    /// One chat of [`SYSTEM`] and `user`, greedy, [`CHAT_N`] tokens, the slot's prefix kept; the
    /// answer's body is saved as `file`.
    fn chat_turn(addr: &str, user: &str, file: &Path) -> Result<Turn, GateError> {
        let url = format!("http://{addr}/v1/chat/completions");
        let messages = json!({
            "messages": [
                { "role": "system", "content": SYSTEM },
                { "role": "user", "content": user },
            ],
        });
        let mut body = greedy_chat(&messages, CHAT_N);
        body["cache_prompt"] = json!(true);
        let (status, text) = curl(&url, Some(&body), false)?;
        std::fs::write(file, &text)?;
        let answer = json_of("/v1/chat/completions", status, &text)?;
        let kept = answer["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .ok_or("the chat answer counts no cached tokens")?;
        Ok(Turn {
            reply: answered(status, &text)?,
            draft: draft_counts(&answer)?,
            kept,
        })
    }

    /// What one server of [`mtp_survives_a_cut_on_the_host_tier`]'s family answered: its plan,
    /// its two chats and the draft's join records (the prompt call's start position and why the
    /// draft skipped it, `none` when it joined).
    struct Arm {
        plan: HostPlan,
        turns: Vec<Turn>,
        joins: Vec<(u64, String)>,
        /// The routed picks the server's host services saw over both chats and, of them, those
        /// it served itself, read through the model file's mapping (the tier census).
        picks: (u64, u64),
    }

    /// One server of the family in `<dir>/<name>/`: the plan at a card budget of its own floor and
    /// [`HOST_EXTRA`] (`--place` `place`, the common rule's when `None`), the draft `draft`, the
    /// residency `residency` (`None`: the word the plan resolves) and, for a drafted load at the
    /// `fixed` width, `BLOOMERY_MTP_WIDTH=fixed`, which every pass verifying a whole proposal
    /// needs so a request's draft counts do not follow the width chooser's reading of the pass
    /// wall; then the two chats of [`USERS`]. `None` where the server never listened or a chat
    /// failed (printed under `clause`).
    fn arm(
        dir: &Path,
        (clause, name): (&str, &str),
        (draft, residency, fixed): (Option<&str>, Option<&str>, bool),
        place: Option<&str>,
    ) -> Result<Option<Arm>, GateError> {
        let plan = host_plan(
            dir,
            &format!("{name}-plan"),
            draft.is_some(),
            residency,
            place,
        )?;
        println!(
            "{clause} {name}: card budget {} B, routed experts {} on the stage card and {} on the \
             host, an expert tier card {}",
            plan.budget, plan.card, plan.host, plan.tiered
        );
        let budget = plan.budget.to_string();
        let mut levers = vec![(bloomery_levers::CARD_BUDGET, budget.as_str())];
        if let Some(word) = residency {
            levers.push((bloomery_levers::RESIDENCY, word));
        }
        if draft.is_some() && fixed {
            levers.push((bloomery_levers::MTP_WIDTH, "fixed"));
        }
        let (mut s, addr, err_log) = match mtp_server(dir, name, draft, 1, &levers, place) {
            Ok(served) => served,
            Err(e) => {
                println!("{clause}: the {name} server never listened: {e}");
                return Ok(None);
            }
        };
        let mut turns = Vec::new();
        for (i, user) in USERS.iter().enumerate() {
            let file = dir.join(name).join(format!("chat{i}.json"));
            match chat_turn(&addr, user, &file) {
                Ok(turn) => turns.push(turn),
                Err(e) => {
                    println!("{clause} {name}: a chat failed: {e}; {}", s.stop()?);
                    return Ok(None);
                }
            }
        }
        let log = server_log(&err_log, &[&record::MTP_PROMPT, &record::TIER_CENSUS])?;
        let mut joins: Vec<(u64, String)> = Vec::new();
        for r in log.all(&record::MTP_PROMPT)? {
            joins.push((r.u64("start")?, r.text("skipped")?.to_owned()));
        }
        let mut picks = (0, 0);
        for r in log.all(&record::TIER_CENSUS)? {
            picks.0 += r.u64("picks")?;
            picks.1 += r.u64("file")?;
        }
        println!(
            "{clause} {name}: ids {:?} then {:?}; drafts proposed/kept {:?} then {:?}; \
             positions kept {} then {}; mtp prompt records {joins:?}; {}",
            turns[0].reply.ids,
            turns[1].reply.ids,
            turns[0].draft,
            turns[1].draft,
            turns[0].kept,
            turns[1].kept,
            s.stop()?
        );
        Ok(Some(Arm {
            plan,
            turns,
            joins,
            picks,
        }))
    }

    /// The drafted load `on` beside the plain load `off` of one placement, held to the clause's
    /// rules: the same plan, the drafted requests proposing at the fixed width and the plain
    /// load's never, the second request keeping a shared prefix one id fewer under the draft, the
    /// draft joining at the kept position, and the ids the plain load's (bit for bit under
    /// residency `off`, within a near tie under the plan's own). `placed` is the caller's verdict
    /// on the plan's shape and the words that name it.
    fn drafted_follows_plain(
        clause: &str,
        (residency, fixed): (Option<&str>, bool),
        (off, on): (&Arm, &Arm),
        placed: (bool, &str),
    ) -> bool {
        let same_plan = (off.plan.card, off.plan.host, off.plan.tiered)
            == (on.plan.card, on.plan.host, on.plan.tiered);
        let both_draft = (!fixed || on.turns.iter().all(|t| t.draft.0 > 0))
            && off.turns.iter().all(|t| t.draft == (0, 0));
        let kept = off.turns[1].kept > 0 && on.turns[1].kept > 0;
        // One id fewer kept than the plain load keeps: the prompt call feeds the id the draft's
        // entry at the first fed position reads the hidden row of.
        let one_more = on.turns[1].kept + 1 == off.turns[1].kept;
        let joined =
            on.joins.len() == 1 && on.joins[0].1 == "none" && on.joins[0].0 == on.turns[1].kept;
        let mut ids_ok = true;
        for i in 0..USERS.len() {
            let (plain, drafted) = (&off.turns[i].reply, &on.turns[i].reply);
            let agree = plain.ids.len().min(drafted.ids.len());
            let (upto, gap, near) = plain.follows(&drafted.ids, agree, NEAR_TIE);
            let same = plain.ids == drafted.ids && !plain.ids.is_empty();
            println!(
                "{clause} request {i}: the drafted load's ids are the plain load's {same}; they \
                 agree to step {upto} of {agree}, the runner-up gap there {gap:?} (a near tie is \
                 below {NEAR_TIE}) {near}"
            );
            ids_ok &= if residency == Some("off") { same } else { near };
        }
        println!(
            "{clause}: {} {}, the same in both loads {same_plan}; the drafted requests proposed \
             (held at the fixed width), the plain load's none {both_draft}; the second request \
             kept a shared prefix {kept}, one id fewer under the draft {one_more}; the draft \
             joined at the kept position, skipped by no reason {joined}; the ids {ids_ok}",
            placed.1, placed.0
        );
        placed.0 && same_plan && both_draft && kept && one_more && joined && ids_ok
    }

    /// `mtp_survives_a_cut_on_the_host_tier` (the module header) under `residency` (`None`: the
    /// word the plan resolves), named `clause`; the drafted load's width is `fixed` or the
    /// default chooser's, under which a request's draft counts follow the pass wall and are not
    /// held.
    fn mtp_survives_a_cut_on_the_host_tier(
        dir: &Path,
        clause: &str,
        residency: Option<&str>,
        fixed: bool,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let mut runs = Vec::new();
        for (name, draft) in [("off", None), ("mtp", Some("mtp"))] {
            let name = format!("{clause}-{name}");
            match arm(dir, (clause, &name), (draft, residency, fixed), None)? {
                Some(a) => runs.push(a),
                None => {
                    check(&mut ok, clause, false);
                    return Ok(ok);
                }
            }
        }
        let (off, on) = (&runs[0], &runs[1]);
        let hosted = off.plan.card > 0 && off.plan.host > 0;
        let verdict = drafted_follows_plain(
            clause,
            (residency, fixed),
            (off, on),
            (hosted, "the experts are on the card and the host"),
        );
        check(&mut ok, clause, verdict);
        Ok(ok)
    }

    /// `mtp_survives_a_cut_on_the_tier` (the module header): the three servers of [`arm`] under
    /// residency `off` at the fixed width, the reference on plan (a) and the plain and drafted
    /// ones on `--place bp`.
    fn mtp_survives_a_cut_on_the_tier(dir: &Path) -> Result<bool, GateError> {
        const CLAUSE: &str = "mtp_survives_a_cut_on_the_tier";
        const FIXED: (Option<&str>, bool) = (Some("off"), true);
        let mut ok = true;
        let mut runs = Vec::new();
        for (name, draft, place) in [
            ("host", None, None),
            ("off", None, Some("bp")),
            ("mtp", Some("mtp"), Some("bp")),
        ] {
            let name = format!("{CLAUSE}-{name}");
            match arm(dir, (CLAUSE, &name), (draft, FIXED.0, FIXED.1), place)? {
                Some(a) => runs.push(a),
                None => {
                    check(&mut ok, CLAUSE, false);
                    return Ok(ok);
                }
            }
        }
        let (host, off, on) = (&runs[0], &runs[1], &runs[2]);
        let tiered = off.plan.tiered && on.plan.tiered && !host.plan.tiered && off.plan.card > 0;
        let drafted = drafted_follows_plain(
            CLAUSE,
            FIXED,
            (off, on),
            (
                tiered,
                "the bp plans name an expert tier card beside the stage's experts, the reference's none",
            ),
        );
        // The tier card's experts are the card's arithmetic where the reference's are the host's, and
        // two arithmetics part at a step a flat-logit fixture makes arbitrary (so does a plan (a)
        // with more experts on the card against this reference), so the ids are held to the first
        // id of each request, and the experts to the host's share of the picks: the tier plan
        // leaves the host `off.plan.host` experts where the reference leaves it `host.plan.host`,
        // so the picks the host reads through the file's mapping fall by that ratio, within
        // [`HOST_SLACK`] for the router's skew.
        let mut ids_ok = true;
        for i in 0..USERS.len() {
            let (tier, reference) = (&off.turns[i].reply, &host.turns[i].reply);
            let agree = tier.ids.len().min(reference.ids.len());
            let (upto, gap, near) = tier.follows(&reference.ids, agree, NEAR_TIE);
            println!(
                "{CLAUSE} request {i}: the tier load's ids against the host-only load's agree to \
                 step {upto} of {agree}, the runner-up gap there {gap:?} (a near tie is below \
                 {NEAR_TIE}) {near}"
            );
            ids_ok &= upto > 0;
        }
        let share = off.picks.1 * host.plan.host < HOST_SLACK * host.picks.1 * off.plan.host;
        println!(
            "{CLAUSE}: the host served {} of {} picks on the tier plan ({} experts of the host's) \
             and {} of {} on the reference ({}) {share}",
            off.picks.1, off.picks.0, off.plan.host, host.picks.1, host.picks.0, host.plan.host
        );
        ids_ok &= share && off.picks.0 > 0 && host.picks.0 > 0;
        check(&mut ok, CLAUSE, drafted && ids_ok);
        Ok(ok)
    }

    /// What the plan of a tier card that the cards before it leave no expert is refused with:
    /// `--place bp` on a fixture whose stage holds every routed expert.
    const EMPTY_TIER: &str = "holds no expert: the cards before it hold every routed expert";

    /// What the placement's refusal of `--place bp` on a census of one card names.
    const PLACED: &str = "--place bp";

    /// `place_bp_follows_the_cards` (the module header); `two` whether the census holds two cards.
    fn place_bp_follows_the_cards(dir: &Path, two: bool) -> Result<bool, GateError> {
        const CLAUSE: &str = "place_bp_follows_the_cards";
        let mut ok = true;
        let seat = SeatGate {
            model: "flint",
            kinds: record::BLOOMERY_SERVE_HYBRID,
        };
        let file = ref_model_path()?;
        let ctx = CTX.to_string();
        let args = ["--ctx", ctx.as_str(), "--place", "bp"];
        let (status, log, text) = seat.plan(dir, "place-bp", &file, &args)?;
        let plan = log.first(&record::PLAN)?;
        let follows = if two {
            let devices: Vec<String> = match &plan {
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
        check(&mut ok, CLAUSE, follows);
        if !two {
            return Ok(ok);
        }
        // The fixture's stage holds every expert, so the plan above is the empty tier's refusal;
        // a card budget of the stage's floor and HOST_EXTRA leaves the stage a part and gives
        // the tier card the rest it has room for.
        let tier = host_plan(dir, "place-bp-budget", false, Some("off"), Some("bp"))?;
        let all = host_plan(dir, "place-a-budget", false, Some("off"), Some("a"))?;
        println!(
            "place-bp at a card budget of {} B: {} routed experts on the stage card, {} on the \
             host, an expert tier card {}; plan (a) at {} B: {} and {} on the host",
            tier.budget, tier.card, tier.host, tier.tiered, all.budget, all.card, all.host
        );
        let held = tier.tiered && tier.card > 0 && tier.card + tier.host < all.card + all.host;
        check(&mut ok, "place_bp_holds_tier_experts", held);
        Ok(ok)
    }

    /// `tool_call_round_trips` (the module header): the scripted engine over the file's own chat
    /// template, `serve_client::scripted_call`'s method.
    fn tool_call_round_trips(split: &Split, dir: &Path) -> Result<bool, GateError> {
        let template = model::arch::spec(split)
            .map_err(|e| format!("the header: {e}"))?
            .spec
            .chat
            .template
            .ok_or("the file carries no chat template")?;
        let mut ok = true;
        check(
            &mut ok,
            "tool_call_round_trips",
            bloomery_gpu_gates::serve_client::scripted_call(&template, dir)?,
        );
        Ok(ok)
    }

    /// The gate's flags: `--dir <dir>` and `--tier`.
    fn args() -> Result<(std::path::PathBuf, bool), GateError> {
        const USAGE: &str = "usage: gate_exaone_serve --dir <dir> [--tier]";
        let (mut dir, mut tier) = (None, false);
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            match flag.as_str() {
                "--dir" => {
                    let v = it
                        .next()
                        .ok_or_else(|| format!("{flag} needs a value: {USAGE}"))?;
                    dir = Some(std::path::PathBuf::from(v));
                }
                "--tier" => tier = true,
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        Ok((dir.ok_or(USAGE)?, tier))
    }

    /// The `--tier` run: the two tier clauses alone, on a census of two cards.
    fn tier_run(dir: &Path) -> Result<(), GateError> {
        if gpu_census::census()?.len() < 2 {
            return Err(
                "gate_exaone_serve --tier: the census holds one card, and the tier clauses plan \
                 `--place bp` on two (run it with both cards in view: just gate-gpu-exaone-twocard)"
                    .into(),
            );
        }
        let mut ok = place_bp_follows_the_cards(dir, true)?;
        ok &= mtp_survives_a_cut_on_the_tier(dir)?;
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: gate_exaone_serve --tier the seat plans an expert tier card where two cards are \
             in view and holds experts on it, and serves a load of it (the stage's experts, the tier \
             card's and the host's) whose drafted ids are the plain load's and whose host serves \
             the picks of the experts left to it (mtp_survives_a_cut_on_the_tier)"
        );
        Ok(())
    }

    pub fn run() -> Result<(), GateError> {
        let (dir, tier) = args()?;
        std::fs::create_dir_all(&dir)?;
        if tier {
            return tier_run(&dir);
        }
        let file = ref_model_path()?;
        let split = Split::open(&file).map_err(|e| format!("open {}: {e}", file.display()))?;
        let layers = u64::try_from(
            model::arch::spec(&split)
                .map_err(|e| format!("the header: {e}"))?
                .spec
                .layers
                .len(),
        )?;
        drop(split);
        let c = serve_gate::Clauses {
            gate: "gate_exaone_serve",
            // The flint seat serves the file whose router row is K-EXAONE's; the load record
            // names the body's own architecture word.
            model: "flint",
            kinds: record::BLOOMERY_SERVE_HYBRID,
            load: &record::LOAD_HYBRID,
            listening: &record::LISTENING_HYBRID,
            arch: "exaone-moe",
            layers,
            think: Some(THINK_TAGS),
            // The six-layer cut never writes the turn end inside the tokens a request asks
            // for (the module header); the shared clause is left by name.
            turn_end: None,
            ctx: CTX,
            // The body's ring takes a shared prefix back only where the sequence stands
            // (`app::arch::flint`'s `Batch::cut`, the rollback's own rule).
            cut: serve_gate::Cut::WhereItStands,
        };
        let two = gpu_census::census()?.len() >= 2;
        let mut ok = serve_gate::plan_names_the_slots(&c, &dir)?;
        ok &= place_bp_follows_the_cards(&dir, two)?;
        ok &= serve_gate::served_with(&c, &dir, ASK)?;
        ok &= tools_request_is_served(&dir)?;
        ok &= mtp_serves_the_plain_ids(&dir)?;
        let name = "mtp_survives_a_cut_on_the_host_tier";
        ok &= mtp_survives_a_cut_on_the_host_tier(&dir, name, Some("off"), true)?;
        let name = "mtp_survives_a_cut_with_residency";
        ok &= mtp_survives_a_cut_on_the_host_tier(&dir, name, None, true)?;
        let name = "mtp_survives_a_cut_at_the_default_width";
        ok &= mtp_survives_a_cut_on_the_host_tier(&dir, name, Some("off"), false)?;
        let split = Split::open(&file).map_err(|e| format!("open {}: {e}", file.display()))?;
        ok &= tool_call_round_trips(&split, &dir)?;
        serve_gate::finish(&c, ok)
    }
}
