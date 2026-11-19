//! `gate_hy4_serve` — the Hy4-flash seat of `bloomery-serve` against the model it serves, on the
//! hy-v4-flash file of the profile (`BLOOMERY_REF_MODEL`; the fixture under the fixture tier): the
//! seat plans and serves resident sequences, loads the file on the placed body, answers a chat
//! turn with its reasoning split at the model's tags, keeps each slot's rows apart, and reads a
//! tool call in the template's own markup.
//!
//!     gate_hy4_serve --dir <dir> [--mmproj <file>]
//!
//! `--mmproj` is the `hy_v4_flash` projector of the file's repo; without it the image clause is
//! skipped by name.
//!
//! Every server is `bloomery-serve --model hy4 -m <file> …` beside this binary,
//! `BLOOMERY_REF_MODEL` removed from its environment. Logs per server in `<dir>/<name>/`
//! (`server.err`, the answers).
//!
//! The shared clauses (`shared/serve_gate.rs`) take the seat's facts: the common hybrid seat's
//! records, the think span the description names (`hy4::spec::REASONING`), the turn end
//! as the vocabulary spells it, and the body's cut: a window layer's ring and the engram history
//! take no position back (`app::arch::hy4`'s `Batch::cut`), so a request that shares less than
//! the slot holds re-feeds its prompt from an empty model ([`serve_gate::Cut::WhereItStands`]).
//!
//! - `plan_names_the_slots`: the shared clause. The body's sequences are resident
//!   (`bloomery_gpu_hy4::Seq`), so `--parallel 2` plans two slots a slot's context wide.
//! - `load_and_listen`, `chat_answers_a_turn`, `chat_splits_the_reasoning_at_the_models_tags`,
//!   `chat_ends_at_the_turn_end`, `live_slot_keeps_the_prefix`: the shared clauses
//!   ([`serve_gate::served_with`], a think-span request: the budget closes the span).
//! - `two_slots_load`: a server at `--ctx` 2·[`CTX`] `--parallel 2` prints its `load` and
//!   `listening` records with two slots of [`CTX`] each.
//! - `slots_keep_their_own_rows`: on that server, P on a slot, then P and its answer there (every
//!   held position kept), Q on the other slot, then the whole of P's history (P, both answers)
//!   back on P's slot: it keeps every position the slot holds (the body's cut keeps where the
//!   sequence stands, so the request continues it) and answers the ids the same request answers
//!   uncached, from an empty model (the server walks its prompts as steps, `--prefill steps`, so a kept
//!   row and a re-fed one are the same bits, and both tiers judge it; a pass's batch sums differ
//!   from a step's in rounding). FAIL-first: a sequence exchange that leaves the stores in
//!   place (`Slots::swap_seq` moving the position alone) runs P's return over Q's rows and the ids
//!   differ.
//! - `tools_request_is_answered`: on that server, a chat request that carries `tools` is a 200
//!   with a choice: the server reads the template's markup (`serve::glmxml::KvSpelling::Hy4`),
//!   never a refusal of tool calls. FAIL-first: a server that reads no markup for the template
//!   refuses the request and the clause is red.
//! - `tool_calls_follow_the_template` (`Tag::FileBound`: the file's template teaches the markup and
//!   its model writes it): the same request answers `finish_reason` `tool_calls` with one
//!   `get_weather` call whose `city` argument is a string.
//! - `tool_call_is_parsed`: a server over a scripted engine whose every reply is the template's own
//!   call markup (`SCRIPTED_CALL`, the span closed first, as the generation prompt leaves it open)
//!   with the file's chat template, asked one tool turn, answers `finish_reason` `tool_calls` and
//!   one call `get_weather` with the arguments `{"city":"Seoul"}` (`serve_client::scripted_call`);
//!   it reads nothing a model chose to reply, so both tiers judge it. FAIL-first: a scripted engine
//!   that replies the Qwen XML call in place of the reply it is given answers no call, and the
//!   clause is red.
//! - `host_share_serves_a_turn`: at a card budget of the plan's floor (the bytes a budget of
//!   1 MiB is refused for) plus half the all-card plan's expert bytes, the plan holds experts on
//!   both the card and the host; a server at that budget loads that split (its `load` record's
//!   counts) and answers [`N`] greedy ids, so a token's routed experts run on the host leg and the
//!   card leg both and join (`bloomery_gpu_hy4::Placed`). FAIL-first: a placed load that refuses a
//!   layer with host experts never listens, and the clause is red.
//! - `image_request_is_served` (with `--mmproj`): a server of one slot at [`CTX`] with the
//!   projector prints the `vision_seat` and `vision` records, and a chat request holding a
//!   448 × 448 image answers 200 with completion tokens whose prompt counts at least the image's
//!   [`IMAGE_ROWS`] rows (14 × 14 tokens and a newline row after each of the 14 grid rows). The
//!   span's rows stand in place of the pad positions' embeddings in the placed pass. Both tiers
//!   judge it: the fixture keeps the real file's width and vocabulary, so the real projector
//!   seats against it, and the clause reads no id the model chose. FAIL-first: a pass that
//!   stages one row fewer than the span holds, or a load without the pass, refuses the call by
//!   name and the clause is red.

#[cfg(all(feature = "deepseek41", feature = "hy4"))]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_hy4_serve", gate::run())
}

#[cfg(not(all(feature = "deepseek41", feature = "hy4")))]
fn main() -> std::process::ExitCode {
    eprintln!("gate_hy4_serve: build with --features deepseek41,hy4");
    std::process::ExitCode::from(64)
}

#[cfg(all(feature = "deepseek41", feature = "hy4"))]
#[allow(
    dead_code,
    reason = "the shared clauses serve several gates; each takes the helpers it reads"
)]
#[path = "shared/serve_gate.rs"]
mod serve_gate;

#[cfg(all(feature = "deepseek41", feature = "hy4"))]
mod gate {
    use std::path::Path;
    use std::time::Duration;

    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::serve_client::{
        SeatGate, Served, base64, chatted, check, curl, file_template, greedy, scripted_call,
        server_log, tokenized, tool_chat,
    };
    use bloomery_gpu_gates::tier::{self, Tag};
    use bloomery_gpu_gates::{GateError, ref_model_path};
    use gguf::Split;
    use model::arch::hy4::place::PlanInputs;
    use serde_json::{Value, json};
    use std::path::PathBuf;

    use crate::serve_gate;

    /// A slot's positions the server is asked for.
    const CTX: u64 = 4096;

    /// The token that closes the model's turn, as its vocabulary spells it.
    const TURN_END: &str = "<｜hy_end:6124c78e｜>";

    /// The call `tool_call_is_parsed` scripts, `get_weather` of Seoul in the template's markup,
    /// after the close of the span the generation prompt opens.
    const SCRIPTED_CALL: &str = "</think:6124c78e><tool_calls:6124c78e><tool_call:6124c78e>\
        get_weather<arg_key:6124c78e>city</arg_key:6124c78e><arg_value:6124c78e>Seoul\
        </arg_value:6124c78e></tool_call:6124c78e></tool_calls:6124c78e>";

    /// Polls of a second that bound the wait for a server to listen: a cold load of the host set.
    const LISTEN_POLLS: usize = 1000;

    /// The tokens each slot clause's request answers, and the tool request's.
    const N: usize = 8;
    const TOOL_TOKENS: u64 = 192;

    /// The checked-in 448 × 448 image of the image clause, from this crate's root, and the rows
    /// its span holds: 14 × 14 tokens and a newline row after each of the 14 grid rows.
    const IMAGE: &str = "../../tools/ref/clefvis/images/free-448x448.png";
    const IMAGE_ROWS: u64 = 14 * 14 + 14;
    /// The image server's `--image-max-tokens`: above [`IMAGE_ROWS`], below the tower's default.
    const IMAGE_MAX_TOKENS: u64 = 1024;
    const _: () = assert!(IMAGE_ROWS <= IMAGE_MAX_TOKENS);
    const IMAGE_ASK: &str = "Describe this image in one short sentence.";

    /// The slot clause's two prompts, which share no first id.
    const P_TEXT: &str = "The capital of France is Paris, and the capital of Italy is";
    const Q_TEXT: &str = "Zebras run across the open plain while the lions watch from the";

    /// The bytes a `--plan` refusal names as the load's floor ("… of the N bytes that takes").
    fn floor_of(text: &str) -> Result<u64, GateError> {
        text.split(" bytes that takes")
            .next()
            .and_then(|head| head.rsplit(" of the ").next())
            .and_then(|n| n.trim().parse().ok())
            .ok_or_else(|| format!("the refusal names no floor: {text}").into())
    }

    /// The card budget of `host_share_serves_a_turn` and the plan's card and host experts at it.
    fn host_share_plan(c: &serve_gate::Clauses, dir: &Path) -> Result<(u64, u64, u64), GateError> {
        let file = ref_model_path()?;
        let ctx = c.ctx.to_string();
        let args = ["--ctx", ctx.as_str()];
        let lever = bloomery_levers::CARD_BUDGET;
        let (status, _, text) =
            seat(c).plan_under((dir, "plan-floor"), &file, &args, &[(lever, Some("1M"))])?;
        if status.success() {
            return Err("a card budget of 1 MiB planned a load: the floor is not read".into());
        }
        let floor = floor_of(&text)?;
        let (status, log, text) =
            seat(c).plan_under((dir, "plan-all"), &file, &args, &[(lever, None)])?;
        if !status.success() {
            return Err(format!("the all-card plan: {text}").into());
        }
        let all = log
            .first(&record::PLAN)?
            .ok_or("--plan printed no plan line")?;
        let budget = floor + all.u64("card_expert_bytes")? / 2;
        let budget_text = budget.to_string();
        let (status, log, text) = seat(c).plan_under(
            (dir, "plan-share"),
            &file,
            &args,
            &[(lever, Some(budget_text.as_str()))],
        )?;
        if !status.success() {
            return Err(format!("the plan at a card budget of {budget} B: {text}").into());
        }
        let share = log
            .first(&record::PLAN)?
            .ok_or("--plan printed no plan line")?;
        println!(
            "host share: floor {floor} B, all-card experts {} ({} B), budget {budget} B: card \
             {} host {}",
            all.u64("card_experts")?,
            all.u64("card_expert_bytes")?,
            share.u64("card_experts")?,
            share.u64("host_experts")?
        );
        Ok((
            budget,
            share.u64("card_experts")?,
            share.u64("host_experts")?,
        ))
    }

    /// `host_share_serves_a_turn` (the module header).
    fn host_share_serves_a_turn(c: &serve_gate::Clauses, dir: &Path) -> Result<bool, GateError> {
        let mut ok = true;
        let (budget, card, host) = host_share_plan(c, dir)?;
        let planned = card > 0 && host > 0;
        let d = dir.join("host-share");
        std::fs::create_dir_all(&d)?;
        let file = ref_model_path()?;
        let (mut cmd, mut all) = seat(c).command(&file)?;
        cmd.env(bloomery_levers::CARD_BUDGET, budget.to_string());
        let ctx = c.ctx.to_string();
        for a in ["--port", "0", "--ctx", &ctx] {
            all.push(a.to_owned());
        }
        let all: Vec<&str> = all.iter().map(String::as_str).collect();
        let mut s = Served::spawn_cmd(cmd, &all, &d)?;
        let err_log = d.join("server.err");
        let served = match s.address(&err_log, LISTEN_POLLS, Duration::from_secs(1)) {
            Ok(addr) => {
                let url = |p: &str| format!("http://{addr}{p}");
                let load = server_log(&err_log, c.kinds)?.one(c.load)?;
                let split = load.u64("card_experts")? == card && load.u64("host_experts")? == host;
                let (a, _) = greedy(&url, tokenized(&url, P_TEXT)?, N, &d, "host_share", false)?;
                println!(
                    "host share: load {:?} the plan's split {split}; answered {} ids {:?}",
                    load.line(),
                    a.tokens.len(),
                    a.tokens
                );
                split && a.tokens.len() == N
            }
            Err(e) => {
                println!("host share: the server never listened: {e}");
                false
            }
        };
        println!("host-share server stopped: {}", s.stop()?);
        check(&mut ok, "host_share_serves_a_turn", planned && served);
        Ok(ok)
    }

    /// `image_request_is_served` (the module header): a server with `mmproj`, one image request.
    fn image_request_is_served(
        c: &serve_gate::Clauses,
        dir: &Path,
        mmproj: &Path,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let d = dir.join("image");
        std::fs::create_dir_all(&d)?;
        let file = ref_model_path()?;
        let (cmd, mut all) = seat(c).command(&file)?;
        let ctx = c.ctx.to_string();
        let m = mmproj.to_str().ok_or("the --mmproj path is not UTF-8")?;
        // The tower's scratch at its 4,096-token default passes the fixture seat's card budget; the clause's
        // image is [`IMAGE_ROWS`] rows, so the cap that still holds it keeps the plan inside the budget.
        let cap = IMAGE_MAX_TOKENS.to_string();
        for a in [
            "--port",
            "0",
            "--ctx",
            &ctx,
            "--mmproj",
            m,
            "--image-max-tokens",
            &cap,
        ] {
            all.push(a.to_owned());
        }
        let all: Vec<&str> = all.iter().map(String::as_str).collect();
        let mut s = Served::spawn_cmd(cmd, &all, &d)?;
        let err_log = d.join("server.err");
        let served = match s.address(&err_log, LISTEN_POLLS, Duration::from_secs(1)) {
            Ok(addr) => {
                let url = format!("http://{addr}/v1/chat/completions");
                let log = server_log(&err_log, c.kinds)?;
                let records = log.first(&record::VISION)?.is_some()
                    && log.first(&record::VISION_SEAT)?.is_some();
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
                let (st, text) = curl(&url, Some(&body), false)?;
                let reply: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
                let prompt = reply["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
                let completion = reply["usage"]["completion_tokens"].as_u64().unwrap_or(0);
                println!(
                    "image chat: HTTP {st}, prompt tokens {prompt} (the image's span is \
                     {IMAGE_ROWS}) in {:?} ms, {completion} completion tokens; vision and \
                     vision_seat records {records}",
                    reply["timings"]["prompt_ms"].as_f64()
                );
                if st != 200 {
                    println!("image chat answer: {text}");
                }
                records && st == 200 && prompt >= IMAGE_ROWS && completion > 0
            }
            Err(e) => {
                println!("image: the server never listened: {e}");
                false
            }
        };
        println!("image server stopped: {}", s.stop()?);
        check(&mut ok, "image_request_is_served", served);
        Ok(ok)
    }

    /// The gate's arguments: `--dir`, the logs' directory, and `--mmproj`, the projector of the
    /// image clause.
    fn args() -> Result<(PathBuf, Option<PathBuf>), GateError> {
        const USAGE: &str = "usage: gate_hy4_serve --dir <dir> [--mmproj <file>]";
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
        Ok((dir.ok_or(USAGE)?, mmproj))
    }

    fn seat(c: &serve_gate::Clauses) -> SeatGate {
        SeatGate {
            model: c.model,
            kinds: c.kinds,
        }
    }

    /// `slots_keep_their_own_rows` (the module header), on the server of two slots at `url`.
    fn slots_keep_their_own_rows(
        url: &dyn Fn(&str) -> String,
        dir: &Path,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let (p, q) = (tokenized(url, P_TEXT)?, tokenized(url, Q_TEXT)?);
        if p.is_empty() || p.first() == q.first() {
            return Err(format!(
                "the slot clause's prompts start with ids {:?} and {:?}: they must share no first id",
                p.first(),
                q.first()
            )
            .into());
        }
        let d = dir.join("two-slots");
        let (a1, v1) = greedy(url, p.clone(), N, &d, "slot_p", true)?;
        let cont: Vec<u32> = p.iter().chain(&a1.tokens).copied().collect();
        let (a2, va) = greedy(url, cont.clone(), N, &d, "slot_p_on", true)?;
        let (_, v2) = greedy(url, q, N, &d, "slot_q", true)?;
        let whole: Vec<u32> = cont.iter().chain(&a2.tokens).copied().collect();
        let (back, vb) = greedy(url, whole.clone(), N, &d, "slot_p_back", true)?;
        let (truth, _) = greedy(url, whole.clone(), N, &d, "slot_p_fresh", false)?;
        let apart = serve_gate::slot_of(&v1)? != serve_gate::slot_of(&v2)?;
        let held = serve_gate::slot_of(&va)? == serve_gate::slot_of(&v1)?
            && a2.cache_n == (cont.len() - 1) as u64;
        let returned = serve_gate::slot_of(&vb)? == serve_gate::slot_of(&v1)?
            && back.cache_n == (whole.len() - 1) as u64
            && back.tokens == truth.tokens;
        println!(
            "slots: P on slot {} and Q on slot {} apart {apart}; P and its answer kept {} of {} \
             {held}; P's history back on slot {} kept {} of {} ids, ids {:?} against the \
             uncached {:?} {returned}",
            serve_gate::slot_of(&v1)?,
            serve_gate::slot_of(&v2)?,
            a2.cache_n,
            cont.len(),
            serve_gate::slot_of(&vb)?,
            back.cache_n,
            whole.len(),
            back.tokens,
            truth.tokens,
        );
        check(
            &mut ok,
            "slots_keep_their_own_rows",
            apart && held && returned,
        );
        Ok(ok)
    }

    /// `tools_request_is_answered` and `tool_calls_follow_the_template` (the module header), on
    /// the server at `url`.
    fn tool_clauses(url: &dyn Fn(&str) -> String, dir: &Path) -> Result<bool, GateError> {
        let mut ok = true;
        let answer = chatted(
            url,
            &dir.join("two-slots"),
            "tools",
            &tool_chat(TOOL_TOKENS),
        );
        let (answered, called) = match &answer {
            Ok(v) => {
                let choice = &v["choices"][0];
                let calls = choice["message"]["tool_calls"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default();
                let function = calls.first().map(|c| &c["function"]);
                let name = function.and_then(|f| f["name"].as_str());
                let city = function
                    .and_then(|f| f["arguments"].as_str())
                    .and_then(|a| serde_json::from_str::<Value>(a).ok())
                    .and_then(|a| a["city"].as_str().map(str::to_owned));
                println!(
                    "tools: finish {:?}, {} call(s), function {name:?}, city {city:?}; content \
                     {:?}",
                    choice["finish_reason"],
                    calls.len(),
                    choice["message"]["content"]
                );
                (
                    choice.is_object(),
                    choice["finish_reason"] == "tool_calls"
                        && calls.len() == 1
                        && name == Some("get_weather")
                        && city.is_some(),
                )
            }
            Err(e) => {
                println!("tools: {e}");
                (false, false)
            }
        };
        check(&mut ok, "tools_request_is_answered", answered);
        if tier::run_clause("tool_calls_follow_the_template", Tag::FileBound)? {
            check(&mut ok, "tool_calls_follow_the_template", called);
        }
        Ok(ok)
    }

    /// The clauses on the server of two slots; the server is stopped at the end.
    fn two_slot_clauses(c: &serve_gate::Clauses, dir: &Path) -> Result<bool, GateError> {
        let (mut s, addr, err_log) = match serve_gate::two_slot_server(c, dir, |_, _, args| {
            args.extend(["--prefill".to_owned(), "steps".to_owned()]);
            Ok(())
        }) {
            Ok(v) => v,
            Err(e) => {
                println!("the two-slot server never listened: {e}");
                let mut ok = true;
                check(&mut ok, "two_slots_load", false);
                return Ok(ok);
            }
        };
        let url = |p: &str| format!("http://{addr}{p}");
        let mut ok = serve_gate::two_slots_load(c, &err_log)?;
        ok &= slots_keep_their_own_rows(&url, dir)?;
        ok &= tool_clauses(&url, dir)?;
        println!("two-slot server stopped: {}", s.stop()?);
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let (dir, mmproj) = args()?;
        let file = ref_model_path()?;
        let split = Split::open(&file).map_err(|e| format!("open {}: {e}", file.display()))?;
        let layers = u64::try_from(PlanInputs::describe(&split)?.model.layers)?;
        let template = file_template(&split)?;
        drop(split);
        let c = serve_gate::Clauses {
            gate: "gate_hy4_serve",
            model: "hy4",
            kinds: record::BLOOMERY_SERVE_HYBRID,
            load: &record::LOAD_HYBRID,
            listening: &record::LISTENING_HYBRID,
            arch: model::arch::hy4::ARCH,
            layers,
            think: Some(["<think:6124c78e>", "</think:6124c78e>"]),
            turn_end: Some(TURN_END),
            ctx: CTX,
            // A window layer's ring and the engram history take no position back: a cut keeps
            // the position the model stands at, or nothing (`app::arch::hy4`'s `Batch::cut`).
            cut: serve_gate::Cut::WhereItStands,
        };
        std::fs::create_dir_all(&dir)?;
        let mut ok = serve_gate::plan_names_the_slots(&c, &dir)?;
        ok &= serve_gate::served_with(&c, &dir, serve_gate::Ask::THINK_SPAN)?;
        check(
            &mut ok,
            "tool_call_is_parsed",
            scripted_call(&template, SCRIPTED_CALL, &dir)?,
        );
        ok &= two_slot_clauses(&c, &dir)?;
        ok &= host_share_serves_a_turn(&c, &dir)?;
        if let Some(m) = &mmproj {
            ok &= image_request_is_served(&c, &dir, m)?;
        } else {
            println!("image_request_is_served: skipped, no --mmproj");
        }
        if !ok {
            return Err(bloomery_gpu_gates::checks_failed());
        }
        println!(
            "PASSED: gate_hy4_serve the seat plans two resident slots, loads at the context it \
             was asked for, answers a chat turn with its reasoning split at the model's tags, \
             keeps the prefix by its cut's rule, keeps each slot's rows apart, answers a tools \
             request, parses the template's call markup, serves a turn with experts on the card and the host, \
             and answers an image request under --mmproj"
        );
        Ok(())
    }
}
