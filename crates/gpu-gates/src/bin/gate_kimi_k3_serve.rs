//! `gate_kimi_k3_serve` — the Kimi K3 seat of `bloomery-serve` against the model it serves, on the
//! kimi-k3 file of the profile (`BLOOMERY_REF_MODEL`; the fixture under the fixture tier): the seat
//! starts, loads the file on the placed body, answers one chat request, keeps the prefix a request
//! shares with the slot by the body's own rule and refuses a request for tool calls by name.
//!
//!     gate_kimi_k3_serve --dir <dir>
//!
//! The seat serves one sequence on one card (`--parallel` above 1 is refused by name), its prompt
//! stepped one id at a time. Every server is `bloomery-serve --model kimik3 -m <file> …` beside
//! this binary, `BLOOMERY_REF_MODEL` removed from its environment. Logs per server in
//! `<dir>/<name>/` (`server.err`, the answer).
//!
//! The clauses are the shared ones (`shared/serve_gate.rs`): the seat refuses a second slot, loads
//! the file at the context it was asked for, answers a chat turn and ends its turn at the model's
//! own end token, and runs the prefix clause `live_slot_keeps_the_prefix` at its own cut's fact:
//! the delta layers' state takes no position back (`app::arch::kimi_k3`'s `Batch::cut`), so a
//! request that shares less than the slot holds re-feeds its prompt from an empty model and the
//! clause holds it to the counting rule alone ([`serve_gate::Cut::WhereItStands`]). K3's
//! description names no reasoning tags and the seat serves the reasoning inline as plain text, so
//! the reasoning-split clause holds the content alone (`Clauses::think` is `None`, `Ask::reasons`
//! false). The model's facts are this file's: the turn end is written here as the model's
//! vocabulary spells it.
//!
//! - `parallel_above_one_is_refused`: the shared clause.
//! - `residency_is_off_for_a_stepped_prompt`: `--plan` at [`CTX`] with `BLOOMERY_RESIDENCY` removed
//!   from the environment prints the `residency lever` record at `off` with why `prefill_steps` and
//!   no `residency host` record: each prompt id is a decode pass the adaptive rule would count,
//!   and the body keeps the load's slot map for its life. `BLOOMERY_RESIDENCY=off` prints `off`
//!   with why `set` and no pool. `BLOOMERY_RESIDENCY=mid-p0-s1` exits non-zero and its stderr
//!   names the refusal (`tiered::check_residency_feed`), the plan record absent. FAIL-first: a seat
//!   whose residency pick is the serving word prints it with why `place` and the unset half is red;
//!   one that runs the word it is given loads the churn machine over a body that cannot and the
//!   refusal half is red.
//! - `load_and_listen`, `chat_answers_a_turn`, `chat_splits_the_reasoning_at_the_models_tags`
//!   (here: the content, with no tag to leak), `chat_ends_at_the_turn_end`,
//!   `live_slot_keeps_the_prefix`: the shared clauses, see `shared/serve_gate.rs`. The chat request
//!   carries no reasoning budget: K3's template opens the think span in the generation prompt and
//!   the seat splits nothing.
//! - `tools_request_is_refused`: a chat request that carries `tools` is answered with an error
//!   (HTTP 400 or above, an `error.message`, no `choices`), never a completion with the tool calls
//!   left in `content`: the template's tool-call markup is one no parser reads
//!   (`serve::dsml::ToolFormat::Unparsed`), and the slot still answers a plain turn after it.
//!   FAIL-first: a seat that renders the tools into the prompt and answers 200 is red.

#[cfg(all(feature = "deepseek41", feature = "kimik3"))]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_kimi_k3_serve", gate::run())
}

#[cfg(all(feature = "deepseek41", feature = "kimik3"))]
#[allow(
    dead_code,
    reason = "the shared clauses serve several gates; each takes the helpers it reads"
)]
#[path = "shared/serve_gate.rs"]
mod serve_gate;

#[cfg(all(feature = "deepseek41", feature = "kimik3"))]
mod gate {
    use std::path::Path;
    use std::process::ExitStatus;
    use std::time::Duration;

    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::serve_client::{
        SeatGate, Served, check, curl, no_residency_host, residency_lever_is,
    };
    use bloomery_gpu_gates::{GateError, ref_model_path};
    use gguf::Split;
    use model::arch::kimi_k3::place::PlanInputs;
    use serde_json::{Value, json};

    use crate::serve_gate;

    /// The slot's positions the server is asked for.
    const CTX: u64 = 4096;

    /// The token that closes the model's turn, as its vocabulary spells it.
    const TURN_END: &str = "<|end_of_msg|>";

    /// Polls of a second that bound the wait for a server to listen.
    const LISTEN_POLLS: usize = 1000;

    /// The words of the residency refusal a set word other than `off` gets.
    const RESIDENCY_REFUSED: &str = "BLOOMERY_RESIDENCY=mid-p0-s1 is refused";

    /// The chat request the seat's turn is asked as: no reasoning budget (there is no span to
    /// close) and no effort (the template reads its own default), and no reasoning required of
    /// the message.
    const ASK: serve_gate::Ask = serve_gate::Ask {
        budget: None,
        effort: None,
        reasons: false,
    };

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

    /// The rendered lever line, or a named absence.
    fn lever_line(log: &record::Log) -> Result<String, GateError> {
        Ok(match log.first(&record::RESIDENCY_LEVER)? {
            Some(l) => l.line().to_owned(),
            None => "(no residency lever record)".to_owned(),
        })
    }

    /// `residency_is_off_for_a_stepped_prompt` (the module header).
    fn residency_is_off_for_a_stepped_prompt(
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
        let (unset_status, unset, _) =
            plan_under(&seat, dir, "plan-residency", &file, &["--ctx", &ctx], None)?;
        let (off_status, off, _) = plan_under(
            &seat,
            dir,
            "plan-residency-off",
            &file,
            &["--ctx", &ctx],
            Some("off"),
        )?;
        let (mid_status, mid, mid_text) = plan_under(
            &seat,
            dir,
            "plan-residency-mid",
            &file,
            &["--ctx", &ctx],
            Some("mid-p0-s1"),
        )?;
        let unset_ok = unset_status.success()
            && residency_lever_is(&unset, "off", "prefill_steps")?
            && no_residency_host(&unset)?;
        let off_ok = off_status.success()
            && residency_lever_is(&off, "off", "set")?
            && no_residency_host(&off)?;
        let mid_refused = !mid_status.success()
            && mid_text.contains(RESIDENCY_REFUSED)
            && mid.first(&record::PLAN)?.is_none();
        println!(
            "residency: unset {:?} is off (prefill_steps) with no pool {unset_ok}; set off {:?} is \
             off (set) with no pool {off_ok}; set mid-p0-s1 exits non-zero, names \
             {RESIDENCY_REFUSED:?} and prints no plan {mid_refused}",
            lever_line(&unset)?,
            lever_line(&off)?,
        );
        check(
            &mut ok,
            "residency_is_off_for_a_stepped_prompt",
            unset_ok && off_ok && mid_refused,
        );
        Ok(ok)
    }

    /// `tools_request_is_refused` (the module header): a server of its own at [`CTX`].
    fn tools_request_is_refused(c: &serve_gate::Clauses, dir: &Path) -> Result<bool, GateError> {
        let mut ok = true;
        let d = dir.join("tools");
        std::fs::create_dir_all(&d)?;
        let file = ref_model_path()?;
        let (cmd, mut all) = SeatGate {
            model: c.model,
            kinds: c.kinds,
        }
        .command(&file)?;
        all.extend(
            ["--port", "0", "--ctx", &c.ctx.to_string()]
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
                check(&mut ok, "tools_request_is_refused", false);
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
        let refused = status >= 400 && !message.is_empty() && answer.get("choices").is_none();
        // The refusal leaves the slot usable: a plain turn after it is answered.
        let plain = json!({
            "messages": turn, "temperature": 0, "max_tokens": 8, "stream": false,
        });
        let (after, after_text) = curl(&url, Some(&plain), false)?;
        std::fs::write(d.join("after.json"), &after_text)?;
        let usable = after == 200;
        println!(
            "tools: HTTP {status} {message:?} (an error, no choices {refused}); a plain turn \
             after it: HTTP {after} {usable}; {}",
            s.stop()?
        );
        check(&mut ok, "tools_request_is_refused", refused && usable);
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let file = ref_model_path()?;
        let split = Split::open(&file).map_err(|e| format!("open {}: {e}", file.display()))?;
        let layers = u64::try_from(PlanInputs::describe(&split)?.model.layers)?;
        drop(split);
        let c = serve_gate::Clauses {
            gate: "gate_kimi_k3_serve",
            model: "kimik3",
            kinds: record::BLOOMERY_SERVE_HYBRID,
            load: &record::LOAD_HYBRID,
            listening: &record::LISTENING_HYBRID,
            arch: "kimi-k3",
            layers,
            think: None,
            turn_end: TURN_END,
            ctx: CTX,
            // The delta layers' state takes no position back: a cut keeps the position the model
            // stands at, or nothing (`app::arch::kimi_k3`'s `Batch::cut`).
            cut: serve_gate::Cut::WhereItStands,
        };
        let dir = serve_gate::dir_of(&c)?;
        std::fs::create_dir_all(&dir)?;
        let mut ok = serve_gate::parallel_above_one_is_refused(&c, &dir)?;
        ok &= residency_is_off_for_a_stepped_prompt(&c, &dir)?;
        ok &= serve_gate::served_with(&c, &dir, ASK)?;
        ok &= tools_request_is_refused(&c, &dir)?;
        serve_gate::finish(&c, ok)
    }
}
