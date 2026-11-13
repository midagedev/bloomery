//! `gate_kimi_k3_serve` — the Kimi K3 seat of `bloomery-serve` against the model it serves, on the
//! kimi-k3 file of the profile (`BLOOMERY_REF_MODEL`; the fixture under the fixture tier): the seat
//! starts, loads the file on the placed body, answers one chat request, keeps the prefix a request
//! shares with the slot by the body's own rule and refuses a request for tool calls by name.
//!
//!     gate_kimi_k3_serve --dir <dir>
//!
//! The seat serves one sequence on one card (`--parallel` above 1 is refused by name), its prompt
//! run as placed passes of the widest shape its plan holds, or stepped one id at a time under
//! `--prefill steps`. Every server is `bloomery-serve --model kimik3 -m <file> …` beside
//! this binary, `BLOOMERY_REF_MODEL` removed from its environment. Logs per server in
//! `<dir>/<name>/` (`server.err`, the answer).
//!
//! The clauses are the shared ones (`shared/serve_gate.rs`): the seat refuses a second slot, loads
//! the file at the context it was asked for, answers a chat turn and ends its turn at the model's
//! own end token, and runs the prefix clause `live_slot_keeps_the_prefix` at its own cut's fact:
//! the delta layers' state takes no position back (`app::arch::kimi_k3`'s `Batch::cut`), so a
//! request that shares less than the slot holds re-feeds its prompt from an empty model and the
//! clause holds it to the counting rule alone ([`serve_gate::Cut::WhereItStands`]). K3's
//! description names the reply's sections (`kimi_k3::spec::REASONING`): the seat hands the server the
//! think span's and the response section's tags, so the reasoning-split clause holds the message
//! to a reasoning of its own and a content with neither the span's close nor the response's open
//! in it (`Clauses::think` those two, `Ask::reasons` true). The model's facts are this file's: the turn end is written here as the model's
//! vocabulary spells it.
//!
//! - `parallel_above_one_is_refused`: the shared clause.
//! - `residency_is_off_for_a_fixed_slot_map`: `--plan` at [`CTX`] with `BLOOMERY_RESIDENCY` removed
//!   from the environment prints the `residency lever` record at `off` with why [`UNSET_WHY`] and
//!   no `residency host` record: the body keeps the load's slot map for its life, so the residency
//!   machine runs over none of its placements. `BLOOMERY_RESIDENCY=off` prints `off` with why `set`
//!   and no pool. `BLOOMERY_RESIDENCY=mid-p0-s1` exits non-zero and its stderr names the refusal
//!   (`tiered::refuse_fixed_slot_map`), the plan record absent. FAIL-first: a seat whose residency
//!   pick is the serving word prints it with why `place` and the unset half is red; one that runs
//!   the word it is given loads the churn machine over a body that cannot and the refusal half is
//!   red.
//! - `prompt_runs_as_passes`: a server at [`CTX`] names `prompt=pass` in its `listening` record,
//!   prints the `prompt_pass` record of the shape its plan holds (R rows), and its `load` record
//!   reads the same R rows off the loaded body (`pass_rows`), which `Batch::prompt` runs a prompt
//!   by; a server under `--prefill steps` names `prompt=steps`, prints no `prompt_pass` record and
//!   loads no pass. FAIL-first (`.mutants/k3seatpass-open-stepped.patch`: the seat plans the pass
//!   and opens the stepped body): the pass server's `load` record has no `pass_rows` and the clause
//!   is red.
//! - `passes_answer_as_steps`: the two servers answer the same `/completion` (a prompt of
//!   R + R/2 + 1 ids, two passes and a part, temperature 0, [`GENERATED`] ids) with the same ids
//!   and stop.
//! - `load_and_listen`, `chat_answers_a_turn`, `chat_splits_the_reasoning_at_the_models_tags`
//!   (here: the reasoning in `reasoning_content`, the content with no section tag to leak),
//!   `chat_ends_at_the_turn_end`,
//!   `live_slot_keeps_the_prefix`: the shared clauses, see `shared/serve_gate.rs`. The chat request
//!   carries no reasoning budget: K3's template opens the think span in the generation prompt and
//!   the model closes it itself. FAIL-first: a seat that names no tags serves the span and the
//!   section markup in `content`, and the reasoning-split clause is red.
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
        SeatGate, Served, check, completion_raw, curl, ids_of, json_of, no_residency_host,
        residency_lever_is, server_log, stop_of, tokenized,
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

    /// Why an unset residency is `off`: the body keeps its load's slot map for its life.
    /// PIN(2026-11-08): `prefill_steps` -> `fixed_place`; the seat runs its prompt as passes, and
    /// its `off` is the body's slot map, not a step feed.
    const UNSET_WHY: &str = "fixed_place";

    /// The ids each `/completion` of `passes_answer_as_steps` generates.
    const GENERATED: usize = 8;

    /// The text the pass clause's prompt is tokenized from, repeated to its length.
    const PROMPT_TEXT: &str = "The river carried the boats past the old mill, and the miller \
                               counted each one as it went by. ";

    /// The chat request the seat's turn is asked as: no reasoning budget and no effort (the
    /// template reads its own default), and the message holds reasoning, which the model writes
    /// in the span the generation prompt opens.
    const ASK: serve_gate::Ask = serve_gate::Ask {
        budget: None,
        effort: None,
        reasons: true,
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

    /// `residency_is_off_for_a_fixed_slot_map` (the module header).
    fn residency_is_off_for_a_fixed_slot_map(
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
            && residency_lever_is(&unset, "off", UNSET_WHY)?
            && no_residency_host(&unset)?;
        let off_ok = off_status.success()
            && residency_lever_is(&off, "off", "set")?
            && no_residency_host(&off)?;
        let mid_refused = !mid_status.success()
            && mid_text.contains(RESIDENCY_REFUSED)
            && mid.first(&record::PLAN)?.is_none();
        println!(
            "residency: unset {:?} is off ({UNSET_WHY}) with no pool {unset_ok}; set off {:?} is \
             off (set) with no pool {off_ok}; set mid-p0-s1 exits non-zero, names \
             {RESIDENCY_REFUSED:?} and prints no plan {mid_refused}",
            lever_line(&unset)?,
            lever_line(&off)?,
        );
        check(
            &mut ok,
            "residency_is_off_for_a_fixed_slot_map",
            unset_ok && off_ok && mid_refused,
        );
        Ok(ok)
    }

    /// The prompt a pass clause server answers, from its URL and the `prompt_pass` rows it printed;
    /// `None` asks none.
    type PromptOf<'a> =
        dyn Fn(&dyn Fn(&str) -> String, Option<u64>) -> Result<Option<Vec<u32>>, GateError> + 'a;

    /// What one server of the pass clauses printed and answered.
    struct Walked {
        /// The `listening` record's `prompt` word.
        prompt: Option<String>,
        /// The `prompt_pass` record's rows.
        planned: Option<u64>,
        /// The `load` record's `pass_rows`.
        loaded: Option<u64>,
        /// The `/completion` answer's ids and stop.
        answer: Option<(Vec<u32>, String)>,
    }

    /// A server at [`CTX`] in `<dir>/<name>` with `extra` flags: its walk records, and its
    /// `/completion` of the prompt `ids` makes from the pass rows (none asked when `ids` gives
    /// `None`). A server that never listens is a named error.
    fn walked(
        c: &serve_gate::Clauses,
        (dir, name): (&Path, &str),
        extra: &[&str],
        ids: &PromptOf<'_>,
    ) -> Result<Walked, GateError> {
        let d = dir.join(name);
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
                .chain(extra)
                .map(|a| (*a).to_owned()),
        );
        let all: Vec<&str> = all.iter().map(String::as_str).collect();
        let mut s = Served::spawn_cmd(cmd, &all, &d)?;
        let err_log = d.join("server.err");
        let addr = s
            .address(&err_log, LISTEN_POLLS, Duration::from_secs(1))
            .map_err(|e| format!("{name}: the server never listened: {e}"))?;
        let records = server_log(&err_log, c.kinds)?;
        let prompt = records
            .one(c.listening)?
            .opt_word("prompt")?
            .map(str::to_owned);
        let planned = match records.first(&record::PROMPT_PASS)? {
            Some(r) => Some(r.u64("rows")?),
            None => None,
        };
        let loaded = records.one(c.load)?.opt_u64("pass_rows")?;
        let url = |p: &str| format!("http://{addr}{p}");
        let answer = match ids(&url, planned)? {
            Some(ids) => {
                let (st, text) = completion_raw(&url, &ids, GENERATED, false)?;
                std::fs::write(d.join("completion.json"), &text)?;
                let v = json_of("/completion", st, &text)?;
                Some((ids_of(&v["tokens"])?, stop_of(&v)?.to_owned()))
            }
            None => None,
        };
        println!("{name}: server stopped: {}", s.stop()?);
        Ok(Walked {
            prompt,
            planned,
            loaded,
            answer,
        })
    }

    /// `prompt_runs_as_passes` and `passes_answer_as_steps` (the module header): a pass server and
    /// a `--prefill steps` server, one after the other.
    fn prompt_runs_as_passes(c: &serve_gate::Clauses, dir: &Path) -> Result<bool, GateError> {
        let mut ok = true;
        // The prompt both servers answer: R + R/2 + 1 ids of the pass server's R rows, the text's
        // ids repeated to that length; fixed once the pass server has made it.
        let made = std::cell::RefCell::new(None::<Vec<u32>>);
        let ids = |url: &dyn Fn(&str) -> String,
                   rows: Option<u64>|
         -> Result<Option<Vec<u32>>, GateError> {
            if let Some(ids) = made.borrow().as_ref() {
                return Ok(Some(ids.clone()));
            }
            let Some(rows) = rows else {
                return Ok(None);
            };
            let len = usize::try_from(rows + rows / 2 + 1)?;
            let text = tokenized(url, PROMPT_TEXT)?;
            if text.is_empty() {
                return Err("the pass clause's prompt text tokenizes to no id".into());
            }
            let ids: Vec<u32> = text.iter().copied().cycle().take(len).collect();
            *made.borrow_mut() = Some(ids.clone());
            Ok(Some(ids))
        };
        let pass = walked(c, (dir, "walk-pass"), &[], &ids)?;
        let steps = walked(c, (dir, "walk-steps"), &["--prefill", "steps"], &ids)?;
        let rows = pass.planned;
        let runs = pass.prompt.as_deref() == Some("pass")
            && rows.is_some_and(|r| r > 0)
            && pass.loaded == rows
            && steps.prompt.as_deref() == Some("steps")
            && steps.planned.is_none()
            && steps.loaded.is_none();
        println!(
            "walk: pass server prompt={:?}, prompt_pass rows {rows:?}, load pass_rows {:?}; \
             --prefill steps server prompt={:?}, prompt_pass rows {:?}, load pass_rows {:?}: {runs}",
            pass.prompt, pass.loaded, steps.prompt, steps.planned, steps.loaded
        );
        check(&mut ok, "prompt_runs_as_passes", runs);
        let same = pass.answer.is_some() && pass.answer == steps.answer;
        println!(
            "answer: {} prompt ids; passes {:?}, steps {:?}: {same}",
            made.borrow().as_ref().map_or(0, Vec::len),
            pass.answer,
            steps.answer
        );
        check(&mut ok, "passes_answer_as_steps", same);
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
            // The span's close and the response's open: markup between the reasoning and the
            // answer, in neither.
            think: Some(["<|close|>think<|sep|>", "<|open|>response<|sep|>"]),
            turn_end: Some(TURN_END),
            ctx: CTX,
            // The delta layers' state takes no position back: a cut keeps the position the model
            // stands at, or nothing (`app::arch::kimi_k3`'s `Batch::cut`).
            cut: serve_gate::Cut::WhereItStands,
        };
        let dir = serve_gate::dir_of(&c)?;
        std::fs::create_dir_all(&dir)?;
        let mut ok = serve_gate::parallel_above_one_is_refused(&c, &dir)?;
        ok &= residency_is_off_for_a_fixed_slot_map(&c, &dir)?;
        ok &= prompt_runs_as_passes(&c, &dir)?;
        ok &= serve_gate::served_with(&c, &dir, ASK)?;
        ok &= tools_request_is_refused(&c, &dir)?;
        serve_gate::finish(&c, ok)
    }
}
