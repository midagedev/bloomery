//! `gate_exaone_serve` — the exaone-moe seat of `bloomery-serve` (the flint seat, whose file's
//! router row picks K-EXAONE's body) against the model it serves, on the fixture GGUF of the
//! profile (`BLOOMERY_REF_MODEL`): the seat starts, plans its two resident slots, loads the file
//! on the placed body, answers one chat request with its reasoning split at the model's own
//! `<think>` span and keeps the prefix a request shares with the slot by the body's own rule.
//!
//!     gate_exaone_serve --dir <dir>
//!
//! The seat serves resident sequences, a select and a step each (`pass=turns`). Every
//! server is `bloomery-serve --model flint -m <file>` beside this binary, `BLOOMERY_REF_MODEL`
//! removed from its environment. Logs per server in `<dir>/<name>/` (`server.err`, the answer).
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
    use bloomery_gpu_gates::serve_client::{SeatGate, Served, check, curl};
    use bloomery_gpu_gates::{GateError, ref_model_path};
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

    pub fn run() -> Result<(), GateError> {
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
        let dir = serve_gate::dir_of(&c)?;
        std::fs::create_dir_all(&dir)?;
        let mut ok = serve_gate::plan_names_the_slots(&c, &dir)?;
        ok &= serve_gate::served_with(&c, &dir, ASK)?;
        ok &= tools_request_is_served(&dir)?;
        let split = Split::open(&file).map_err(|e| format!("open {}: {e}", file.display()))?;
        ok &= tool_call_round_trips(&split, &dir)?;
        serve_gate::finish(&c, ok)
    }
}
