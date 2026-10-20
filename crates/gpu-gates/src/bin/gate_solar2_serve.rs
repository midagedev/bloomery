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
//! - `parallel_above_one_is_refused`: `--parallel 2 --plan` exits non-zero, its stderr names the
//!   body's one sequence, and no `plan` record is printed (the refusal comes before any plan).
//!   FAIL-first: a seat that takes a second slot prints its plan and the clause is red.
//! - `load_and_listen`: one server at `--ctx` [`CTX`], `--port 0`, loads and listens; the `load`
//!   record comes before the `listening` record; the load names the file's architecture word, one
//!   slot, the file's layer count and [`CTX`]; the listening record names one slot and [`CTX`];
//!   `/health` answers 200. FAIL-first: a seat that loads another context than the one it was
//!   asked for names it in the load record and the clause is red.
//! - `chat_splits_the_reasoning_at_the_models_tags`: `/v1/chat/completions` of one user turn at
//!   temperature 0, `max_tokens` [`MAX_TOKENS`] and `reasoning_budget` [`BUDGET`] is a 200 whose
//!   `usage` counts a prompt and at most [`MAX_TOKENS`] completion tokens, whose message holds
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
//!   server stops there and leaves it out of the text. FAIL-first: a vocabulary that lists only the
//!   file's `eos_token_id` (`<|endoftext|>`) lets the model run past its turn end, the token's text
//!   is in the content and the clause is red.

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_solar2_serve", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use std::path::Path;
    use std::time::Duration;

    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::serve_client::{SeatGate, Served, check, curl, json_of, server_log};
    use bloomery_gpu_gates::{GateError, checks_failed, ref_model_path};
    use gguf::Split;
    use model::arch::solar2::place::PlanInputs;
    use serde_json::json;

    const USAGE: &str = "usage: gate_solar2_serve --dir <dir>";

    /// The slot's positions the server is asked for.
    const CTX: u64 = 4096;

    /// The tokens the chat request asks for at most, and the reasoning tokens the budget lets the
    /// model spend before the server writes the close.
    const MAX_TOKENS: u64 = 64;
    const BUDGET: u64 = 16;

    /// Polls of a second that bound the wait for a server to listen: a cold load of the whole
    /// host set.
    const LISTEN_POLLS: usize = 1000;

    /// The chat turn the file answers.
    const TURN: &str = "Name three primary colours.";

    /// The tags of the model's think span, as its vocabulary spells them.
    const THINK_TAGS: [&str; 2] = ["<|think:start|>", "<|think:end|>"];

    /// The token that closes the model's turn, as its vocabulary spells it.
    const TURN_END: &str = "<|im:end|>";

    /// The words of the refusal of a second slot.
    const ONE_SEQUENCE: &str = "holds one sequence";

    /// This gate's seat: the one server's model word and the kinds it prints.
    const SEAT: SeatGate = SeatGate {
        model: "solar2",
        kinds: record::BLOOMERY_SERVE_SOLAR2,
    };

    /// `parallel_above_one_is_refused` (the module header).
    fn parallel_above_one_is_refused(dir: &Path, file: &Path) -> Result<bool, GateError> {
        let mut ok = true;
        let (status, log, text) = SEAT.plan(dir, "refuse-parallel", file, &["--parallel", "2"])?;
        let named = text.contains(ONE_SEQUENCE);
        let no_plan = log.first(&record::PLAN)?.is_none();
        println!(
            "refuse-parallel: exit non-zero {}; names {ONE_SEQUENCE:?} {named}; no plan record \
             {no_plan}",
            !status.success()
        );
        if !named {
            println!("refuse-parallel stderr:\n{text}");
        }
        check(
            &mut ok,
            "parallel_above_one_is_refused",
            !status.success() && named && no_plan,
        );
        Ok(ok)
    }

    /// The clauses on one server: `load_and_listen` and the chat; the server is stopped at the
    /// end.
    fn served(dir: &Path, file: &Path, layers: u64) -> Result<bool, GateError> {
        let mut ok = true;
        let d = dir.join("serve");
        std::fs::create_dir_all(&d)?;
        let (cmd, mut all) = SEAT.command(file)?;
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
                println!("the server never listened: {e}");
                check(&mut ok, "load_and_listen", false);
                return Ok(ok);
            }
        };
        let records = server_log(&err_log, record::BLOOMERY_SERVE_SOLAR2)?;
        let load = records.one(&record::LOAD_SOLAR2)?;
        let listen = records.one(&record::LISTENING_SOLAR2)?;
        let url = |p: &str| format!("http://{addr}{p}");
        let (health, _) = curl(&url("/health"), None, false)?;
        let loaded = load.word("arch")? == "solar-open2"
            && load.u64("slots")? == 1
            && load.u64("layers")? == layers
            && load.u64("ctx")? == CTX;
        let listening = listen.u64("slots")? == 1 && listen.u64("ctx")? == CTX;
        let in_order = load.at() < listen.at();
        println!(
            "serve at {addr}: load {:?}; listening {:?}; load names solar-open2, one slot, \
             {layers} layers and ctx {CTX} {loaded}; listening names one slot and ctx {CTX} \
             {listening}; load before listening {in_order}; /health {health}",
            load.line(),
            listen.line()
        );
        check(
            &mut ok,
            "load_and_listen",
            loaded && listening && in_order && health == 200,
        );
        match chat(&url, &d) {
            Ok((split, ended)) => {
                check(&mut ok, CHAT, split);
                check(&mut ok, ENDED, ended);
            }
            Err(e) => {
                println!("{CHAT}: {e}");
                check(&mut ok, CHAT, false);
                check(&mut ok, ENDED, false);
            }
        }
        println!("server stopped: {}", s.stop()?);
        Ok(ok)
    }

    const CHAT: &str = "chat_splits_the_reasoning_at_the_models_tags";
    const ENDED: &str = "chat_ends_at_the_turn_end";

    /// [`CHAT`] and [`ENDED`] (the module header), from one request.
    fn chat(url: &dyn Fn(&str) -> String, dir: &Path) -> Result<(bool, bool), GateError> {
        let body = json!({
            "messages": [{ "role": "user", "content": TURN }],
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
            "reasoning_budget": BUDGET,
            "stream": false,
        });
        let (status, text) = curl(&url("/v1/chat/completions"), Some(&body), false)?;
        std::fs::write(dir.join("chat.json"), &text)?;
        let answer = json_of("/v1/chat/completions", status, &text)?;
        let message = &answer["choices"][0]["message"];
        let reasoning = message["reasoning_content"].as_str().unwrap_or("");
        let content = message["content"].as_str().unwrap_or("");
        let leaked = THINK_TAGS
            .iter()
            .any(|tag| reasoning.contains(tag) || content.contains(tag));
        let usage = &answer["usage"];
        let counted = usage["prompt_tokens"].as_u64().is_some_and(|n| n > 0)
            && usage["completion_tokens"]
                .as_u64()
                .is_some_and(|n| (1..=MAX_TOKENS).contains(&n));
        let finish = answer["choices"][0]["finish_reason"].as_str().unwrap_or("");
        let ended =
            finish == "stop" && !content.contains(TURN_END) && !reasoning.contains(TURN_END);
        println!(
            "chat: usage {usage}; finish_reason {finish:?}; reasoning {} chars, content {} chars; a \
             think tag in the message {leaked}; the turn end in the message {}; reasoning \
             {reasoning:?}; content {content:?}",
            reasoning.trim().len(),
            content.trim().len(),
            content.contains(TURN_END) || reasoning.contains(TURN_END)
        );
        Ok((
            counted && !reasoning.trim().is_empty() && !content.trim().is_empty() && !leaked,
            ended,
        ))
    }

    pub fn run() -> Result<(), GateError> {
        let dir = SEAT.dir(USAGE)?;
        std::fs::create_dir_all(&dir)?;
        let file = ref_model_path()?;
        let split = Split::open(&file).map_err(|e| format!("open {}: {e}", file.display()))?;
        let layers = u64::try_from(PlanInputs::describe(&split)?.model.layers)?;
        drop(split);
        let mut ok = parallel_above_one_is_refused(&dir, &file)?;
        ok &= served(&dir, &file, layers)?;
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: gate_solar2_serve the seat refuses a second slot, loads at the context it was \
             asked for, and answers a chat turn with its reasoning split at the model's tags"
        );
        Ok(())
    }
}
