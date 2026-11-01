//! `gate_llama_serve` — the qwen3 seat of `bloomery-serve` on a `llama` file (Nanbeige4.3-33B's Q4_K_M, the
//! profile's `BLOOMERY_REF_MODEL`): the seat is the file's by the program that runs its description, with no seat
//! word and no name list, and its chat surface is the template's own — ChatML, a `<think>` span, XML tool calls.
//!
//!     gate_llama_serve --dir <dir>
//!
//! Every server is `bloomery-serve` beside this binary with `BLOOMERY_REF_MODEL` removed from its environment (the
//! file is named by `-m`). Logs per server in `<dir>/<name>/` (`server.err`, the answers).
//!
//! - `llama_file_picks_the_qwen3_seat`: `-m <file> --plan` with no seat word exits 0 and prints the qwen3 seat's
//!   `parallel` line, as `--model qwen3` does; `--model ds41` on the file exits non-zero and names the seat the
//!   file's is (`which serves --model qwen3`); and the server that loads and listens prints the qwen3 seat's `load`
//!   and `listening` records, both naming the file's own architecture word. FAIL-first: a `Model::serves` that
//!   matches the qwen3 seat by an architecture name list refuses the file as one no seat serves, and the clause is
//!   red.
//! - `chatml_starts_with_one_im_start_and_no_bos`: `/apply-template` of one user turn renders the file's ChatML, which
//!   starts with the system turn, `/tokenize` of it (no special tokens added) starts with exactly one `<|im_start|>`
//!   id ([`IM_START`], which is also the file's BOS id), and `/v1/chat/completions` counts those ids as its prompt:
//!   the chat door adds no BOS. The raw door does: `/tokenize` with `add_special` puts the BOS before the same ids,
//!   which is `add_bos_token = true`; llama-server with `add_special` on a rendered chat prompt would do the same, and
//!   serve keeps the HF `apply_chat_template` rule. FAIL-first: a seat whose vocabulary adds the BOS to every encode
//!   leaves two `<|im_start|>` ids at the start and the clause is red.
//! - `thinking_span_is_split_at_the_prompts_end`: the generation prompt ends `<think>\n`, and with
//!   `enable_thinking: false` ends `<think>\n\n</think>\n\n`; a thinking-on reply at temperature 0 with a
//!   `reasoning_budget` of [`BUDGET`] holds reasoning and content both, neither with a think tag, and a thinking-off
//!   reply holds content and no reasoning. FAIL-first: a splitter that starts outside the span whatever the prompt
//!   ends with leaves the reasoning in the content, and the clause is red.
//! - `tool_request_is_rendered_by_the_template`: `/apply-template` with one function lists it inside the file's
//!   `<tools>` block and teaches the XML call format (`<function=…>`, `<parameter=…>`, not the template's JSON form);
//!   `/v1/chat/completions/input_tokens` of the same turn counts exactly the ids `/tokenize` gives that prompt, so the
//!   prompt the server tokenizes is that rendering. Neither route generates: nothing the model replies is read.
//! - `scripted_tool_call_is_one_parsed_call`: a server over a scripted engine (`serve::ScriptedEngine`, whose every
//!   reply is `SCRIPTED_CALL`) with the file's chat template, asked the same tool turn, answers 200 with
//!   `finish_reason` `tool_calls` and one call: `get_weather` with the arguments `{"city":"Seoul"}`. It runs the
//!   server's own template read, think-span entry, XML scanner and response, with no model behind them. FAIL-first: a
//!   scanner that drops a call's parameters answers `{}` and the clause is red.
//! - `cut_tool_call_is_an_answer`: the same scripted server asked the same turn with a `max_tokens` that ends the
//!   generation inside the call, answers as llama-server does (its chat parser reads the end of the text as the
//!   end of the input): 200 with `finish_reason` `length`, never a server error. Cut before the function's name is
//!   whole, the message is the text read, as `content`, and holds no call; cut after the name, it holds that call
//!   with the arguments read so far (`{`) and no content; and a stream folds to the same. FAIL-first: a server
//!   that finishes a cut generation as it finishes one that ended on its own answers 500 and the clause is red.
//! - `dense_whole_load_sizes_its_ubatch`: the file has no routed expert, so its only load is the whole one, and
//!   the prompt ubatch its arena sizes is part of that verdict (`ubatch_pick`). The no-flag server at [`CTX`]
//!   prints one `ubatch` line: the sizes of 4096, 2048, 1024 and 512 it tried from the top, each one's whole
//!   need, what the card had and the pick. Read back from that line alone, the pick is the largest size whose
//!   need is within what the card had and every larger size is over it (`ubatch_pick::judge`), the `whole fit`
//!   line of the load names the pick's need and the card's bytes, and the `load` record's `ubatch` is the pick.
//!   `-ub 4096 --plan` opens no card: where the 4096 need is over the card it exits non-zero and its refusal
//!   names that need, the bytes the card holds, a smaller size as the remedy (not the context's fewer slots
//!   or a smaller file) and the largest size that fits, the no-flag pick or none down to 512, else it exits 0,
//!   and either way it agrees with the no-flag pick; `-ub 512 --plan` exits 0 with 512 the pick, the user's
//!   size over the larger one the card would take.
//!   FAIL-first: a dense load whose ubatch is 4096 again leaves the no-flag start refused on a card whose room
//!   is under that arena, and the clause is red.
//! - `completion_ids_are_the_cli_ids`: `/completion` at temperature 0, [`N`] tokens, of the chat turn's ids gives
//!   the ids `generate_qwen3moe --arm <ids>/N --last-step` prints for the same file (the server's cut: the prompt
//!   less its last id, then a step) — all of them, or a prefix ending at an end-of-generation id: one engine, two
//!   doors. The CLI runs after the server stops, because the file does not fit a card twice.

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_llama_serve", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use std::path::Path;
    use std::time::Duration;

    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::serve_client::{
        SeatGate, Served, agree, chatted, check, curl, cut_call, generate_arms, greedy, ids_of,
        input_tokens, json_of, plan_run, rendered, scripted_call, serve_cmd, server_log, tokenized,
        tool_turn, utf8, whole_terms,
    };
    use bloomery_gpu_gates::ubatch_pick::{self, Source};
    use bloomery_gpu_gates::{GateError, checks_failed, ref_model_path};
    use gguf::Split;
    use serde_json::{Value, json};

    const USAGE: &str = "usage: gate_llama_serve --dir <dir>";

    /// The qwen3 seat, by its word.
    const SEAT: SeatGate = SeatGate {
        model: "qwen3",
        kinds: record::BLOOMERY_SERVE_QWEN3,
    };

    /// The positions the server loads: a slot's cache rows.
    const CTX: usize = 4096;

    /// Polls of a second that bound the wait for a server to listen: the cold load of the file.
    const LISTEN_POLLS: usize = 1000;

    /// The file's `<|im_start|>` id, which its vocabulary also names BOS (`tokenizer.ggml.bos_token_id`).
    const IM_START: u32 = 166_100;

    /// The tokens a greedy completion runs.
    const N: usize = 16;

    /// The chat turn the clauses ask.
    const TURN: &str = "Name three primary colours.";

    /// The tokens a thinking-on reply may run, and the reasoning tokens the budget lets the model spend before the
    /// server writes the close of the span.
    const MAX_TOKENS: u64 = 96;
    const BUDGET: u64 = 16;

    /// The tokens a thinking-off reply runs.
    const OFF_TOKENS: u64 = 32;

    /// The line the qwen3 seat prints before its load, and the words of the refusal that names the file's seat.
    const PARALLEL: &str = "parallel rule=slots";
    const SERVES_QWEN3: &str = "which serves --model qwen3";

    /// The generation prompt's ends.
    const THINK_ON: &str = "<think>\n";
    const THINK_OFF: &str = "<think>\n\n</think>\n\n";

    /// One user turn as the chat messages.
    fn user(text: &str) -> Value {
        json!([{ "role": "user", "content": text }])
    }

    /// `body` through `/apply-template`: the rendered prompt.
    fn applied(url: &dyn Fn(&str) -> String, body: &Value) -> Result<String, GateError> {
        let (st, text) = curl(&url("/apply-template"), Some(body), false)?;
        Ok(json_of("/apply-template", st, &text)?["prompt"]
            .as_str()
            .ok_or("/apply-template: no prompt")?
            .to_owned())
    }

    /// `llama_file_picks_the_qwen3_seat`, the three `--plan` runs (the module header): the seat with no word, the
    /// seat by its word, and another seat's word on the file.
    fn plans_pick_the_seat(dir: &Path, file: &Path) -> Result<bool, GateError> {
        let f = utf8(file)?;
        let run = |name: &str, args: &[&str]| -> Result<(bool, Vec<String>), GateError> {
            let d = dir.join(name);
            std::fs::create_dir_all(&d)?;
            let (status, lines) = plan_run(serve_cmd()?, args, &d.join("server.err"))?;
            println!("{name}: {} --plan: {status}", args.join(" "));
            Ok((status.success(), lines))
        };
        let (unworded, lines) = run("pick", &["-m", f])?;
        let by_file = unworded && lines.iter().any(|l| l.starts_with(PARALLEL));
        let (worded, lines) = run("pick-word", &["--model", "qwen3", "-m", f])?;
        let by_word = worded && lines.iter().any(|l| l.starts_with(PARALLEL));
        let (other, lines) = run("pick-other", &["--model", "ds41", "-m", f])?;
        let refused = !other && lines.iter().any(|l| l.contains(SERVES_QWEN3));
        println!(
            "plans: no word picks the qwen3 seat {by_file}; --model qwen3 {by_word}; --model ds41 is refused naming \
             the file's seat {refused}"
        );
        if !refused {
            println!("pick-other stderr:\n{}", lines.join("\n"));
        }
        Ok(by_file && by_word && refused)
    }

    /// `chatml_starts_with_one_im_start_and_no_bos` (the module header).
    fn chatml_and_no_bos(url: &dyn Fn(&str) -> String, dir: &Path) -> Result<bool, GateError> {
        let text = applied(url, &json!({ "messages": user(TURN) }))?;
        std::fs::write(dir.join("prompt.txt"), &text)?;
        let ids = tokenized(url, &text)?;
        let (st, body) = curl(
            &url("/tokenize"),
            Some(&json!({ "content": text, "add_special": true })),
            false,
        )?;
        let raw = ids_of(&json_of("/tokenize", st, &body)?["tokens"])?;
        let answer = chatted(
            url,
            dir,
            "bos",
            &json!({ "messages": user(TURN), "max_tokens": 1, "temperature": 0, "stream": false }),
        )?;
        let counted = answer["usage"]["prompt_tokens"].as_u64() == u64::try_from(ids.len()).ok();
        let chatml =
            text.starts_with("<|im_start|>system\n") && text.contains("<|im_start|>user\n");
        let one = ids.first() == Some(&IM_START) && ids.get(1) != Some(&IM_START);
        let raw_adds = raw.first() == Some(&IM_START) && raw.get(1..) == Some(ids.as_slice());
        println!(
            "chatml: starts with the system turn {chatml}; {} ids, the first {:?} and the second {:?}; the chat \
             door counts {:?} prompt tokens {counted}; the raw door's ids are the BOS and those ids {raw_adds}",
            ids.len(),
            ids.first(),
            ids.get(1),
            answer["usage"]["prompt_tokens"]
        );
        Ok(chatml && one && counted && raw_adds)
    }

    /// `thinking_span_is_split_at_the_prompts_end` (the module header).
    fn thinking_span(url: &dyn Fn(&str) -> String, dir: &Path) -> Result<bool, GateError> {
        let on = applied(url, &json!({ "messages": user(TURN) }))?;
        let off = applied(
            url,
            &json!({ "messages": user(TURN), "chat_template_kwargs": { "enable_thinking": false } }),
        )?;
        let ends = on.ends_with(THINK_ON) && off.ends_with(THINK_OFF);
        let tagged = |s: &str| s.contains("<think>") || s.contains("</think>");

        let thinking = chatted(
            url,
            dir,
            "thinking",
            &json!({
                "messages": user(TURN), "max_tokens": MAX_TOKENS, "reasoning_budget": BUDGET,
                "temperature": 0, "stream": false,
            }),
        )?;
        let message = &thinking["choices"][0]["message"];
        let (reasoning, content) = (
            message["reasoning_content"].as_str().unwrap_or(""),
            message["content"].as_str().unwrap_or(""),
        );
        let completed = thinking["usage"]["completion_tokens"]
            .as_u64()
            .is_some_and(|n| (1..=MAX_TOKENS).contains(&n));
        let split = completed
            && !reasoning.trim().is_empty()
            && !content.trim().is_empty()
            && !tagged(reasoning)
            && !tagged(content);

        let plain = chatted(
            url,
            dir,
            "plain",
            &json!({
                "messages": user(TURN), "max_tokens": OFF_TOKENS, "temperature": 0, "stream": false,
                "chat_template_kwargs": { "enable_thinking": false },
            }),
        )?;
        let message = &plain["choices"][0]["message"];
        let (off_reasoning, off_content) = (
            message["reasoning_content"].as_str().unwrap_or(""),
            message["content"].as_str().unwrap_or(""),
        );
        let unsplit = off_reasoning.trim().is_empty()
            && !off_content.trim().is_empty()
            && !tagged(off_content);
        println!(
            "thinking: the prompts end {THINK_ON:?} {} and {THINK_OFF:?} {}; thinking on: reasoning {} chars, \
             content {} chars, a think tag in either {} (split {split}); thinking off: reasoning {} chars, content \
             {} chars (unsplit {unsplit})",
            on.ends_with(THINK_ON),
            off.ends_with(THINK_OFF),
            reasoning.trim().len(),
            content.trim().len(),
            tagged(reasoning) || tagged(content),
            off_reasoning.trim().len(),
            off_content.trim().len()
        );
        Ok(ends && split && unsplit)
    }

    /// `tool_request_is_rendered_by_the_template` (the module header).
    fn tool_request(url: &dyn Fn(&str) -> String, dir: &Path) -> Result<bool, GateError> {
        let prompt = applied(url, &tool_turn())?;
        std::fs::write(dir.join("tools-prompt.txt"), &prompt)?;
        // The template names the tags in a sentence before it opens the block: the last `<tools>` opens the list.
        let listed = prompt
            .rsplit_once("<tools>")
            .and_then(|(_, rest)| rest.split_once("</tools>"))
            .is_some_and(|(block, _)| block.contains("get_weather"));
        let xml = prompt.contains("<function=example_function_name>")
            && prompt.contains("<parameter=example_parameter_1>")
            && !prompt.contains("<function-name>");
        let ids = tokenized(url, &prompt)?;
        let held = input_tokens(url, &tool_turn())?;
        let counted = !ids.is_empty() && held == ids.len();
        println!(
            "tool request: the function is listed inside <tools> {listed}; the prompt teaches the XML format {xml}; \
             the chat door counts {held} prompt tokens, /tokenize of the rendering gives {} {counted}",
            ids.len()
        );
        Ok(listed && xml && counted)
    }

    /// What the no-flag server left: its stderr and its `load` record's ubatch.
    struct Started {
        log: String,
        loaded_ubatch: u64,
    }

    /// One `--plan` run of the server with `args` beside the file at [`CTX`] positions, opening no card: its
    /// exit status, its stderr, and the `ubatch` lines on it.
    fn plan_ubatch(
        dir: &Path,
        name: &str,
        file: &str,
        args: &[&str],
    ) -> Result<(bool, String, Vec<ubatch_pick::Choice>), GateError> {
        let d = dir.join(name);
        std::fs::create_dir_all(&d)?;
        let ctx = CTX.to_string();
        let mut all = vec!["-m", file, "--ctx", ctx.as_str()];
        all.extend_from_slice(args);
        let (status, lines) = plan_run(serve_cmd()?, &all, &d.join("server.err"))?;
        let text = lines.join("\n");
        println!("{name}: {} --plan: {status}", all.join(" "));
        Ok((status.success(), text.clone(), ubatch_pick::choices(&text)))
    }

    /// `dense_whole_load_sizes_its_ubatch` (the module header), from the no-flag server's stderr and `load`
    /// record (`started`) and the two `--plan` runs that ask a size.
    fn dense_ubatch(dir: &Path, file: &Path, started: Option<Started>) -> Result<bool, GateError> {
        let Some(Started { log, loaded_ubatch }) = started else {
            println!("ubatch: the no-flag server never listened, so it took no size");
            return Ok(false);
        };
        let found = ubatch_pick::choices(&log);
        let [free] = found.as_slice() else {
            println!(
                "ubatch: the no-flag server printed {} ubatch lines, one wanted",
                found.len()
            );
            return Ok(false);
        };
        let judged = ubatch_pick::judge(free);
        let whole = log
            .lines()
            .find(|l| l.starts_with(&format!("whole fit at ctx {CTX}: ")))
            .map(whole_terms)
            .transpose()?;
        let line_agrees = match (&whole, free.tried.last()) {
            (Some(w), Some(&(_, need))) => w.need == need && w.budget == free.had && w.fits,
            _ => false,
        };
        let loaded = free.pick.and_then(|p| u64::try_from(p).ok()) == Some(loaded_ubatch);
        println!(
            "ubatch: the no-flag server tried {:?} of {} B the card had and took {:?} ({}); its whole-fit line \
             names the pick's need and the card's bytes {line_agrees}; the load record's ubatch is {loaded_ubatch} \
             {loaded}",
            free.tried,
            free.had,
            free.pick,
            match &judged {
                Ok(()) => "the largest within the card".to_owned(),
                Err(e) => e.clone(),
            }
        );
        let f = utf8(file)?;
        let (big_ok, big_text, big) = plan_ubatch(dir, "ubatch-4096", f, &["-ub", "4096"])?;
        let (small_ok, _, small) = plan_ubatch(dir, "ubatch-512", f, &["-ub", "512"])?;
        // The first `ubatch` line of a run is the probe at the asked context; a refused run's search
        // for a context that holds prints the lines of its further probes after it.
        let flagged = |c: &[ubatch_pick::Choice], size: usize| -> Option<ubatch_pick::Choice> {
            c.first()
                .filter(|one| {
                    one.asked == Some(Source::Flag)
                        && one.ctx == CTX
                        && one.pick == Some(size)
                        && ubatch_pick::judge(one).is_ok()
                })
                .cloned()
        };
        let (big, small) = (flagged(&big, 4096), flagged(&small, 512));
        // The 4096 need is the same figure on every run; the card's room decides the run's exit.
        let top = big.as_ref().and_then(|c| c.tried.first().copied());
        let over = big
            .as_ref()
            .zip(top)
            .is_some_and(|(c, (_, need))| need > c.had);
        // The refusal is made at the probe's context over one slot, where the no-flag pick is the
        // largest size of the ladder the card holds.
        let fit = free.pick.map_or_else(
            || {
                format!(
                    "none at {}",
                    ubatch_pick::LADDER[ubatch_pick::LADDER.len() - 1]
                )
            },
            |p| p.to_string(),
        );
        let refused = big.as_ref().zip(top).is_some_and(|(c, (_, need))| {
            !big_ok
                && big_text.contains(&format!(
                    "takes {need} B, and the card holds {} B: ask a smaller --ubatch-size \
                     (the largest that fits here: {fit})",
                    c.had
                ))
        });
        let top_agrees = top.is_some() && top == free.tried.first().copied();
        let largest = (free.pick == Some(4096)) != over;
        println!(
            "ubatch: -ub 4096 over the card {over}: exit ok {big_ok}, the refusal names the card's bytes, the need, a smaller size as the remedy and the largest size that fits ({fit}) {refused}, its \
             4096 need is the server's {top_agrees}, the no-flag pick is 4096 exactly when it is not over \
             {largest}; -ub 512 takes 512 {} (exit ok {small_ok})",
            small.is_some()
        );
        let big_held = if over { refused } else { big_ok };
        Ok(free.asked.is_none()
            && judged.is_ok()
            && (free.ctx, free.slots) == (CTX, 1)
            && line_agrees
            && loaded
            && big_held
            && top_agrees
            && largest
            && small.is_some()
            && small_ok)
    }

    /// Folds a clause that may fail on a request into `ok`: the error is printed, and the clause is red.
    fn clause(ok: &mut bool, name: &str, result: Result<bool, GateError>) {
        match result {
            Ok(pass) => check(ok, name, pass),
            Err(e) => {
                println!("{name}: {e}");
                check(ok, name, false);
            }
        }
    }

    /// What the one server leaves once it is stopped: its completion's ids and stop, the ids it was asked, and what its
    /// start left.
    type ServedRun = (Vec<u32>, String, Vec<u32>, Started);

    /// The clauses that need a server: the one server at [`CTX`] with no seat word, whose records finish
    /// `llama_file_picks_the_qwen3_seat` (`picks`: its `--plan` runs held), then its completion's ids for the CLI to
    /// match. Returns the ids and the stop the server answered, the ids it was asked and what its start left
    /// ([`Started`]), once the server is stopped; `None` where it never listened.
    fn served(
        dir: &Path,
        file: &Path,
        (arch, picks): (&str, bool),
        ok: &mut bool,
    ) -> Result<Option<ServedRun>, GateError> {
        const PICKS: &str = "llama_file_picks_the_qwen3_seat";
        let d = dir.join("serve");
        std::fs::create_dir_all(&d)?;
        let ctx = CTX.to_string();
        let all = ["-m", utf8(file)?, "--port", "0", "--ctx", ctx.as_str()];
        let mut s = Served::spawn_cmd(serve_cmd()?, &all, &d)?;
        let err_log = d.join("server.err");
        let addr = match s.address(&err_log, LISTEN_POLLS, Duration::from_secs(1)) {
            Ok(addr) => addr,
            Err(e) => {
                println!("the server never listened: {e}");
                check(ok, PICKS, false);
                return Ok(None);
            }
        };
        let records = server_log(&err_log, SEAT.kinds)?;
        let load = records.one(&record::LOAD_QWEN3)?;
        let listen = records.one(&record::LISTENING_QWEN3)?;
        let named = load.word("arch")? == arch && listen.word("arch")? == arch;
        println!(
            "serve at {addr}: load {:?}; listening {:?}; both name {arch:?} {named}",
            load.line(),
            listen.line()
        );
        check(ok, PICKS, picks && named);
        let url = |p: &str| format!("http://{addr}{p}");
        clause(
            ok,
            "chatml_starts_with_one_im_start_and_no_bos",
            chatml_and_no_bos(&url, &d),
        );
        clause(
            ok,
            "thinking_span_is_split_at_the_prompts_end",
            thinking_span(&url, &d),
        );
        clause(
            ok,
            "tool_request_is_rendered_by_the_template",
            tool_request(&url, &d),
        );
        let ids = rendered(&url, user(TURN))?;
        let (answer, _) = greedy(&url, ids.clone(), N, &d, "completion", false)?;
        println!("server stopped: {}", s.stop()?);
        let started = Started {
            log: std::fs::read_to_string(&err_log)?,
            loaded_ubatch: load.u64("ubatch")?,
        };
        Ok(Some((answer.tokens, answer.stop, ids, started)))
    }

    pub fn run() -> Result<(), GateError> {
        let dir = SEAT.dir(USAGE)?;
        std::fs::create_dir_all(&dir)?;
        let file = ref_model_path()?;
        let split = Split::open(&file).map_err(|e| format!("open {}: {e}", file.display()))?;
        let arch = split.architecture().unwrap_or("<missing>").to_owned();
        let template = model::arch::spec(&split)
            .map_err(|e| format!("{}: {e}", file.display()))?
            .spec
            .chat
            .template
            .ok_or_else(|| format!("{}: the file carries no chat template", file.display()))?;
        drop(split);
        let mut ok = true;
        let picks = plans_pick_the_seat(&dir, &file).unwrap_or_else(|e| {
            println!("the --plan runs: {e}");
            false
        });
        clause(
            &mut ok,
            "scripted_tool_call_is_one_parsed_call",
            scripted_call(&template, &dir),
        );
        clause(
            &mut ok,
            "cut_tool_call_is_an_answer",
            cut_call(&template, &dir),
        );
        const DENSE: &str = "dense_whole_load_sizes_its_ubatch";
        if let Some((tokens, stop, ids, started)) = served(&dir, &file, (&arch, picks), &mut ok)? {
            clause(&mut ok, DENSE, dense_ubatch(&dir, &file, Some(started)));
            let cli = generate_arms(
                "generate_qwen3moe",
                &file,
                &[ids.as_slice()],
                N,
                u64::try_from(CTX)?,
                &dir,
            )?;
            println!("cli completion tokens {:?}", cli[0]);
            check(
                &mut ok,
                "completion_ids_are_the_cli_ids",
                agree(&tokens, &stop, &cli[0]),
            );
        } else {
            clause(&mut ok, DENSE, dense_ubatch(&dir, &file, None));
        }
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: gate_llama_serve the qwen3 seat is the llama file's by its description's program, its chat \
             surface is the template's (one <|im_start|>, no BOS, the <think> span split, the tool block rendered, a \
             scripted XML call parsed, one the limit cut answered as llama-server does), its prompt ubatch is the user's or the largest the card holds, and its greedy \
             ids are the CLI's"
        );
        Ok(())
    }
}
