//! `gate_qwen3_serve` — the qwen3 seat of `bloomery-serve` against
//! `generate_qwen3moe`, on each model file given, one process at a time on
//! the card the runner pins.
//!
//!     gate_qwen3_serve --model <gguf> [--model <gguf> ...] --dir <dir>
//!
//! For each file it starts `bloomery-serve --model qwen3 -m <file> --port 0`
//! (beside this binary, `BLOOMERY_REF_MODEL` removed from its environment:
//! the file is named once, by `-m`) and holds:
//!
//! - `load_and_listen`: the server prints its `load` record for the file's
//!   architecture before its `listening` record;
//! - the chat turn [`MESSAGES`], rendered by the server's own template
//!   (`/apply-template`, `/tokenize`), is longer than the eight ids a pass
//!   takes, so the prompt call runs the ubatch walk;
//! - `completion_ids_are_the_cli_ids`: `/completion` at temperature 0, [`N`]
//!   tokens, of those ids and of [`PROSE`] (`/tokenize`, a raw continuation,
//!   which a model's next ids follow more loosely than a chat answer, so a
//!   prompt call fed other ids moves them sooner), gives for each the ids
//!   `generate_qwen3moe --arm <ids>/N … --last-step` prints for the same
//!   file (the server's cut: the prompt less its last id, then a step) — all
//!   of them, or a prefix ending at an end-of-generation id;
//! - `chat_is_those_ids`: `/v1/chat/completions` of the same turn at
//!   temperature 0 and `max_tokens` N is a 200 whose `usage` counts the
//!   prompt's ids and the completion's tokens, and whose message text (the
//!   reasoning and the content) is inside the text of those ids
//!   (`/detokenize`).
//!
//! The server is stopped by the handle this binary spawned it with before
//! the CLI loads. Logs per file in `<dir>/<n>/` (`server.err`, `gen.log`,
//! the responses).

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_qwen3_serve: built without the `deepseek41` feature; see `just gate-gpu-qwen3-serve`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen3_serve", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};
    use std::time::Duration;

    use bloomery_gpu_gates::serve_client::{Served, curl, ids_of, json_of, parse_ids};
    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use serde_json::{Value, json};

    const USAGE: &str = "usage: gate_qwen3_serve --model <gguf> [--model <gguf> ...] --dir <dir>";

    /// The tokens each request makes.
    const N: usize = 64;

    /// The chat turn every file answers.
    fn messages() -> Value {
        json!([{ "role": "user", "content": "Name the three primary colors of light, and say in one sentence why a screen mixes them." }])
    }

    /// The raw prompt every file continues.
    const PROSE: &str = "The lighthouse keeper counted the steps as he climbed: one hundred and twelve, the same as every night, until the hundred and thirteenth";

    /// The prompt ids a pass takes at most: the rendered turn must be
    /// longer, so the clause covers the ubatch walk.
    const PASS_IDS: usize = 8;

    struct Args {
        models: Vec<PathBuf>,
        dir: PathBuf,
    }

    fn parse_args() -> Result<Args, GateError> {
        let (mut models, mut dir) = (Vec::new(), None);
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} needs a value: {USAGE}"))?;
            match flag.as_str() {
                "--model" => models.push(PathBuf::from(v)),
                "--dir" => dir = Some(PathBuf::from(v)),
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        match (models.is_empty(), dir) {
            (false, Some(dir)) => Ok(Args { models, dir }),
            _ => Err(USAGE.into()),
        }
    }

    fn check(ok: &mut bool, name: &str, pass: bool) {
        println!("check {name}: {}", verdict(pass));
        *ok &= pass;
    }

    /// A binary beside this one.
    fn beside(name: &str) -> Result<PathBuf, GateError> {
        Ok(std::env::current_exe()?.with_file_name(name))
    }

    /// `bloomery-serve --model qwen3 -m <model> --port 0` beside this binary,
    /// `BLOOMERY_REF_MODEL` removed, its logs in `dir`.
    fn spawn(model: &Path, dir: &Path) -> Result<Served, GateError> {
        let mut cmd = Command::new(beside("bloomery-serve")?);
        cmd.env_remove("BLOOMERY_REF_MODEL");
        let m = model.to_str().ok_or("the model path is not UTF-8")?;
        Served::spawn_cmd(cmd, &["--model", "qwen3", "--port", "0", "-m", m], dir)
    }

    /// What the server answered for one prompt.
    struct Answer {
        prompt: Vec<u32>,
        tokens: Vec<u32>,
        stop: String,
    }

    /// `/completion` of `prompt` at temperature 0, its body in `<dir>/<name>.json`.
    fn greedy(
        url: &dyn Fn(&str) -> String,
        prompt: Vec<u32>,
        dir: &Path,
        name: &str,
    ) -> Result<(Answer, Value), GateError> {
        let body = json!({
            "prompt": prompt, "n_predict": N, "temperature": 0, "return_tokens": true,
            "cache_prompt": false,
        });
        let (st, text) = curl(&url("/completion"), Some(&body), false)?;
        std::fs::write(dir.join(format!("{name}.json")), &text)?;
        let v = json_of("/completion", st, &text)?;
        let tokens = ids_of(&v["tokens"]);
        let stop = v["stop_type"].as_str().unwrap_or("").to_owned();
        println!("{name} completion tokens {tokens:?} stop {stop}");
        Ok((
            Answer {
                prompt,
                tokens,
                stop,
            },
            v,
        ))
    }

    /// The server's clauses for `model`, its logs in `dir`; the answers the
    /// CLI is held to, the chat turn's first.
    fn served(model: &Path, dir: &Path, ok: &mut bool) -> Result<Vec<Answer>, GateError> {
        let err_log = dir.join("server.err");
        let mut s = spawn(model, dir)?;
        // The load reads the whole file: up to ten minutes from a cold cache.
        let addr = s.address(&err_log, 600, Duration::from_secs(1))?;
        let url = |p: &str| format!("http://{addr}{p}");
        let log = std::fs::read_to_string(&err_log)?;
        let load = log.lines().position(|l| l.starts_with("load arch="));
        let listen = log.lines().position(|l| l.contains("listening on http://"));
        println!(
            "server {} at {addr}; load record {:?}",
            model.display(),
            load.and_then(|i| log.lines().nth(i))
        );
        check(
            ok,
            "load_and_listen",
            matches!((load, listen), (Some(a), Some(b)) if a < b),
        );

        let (st, body) = curl(
            &url("/apply-template"),
            Some(&json!({ "messages": messages() })),
            false,
        )?;
        let text = json_of("/apply-template", st, &body)?["prompt"]
            .as_str()
            .ok_or("/apply-template: no prompt")?
            .to_owned();
        std::fs::write(dir.join("prompt.txt"), &text)?;
        let (st, body) = curl(&url("/tokenize"), Some(&json!({ "content": text })), false)?;
        let prompt = ids_of(&json_of("/tokenize", st, &body)?["tokens"]);
        println!("rendered prompt: {} ids", prompt.len());
        check(ok, "the_turn_is_past_a_pass", prompt.len() > PASS_IDS);

        let (chat_ids, v) = greedy(&url, prompt, dir, "completion")?;
        let (st, body) = curl(&url("/tokenize"), Some(&json!({ "content": PROSE })), false)?;
        let prose = ids_of(&json_of("/tokenize", st, &body)?["tokens"]);
        println!("prose prompt: {} ids", prose.len());
        check(ok, "the_prose_is_past_a_pass", prose.len() > PASS_IDS);
        let (prose_ids, _) = greedy(&url, prose, dir, "prose")?;

        let body = json!({
            "messages": messages(), "max_tokens": N, "temperature": 0, "stream": false,
        });
        let (st, text) = curl(&url("/v1/chat/completions"), Some(&body), false)?;
        std::fs::write(dir.join("chat.json"), &text)?;
        let chat = json_of("/v1/chat/completions", st, &text)?;
        let msg = &chat["choices"][0]["message"];
        let said = format!(
            "{}{}",
            msg["reasoning_content"].as_str().unwrap_or(""),
            msg["content"].as_str().unwrap_or("")
        );
        let (st, text) = curl(
            &url("/detokenize"),
            Some(&json!({ "tokens": chat_ids.tokens })),
            false,
        )?;
        let detok = json_of("/detokenize", st, &text)?["content"]
            .as_str()
            .unwrap_or("")
            .to_owned();
        let usage = &chat["usage"];
        let counted = usage["prompt_tokens"].as_u64() == Some(chat_ids.prompt.len() as u64)
            && usage["completion_tokens"].as_u64() == v["tokens_predicted"].as_u64();
        let parts_inside = [
            msg["reasoning_content"].as_str().unwrap_or(""),
            msg["content"].as_str().unwrap_or(""),
        ]
        .iter()
        .all(|p| detok.contains(p.trim()));
        println!(
            "chat usage {usage}; said {} chars, the ids' text {} chars",
            said.len(),
            detok.len()
        );
        check(
            ok,
            "chat_is_those_ids",
            counted && !said.trim().is_empty() && parts_inside,
        );
        println!("server stopped: {}", s.stop()?);
        Ok(vec![chat_ids, prose_ids])
    }

    /// `generate_qwen3moe --arm <prompt>/N … --last-step` on `model`, an arm
    /// a prompt; its `tokens` lines in order.
    fn cli(model: &Path, prompts: &[&[u32]], dir: &Path) -> Result<Vec<Vec<u32>>, GateError> {
        let exe = beside("generate_qwen3moe")?;
        let mut args = Vec::new();
        for p in prompts {
            let ids: Vec<String> = p.iter().map(u32::to_string).collect();
            args.push("--arm".to_owned());
            args.push(format!("{}/{N}", ids.join(",")));
        }
        args.push("--last-step".to_owned());
        let out = Command::new(&exe)
            .args(&args)
            .env("BLOOMERY_REF_MODEL", model)
            .stdin(Stdio::null())
            .output()
            .map_err(|e| format!("spawn {}: {e}", exe.display()))?;
        let log = dir.join("gen.log");
        std::fs::write(&log, &out.stdout)?;
        std::fs::write(dir.join("gen.err"), &out.stderr)?;
        if !out.status.success() {
            return Err(format!(
                "{} {}: {}",
                exe.display(),
                out.status,
                String::from_utf8_lossy(&out.stderr)
            )
            .into());
        }
        let text = String::from_utf8(out.stdout)?;
        let lines: Vec<Vec<u32>> = text
            .lines()
            .filter_map(|l| l.strip_prefix("tokens "))
            .map(parse_ids)
            .collect::<Result<_, _>>()?;
        if lines.len() != prompts.len() {
            return Err(format!(
                "{}: {} `tokens` lines for {} arms",
                log.display(),
                lines.len(),
                prompts.len()
            )
            .into());
        }
        Ok(lines)
    }

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let a = parse_args()?;
        let mut ok = true;
        for (i, model) in a.models.iter().enumerate() {
            let dir = a.dir.join(i.to_string());
            std::fs::create_dir_all(&dir)?;
            println!("== {}", model.display());
            let answers = served(model, &dir, &mut ok)?;
            let prompts: Vec<&[u32]> = answers.iter().map(|a| a.prompt.as_slice()).collect();
            let references = cli(model, &prompts, &dir)?;
            let mut agree = true;
            for (ans, reference) in answers.iter().zip(&references) {
                println!("cli tokens {reference:?}");
                agree &= match ans.stop.as_str() {
                    "eos" => !ans.tokens.is_empty() && reference.starts_with(&ans.tokens),
                    _ => ans.tokens == *reference,
                };
            }
            check(&mut ok, "completion_ids_are_the_cli_ids", agree);
        }
        if ok {
            println!("PASS");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
