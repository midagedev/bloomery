//! `gate_qwen3_serve` — the qwen3 seat of `bloomery-serve` against
//! `generate_qwen3moe`, on each model file given, one process at a time on
//! the card the runner pins.
//!
//!     gate_qwen3_serve --model <gguf> [--model <gguf> ...] --dir <dir>
//!
//! For each file it starts `bloomery-serve --model qwen3 -m <file> --port 0
//! --parallel 1` (beside this binary, `BLOOMERY_REF_MODEL` removed from its
//! environment: the file is named once, by `-m`; the plain engine pinned,
//! the prefix clauses holding the one-slot path's keeps) and holds:
//!
//! - `load_and_listen`: the server prints its `load` record for the file's
//!   architecture before its `listening` record;
//! - `unplaced_default_names_itself`: with no `--place` the default is the
//!   whole-card load printing no `plan` record (a card with room; today's
//!   lines), or the auto-placed plan whose one record names
//!   `whole_does_not_fit` (FAIL-first: a default that plans without the why);
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
//!   (`/detokenize`);
//! - on a qwen3moe file, whose seat keeps a prefix of what the slot holds,
//!   the prefix clauses: the edit clause — the turn of [`EDIT_A`] answered,
//!   then resent with its user message changed to [`EDIT_B`], which diverges
//!   at `j` inside the rows the first run's prompt call wrote with at least
//!   [`GEMM_FROM`] ids after it — keeps `j` (`cache_n`) and its ids are the
//!   same ids fed fresh (`cache_prompt: false`: a cut call of at least
//!   `GEMM_FROM` rows is the GEMM walk wherever it is cut); and the
//!   extension clause — the turn resent with its reply and [`LATER`]'s user
//!   turn — keeps every position the slot held (`cache_n == held`), its ids
//!   printed only, for the rows the first run's steps wrote are not a fresh
//!   prompt call's rows;
//! - on a qwen35moe file, whose seat keeps a prefix back to the checkpoints
//!   a marked prompt call took (every 512 positions and its end), the
//!   prefix clauses: the stripped clause — the turn of [`EDIT_A`] answered,
//!   then resent with a reasoning-free reply ([`STRIPPED_35`]) in place of
//!   the reply, so the shared prefix ends at the turn's prompt end — keeps
//!   the checkpoint there (`cache_n == len(p1) - 1`) and its ids are the
//!   same ids fed fresh (a re-fed call of at least [`GEMM_FROM`] rows is
//!   the wide walk wherever it is cut); and the extension clause — the turn
//!   resent with its reply and [`LATER`]'s user turn — keeps every position
//!   the slot held (`cache_n == held`: the ask reaches the standing
//!   position, which no cut takes back), its ids printed only.
//!   FAIL-first mutants, each red on its line: a keep rule that grants the
//!   common prefix instead of the checkpoint makes the session's cut refuse
//!   by name and both clauses' requests fail; the cut's restore omitted
//!   leaves the re-fed ids a fresh run's (the stripped clause's ids red).
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
    use gguf::Split;
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

    /// The edit clause's turn, and the same turn with its user message
    /// changed: the resend diverges inside the rows the first run's prompt
    /// call wrote, with the rest of the prompt after it.
    const EDIT_A: &str =
        "Name the three primary colors of light, and say in one sentence why a screen mixes them.";
    const EDIT_B: &str =
        "Name the three primary colors of ink, and say in one sentence why a page reflects them.";

    /// The user turn the extension clause appends after the reply, as the
    /// template renders it past the reply's end.
    const LATER: &str = "<|im_end|>\n<|im_start|>user\nAnd which of the three does a screen show when it shows none of them?\n<|im_end|>\n";

    /// The stripped clause's resend on a qwen35moe file (the module header):
    /// a reasoning-free reply in place of the reply, then a later user turn
    /// — the template's rendering of the reply with its reasoning dropped.
    /// Its first id is a word of the answer, never the reply's first id.
    const STRIPPED_35: &str = "Red, green and blue; a screen mixes them because each of its \
                               pixels emits those three lights side by side.<|im_end|>\n\
                               <|im_start|>user\nAnd which of the three does a screen show when \
                               it shows none of them?<|im_end|>\n<|im_start|>assistant\n";

    /// The prompt ids a pass takes at most: the rendered turn must be
    /// longer, so the clause covers the ubatch walk.
    const PASS_IDS: usize = 8;

    /// The fewest rows the seat's prompt call runs as the GEMM walk
    /// (`app::arch::qwen3moe::GEMM_FROM`): the rows a kept prefix leaves
    /// behind are a whole fresh run's.
    const GEMM_FROM: usize = app::arch::qwen3moe::GEMM_FROM;

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

    /// The seat's default `--ctx` against the file and this gate's card:
    /// unset, the whole-card load takes the file's trained context capped
    /// to what the card had free — at least the 4096 floor, a multiple of
    /// 1024, and past the floor when the card has room past it (a 24 GB
    /// card and a Q4_K_M 30B file leave tens of thousands of rows); the
    /// `--ctx` flag still wins; a load under `--place` keeps the floor.
    /// FAIL-first: a search that hands back the trained context uncapped
    /// makes the default arm's load a plan the card cannot hold (the spawn
    /// never listens), and one that hands back nothing leaves the default
    /// at the floor.
    fn ctx_default(model: &Path, dir: &Path, ok: &mut bool) -> Result<(), GateError> {
        // The trained context read from the file beside the server, not the
        // server's own echo of it.
        let split = Split::open(model).map_err(|e| format!("open {}: {e}", model.display()))?;
        let trained = split
            .arch_get_u64("context_length")
            .and_then(|v| u64::try_from(v).ok())
            .ok_or_else(|| format!("{}: no context_length", model.display()))?;
        let props_ctx = |url: &dyn Fn(&str) -> String| -> Result<u64, GateError> {
            let (st, body) = curl(&url("/props"), None, false)?;
            let v = json_of("/props", st, &body)?;
            Ok(v["n_ctx"].as_u64().unwrap_or(u64::MAX))
        };
        // Each arm loads the model, so each takes its own directory.
        let arms: [(&str, &[&str]); 3] = [
            ("default", &[]),
            ("flag", &["--ctx", "2048"]),
            ("placed", &["--place", "a"]),
        ];
        for (name, extra) in arms {
            let d = dir.join(format!("ctx-{name}"));
            std::fs::create_dir_all(&d)?;
            let err_log = d.join("server.err");
            let mut cmd = Command::new(beside("bloomery-serve")?);
            cmd.env_remove("BLOOMERY_REF_MODEL");
            let m = model.to_str().ok_or("the model path is not UTF-8")?;
            let mut args: Vec<&str> = vec![
                "--model",
                "qwen3",
                "--port",
                "0",
                "--parallel",
                "1",
                "-m",
                m,
            ];
            args.extend_from_slice(extra);
            let mut s = Served::spawn_cmd(cmd, &args, &d)?;
            let addr = s.address(&err_log, 600, Duration::from_secs(1))?;
            let url = |p: &str| format!("http://{addr}{p}");
            let n = props_ctx(&url)?;
            println!("ctx arm {name}: props n_ctx {n} of trained {trained}");
            check(ok, "ctx_never_passes_the_trained_context", n <= trained);
            match name {
                "default" => check(
                    ok,
                    "ctx_default_is_capped_to_the_card",
                    n >= 4096 && n % 1024 == 0 && n > 4096,
                ),
                "flag" => check(ok, "ctx_flag_wins", n == 2048),
                _ => check(ok, "placed_keeps_the_ctx_floor", n == 4096),
            }
            println!("ctx arm {name}: server stopped: {}", s.stop()?);
        }
        Ok(())
    }

    /// `bloomery-serve --model qwen3 -m <model> --port 0 --parallel 1` beside
    /// this binary, `BLOOMERY_REF_MODEL` removed, its logs in `dir`. The
    /// plain engine is pinned: the prefix clauses below hold the one-slot
    /// path's keeps.
    fn spawn(model: &Path, dir: &Path) -> Result<Served, GateError> {
        let mut cmd = Command::new(beside("bloomery-serve")?);
        cmd.env_remove("BLOOMERY_REF_MODEL");
        let m = model.to_str().ok_or("the model path is not UTF-8")?;
        Served::spawn_cmd(
            cmd,
            &[
                "--model",
                "qwen3",
                "--port",
                "0",
                "--parallel",
                "1",
                "-m",
                m,
            ],
            dir,
        )
    }

    /// What the server answered for one prompt.
    struct Answer {
        prompt: Vec<u32>,
        tokens: Vec<u32>,
        stop: String,
        /// `cache_n`: the positions the request kept of what the slot held.
        cache_n: u64,
    }

    /// `/completion` of `prompt` at temperature 0 — `cache` asks the server
    /// to keep the prefix the prompt shares with the slot — its body in
    /// `<dir>/<name>.json`.
    fn greedy(
        url: &dyn Fn(&str) -> String,
        prompt: Vec<u32>,
        dir: &Path,
        name: &str,
        cache: bool,
    ) -> Result<(Answer, Value), GateError> {
        let body = json!({
            "prompt": prompt, "n_predict": N, "temperature": 0, "return_tokens": true,
            "cache_prompt": cache,
        });
        let (st, text) = curl(&url("/completion"), Some(&body), false)?;
        std::fs::write(dir.join(format!("{name}.json")), &text)?;
        let v = json_of("/completion", st, &text)?;
        let tokens = ids_of(&v["tokens"]);
        let stop = v["stop_type"].as_str().unwrap_or("").to_owned();
        let cache_n = v["timings"]["cache_n"].as_u64().unwrap_or(u64::MAX);
        println!("{name} completion tokens {tokens:?} stop {stop} cache_n {cache_n}");
        Ok((
            Answer {
                prompt,
                tokens,
                stop,
                cache_n,
            },
            v,
        ))
    }

    /// The ids of `messages` as the server's chat template renders them
    /// (`/apply-template`, then `/tokenize`).
    fn rendered(url: &dyn Fn(&str) -> String, messages: Value) -> Result<Vec<u32>, GateError> {
        let (st, body) = curl(
            &url("/apply-template"),
            Some(&json!({ "messages": messages })),
            false,
        )?;
        let text = json_of("/apply-template", st, &body)?["prompt"]
            .as_str()
            .ok_or("/apply-template: no prompt")?
            .to_owned();
        tokenized(url, &text)
    }

    /// `/tokenize` of `text`.
    fn tokenized(url: &dyn Fn(&str) -> String, text: &str) -> Result<Vec<u32>, GateError> {
        let (st, body) = curl(&url("/tokenize"), Some(&json!({ "content": text })), false)?;
        Ok(ids_of(&json_of("/tokenize", st, &body)?["tokens"]))
    }

    /// The prefix clauses (the module header), on the qwen3moe arm alone:
    /// the edit clause — the turn answered, then resent with its user
    /// message changed — diverges at `j`, inside the rows the first run's
    /// prompt call wrote, with at least [`GEMM_FROM`] ids after it, keeps
    /// `j` (`cache_n`) and answers the ids of the same ids fed fresh; the
    /// extension clause — the turn resent with its reply and [`LATER`]'s
    /// user turn — keeps every position the slot held, its ids printed
    /// only.
    fn prefix(url: &dyn Fn(&str) -> String, dir: &Path, ok: &mut bool) -> Result<(), GateError> {
        let p = rendered(url, json!([{ "role": "user", "content": EDIT_A }]))?;
        let q = rendered(url, json!([{ "role": "user", "content": EDIT_B }]))?;
        let later = tokenized(url, LATER)?;
        let j = p.iter().zip(&q).take_while(|(a, b)| a == b).count();
        if j == 0 || j >= p.len() || q.len() < j + GEMM_FROM + 1 {
            return Err(format!(
                "the edit clause's turns diverge at {j} of {} and {} ids; the divergence must \
                 sit inside the first turn's prompt rows with at least {} ids after it",
                p.len(),
                q.len(),
                GEMM_FROM
            )
            .into());
        }
        // The edit clause: the turn answered fresh, resent with the user
        // message changed, and the changed turn fed fresh.
        let (first, _) = greedy(url, p.clone(), dir, "edit_first", false)?;
        let (edit, _) = greedy(url, q.clone(), dir, "edit", true)?;
        let (fresh, _) = greedy(url, q, dir, "edit_fresh", false)?;
        println!(
            "edit: the turns diverge at {j}; the resend kept {}; its ids {:?}; the fresh run's \
             {:?}",
            edit.cache_n, edit.tokens, fresh.tokens
        );
        check(
            ok,
            "edit_resend_keeps_the_row_where_it_diverges",
            first.cache_n == 0 && edit.cache_n == j as u64,
        );
        check(
            ok,
            "edit_resend_ids_are_a_fresh_runs",
            !fresh.tokens.is_empty() && edit.tokens == fresh.tokens && fresh.cache_n == 0,
        );

        // The extension clause: the turn answered again, then resent with
        // its reply and a later user turn.
        let (held_run, _) = greedy(url, p, dir, "extend_first", false)?;
        let held = (held_run.prompt.len() + held_run.tokens.len()).saturating_sub(1) as u64;
        let mut extend = held_run.prompt.clone();
        extend.extend_from_slice(&held_run.tokens);
        extend.extend_from_slice(&later);
        let (resend, _) = greedy(url, extend, dir, "extend", true)?;
        println!(
            "extension: the slot held {held}; the resend kept {} and answered {:?}",
            resend.cache_n, resend.tokens
        );
        check(
            ok,
            "extension_keeps_every_held_position",
            held_run.cache_n == 0 && resend.cache_n == held && !resend.tokens.is_empty(),
        );
        Ok(())
    }

    /// The prefix clauses of the qwen35moe arm (the module header): the
    /// stripped clause — the turn answered, then resent with a
    /// reasoning-free reply in place of the reply, so the shared prefix
    /// ends at the turn's prompt end — keeps the checkpoint there and
    /// answers the ids of the same ids fed fresh; the extension clause —
    /// the turn resent with its reply and [`LATER`]'s user turn — keeps
    /// every position the slot held, its ids printed only.
    fn prefix35(url: &dyn Fn(&str) -> String, dir: &Path, ok: &mut bool) -> Result<(), GateError> {
        let p1 = rendered(url, json!([{ "role": "user", "content": EDIT_A }]))?;
        let later = tokenized(url, LATER)?;
        let stripped = tokenized(url, STRIPPED_35)?;
        if stripped.len() < GEMM_FROM || later.len() < GEMM_FROM {
            return Err(format!(
                "the prefix clauses' resends hold {} and {} ids past the turn; each needs at \
                 least {GEMM_FROM}",
                stripped.len(),
                later.len()
            )
            .into());
        }
        // The stripped clause: the turn answered fresh, resent with the
        // reasoning-free reply, and the same ids fed fresh.
        let (held_run, _) = greedy(url, p1.clone(), dir, "strip35_first", false)?;
        if held_run.tokens.is_empty() || stripped.first() == held_run.tokens.first() {
            return Err(format!(
                "the reply's first id {:?} is the stripped reply's {:?}: the shared prefix \
                 would not end at the turn",
                held_run.tokens.first(),
                stripped.first()
            )
            .into());
        }
        let mut strip = p1.clone();
        strip.extend_from_slice(&stripped);
        let (resend, _) = greedy(url, strip.clone(), dir, "strip35", true)?;
        let (fresh, _) = greedy(url, strip, dir, "strip35_fresh", false)?;
        let turn_end = p1.len() as u64 - 1;
        println!(
            "strip35: the turn ends at {turn_end}; the resend kept {} and answered {:?}; the \
             fresh run's {:?}",
            resend.cache_n, resend.tokens, fresh.tokens
        );
        check(
            ok,
            "qwen35_stripped_resend_keeps_the_turns_end",
            held_run.cache_n == 0 && resend.cache_n == turn_end,
        );
        check(
            ok,
            "qwen35_stripped_resend_ids_are_a_fresh_runs",
            !resend.tokens.is_empty() && resend.tokens == fresh.tokens && fresh.cache_n == 0,
        );

        // The extension clause: the turn resent with its reply and a later
        // user turn. The stripped clause's runs left the slot holding the
        // stripped conversation, so the fresh turn runs again first — the
        // slot back at the reply's end, its prompt call's end mark standing.
        let (held_again, _) = greedy(url, p1.clone(), dir, "extend35_first", false)?;
        if held_again.tokens != held_run.tokens {
            return Err(
                "the extension's fresh turn answered ids the first one did not: the clause                  cannot hold a shared prefix"
                    .into(),
            );
        }
        let held = (held_again.prompt.len() + held_again.tokens.len()).saturating_sub(1) as u64;
        let mut extend = held_again.prompt.clone();
        extend.extend_from_slice(&held_again.tokens);
        extend.extend_from_slice(&later);
        let (resend, _) = greedy(url, extend, dir, "extend35", true)?;
        println!(
            "extension35: the slot held {held}; the resend kept {} and answered {:?}",
            resend.cache_n, resend.tokens
        );
        check(
            ok,
            "qwen35_extension_keeps_every_held_position",
            resend.cache_n == held && !resend.tokens.is_empty(),
        );
        Ok(())
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
        let arch = log
            .lines()
            .find_map(|l| l.strip_prefix("load arch="))
            .and_then(|r| r.split(' ').next())
            .unwrap_or("")
            .to_owned();
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
        // The server ran with no `--place`: the default is exactly one of
        // two things — the whole-card load, printing no `plan` record (a
        // card with room; today's lines), or the auto-placed plan, its one
        // record naming `whole_does_not_fit`. FAIL-first: a default that
        // plans without the why, or prints more than one plan record, turns
        // this red.
        let plans: Vec<&str> = log.lines().filter(|l| l.starts_with("plan ")).collect();
        let default_ok =
            plans.is_empty() || (plans.len() == 1 && plans[0].contains("why=whole_does_not_fit"));
        println!("unplaced default: {} plan record(s) {plans:?}", plans.len());
        check(
            ok,
            "unplaced_default_names_itself",
            load.is_some() && default_ok,
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

        let (chat_ids, v) = greedy(&url, prompt, dir, "completion", false)?;
        let (st, body) = curl(&url("/tokenize"), Some(&json!({ "content": PROSE })), false)?;
        let prose = ids_of(&json_of("/tokenize", st, &body)?["tokens"]);
        println!("prose prompt: {} ids", prose.len());
        check(ok, "the_prose_is_past_a_pass", prose.len() > PASS_IDS);
        let (prose_ids, _) = greedy(&url, prose, dir, "prose", false)?;

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
        if arch == "qwen3moe" {
            prefix(&url, dir, ok)?;
        }
        if arch == "qwen35moe" {
            prefix35(&url, dir, ok)?;
        }
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
            ctx_default(model, &dir, &mut ok)?;
        }
        if ok {
            println!("PASS");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
