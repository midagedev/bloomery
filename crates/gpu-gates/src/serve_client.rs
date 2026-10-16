//! The HTTP side of the binaries that drive a server (`gate_ds41_serve`,
//! `soak_ds41_serve` on `bloomery-serve-ds41`, `gate_qwen3_serve`,
//! `gate_glm5next_serve` and `gate_mimo2_serve` on `bloomery-serve`): the
//! server started beside the calling binary and killed by its handle on every
//! way out, its address read from its stderr, one request through curl, a
//! `/metrics` value, the server's stderr read by its binary's record kinds
//! ([`server_log`]), and the requests the seat gates share: the chat turn's
//! ids ([`rendered`], [`tokenized`]), a greedy completion ([`greedy`]), the
//! chat answer held to a completion's ids ([`chat_is_those_ids`]) and the
//! prefix clauses of a seat that keeps a shared prefix
//! ([`edit_and_extension`]).

use std::fs::File;
use std::io::Write;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use serde_json::{Value, json};

use crate::record::{Fields, Kind, Log};
use crate::{GateError, verdict};

/// The server this binary started; killed and reaped on every way out.
pub struct Served {
    pub child: Child,
}

impl Served {
    /// Starts `bloomery-serve-ds41` beside this binary with `args`, stdout to
    /// `<dir>/server.out` and stderr to `<dir>/server.err`. The child is
    /// killed when this process dies, so a runner's bound that ends this
    /// process does not leave the server holding a card.
    pub fn spawn(args: &[&str], dir: &Path) -> Result<Served, GateError> {
        Self::spawn_cmd(Command::new(Self::exe()?), args, dir)
    }

    /// [`Served::spawn`] of `cmd`: another server binary, or one with the
    /// environment the caller set on it.
    pub fn spawn_cmd(mut cmd: Command, args: &[&str], dir: &Path) -> Result<Served, GateError> {
        let exe = PathBuf::from(cmd.get_program());
        cmd.args(args)
            .stdin(Stdio::null())
            .stdout(File::create(dir.join("server.out"))?)
            .stderr(File::create(dir.join("server.err"))?);
        // SAFETY: the closure runs in the child between fork and exec and calls
        // only `prctl`, which is async-signal-safe and touches no memory of ours.
        unsafe {
            cmd.pre_exec(|| {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) == 0 {
                    Ok(())
                } else {
                    Err(std::io::Error::last_os_error())
                }
            });
        }
        let child = cmd
            .spawn()
            .map_err(|e| format!("spawn {}: {e}", exe.display()))?;
        Ok(Served { child })
    }

    /// The server's path: `bloomery-serve-ds41` beside this binary.
    pub fn exe() -> Result<PathBuf, GateError> {
        Ok(std::env::current_exe()?.with_file_name("bloomery-serve-ds41"))
    }

    /// Waits for the `listening on http://<addr>` line in the server's stderr,
    /// `polls` reads `poll` apart.
    pub fn address(
        &mut self,
        err_log: &Path,
        polls: usize,
        poll: Duration,
    ) -> Result<String, GateError> {
        for _ in 0..polls {
            let text = std::fs::read_to_string(err_log).unwrap_or_default();
            if let Some(addr) = text
                .lines()
                .find_map(|l| l.split_once("listening on http://").map(|(_, a)| a.trim()))
            {
                return Ok(addr.to_owned());
            }
            if let Some(status) = self.child.try_wait()? {
                return Err(format!(
                    "the server exited ({status}) before listening; {}:\n{text}",
                    err_log.display()
                )
                .into());
            }
            std::thread::sleep(poll);
        }
        Err(format!("the server did not listen within {polls} polls").into())
    }

    pub fn stop(&mut self) -> Result<String, GateError> {
        self.child.kill()?;
        Ok(format!("{}", self.child.wait()?))
    }
}

impl Drop for Served {
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(None)) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// One request through curl: the status and the body. The request body goes
/// through curl's stdin, not its argv: a prompt of a serving context's ids is
/// longer than one argument may be (Linux `MAX_ARG_STRLEN`, 128 KiB).
pub fn curl(url: &str, body: Option<&Value>, stream: bool) -> Result<(u16, String), GateError> {
    let mut c = Command::new("curl");
    c.args(["-sS", "--max-time", "600", "-w", "\n%{http_code}"]);
    if stream {
        c.arg("-N");
    }
    if body.is_some() {
        c.args([
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            "@-",
        ]);
    }
    let mut child = c
        .arg(url)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| format!("curl {url}: no stdin"))?;
    if let Some(b) = body {
        stdin.write_all(b.to_string().as_bytes())?;
    }
    drop(stdin);
    let out = child.wait_with_output()?;
    if !out.status.success() {
        return Err(format!(
            "curl {url}: {} {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    let text = String::from_utf8(out.stdout)?;
    let (body, code) = text
        .rsplit_once('\n')
        .ok_or_else(|| format!("curl {url}: no status line"))?;
    Ok((code.trim().parse()?, body.to_owned()))
}

/// `/metrics`' `llamacpp:<name>` value; `None` when it carries none, and a
/// value that is no number refused by name.
pub fn metric(url: &dyn Fn(&str) -> String, name: &str) -> Result<Option<f64>, GateError> {
    let (st, body) = curl(&url("/metrics"), None, false)?;
    if st != 200 {
        return Err(format!("/metrics: HTTP {st}: {body}").into());
    }
    let key = format!("llamacpp:{name} ");
    let Some(v) = body.lines().find_map(|l| l.strip_prefix(&key)) else {
        return Ok(None);
    };
    match v.trim().parse() {
        Ok(x) => Ok(Some(x)),
        Err(e) => Err(format!("/metrics: llamacpp:{name} {v:?}: {e}").into()),
    }
}

/// The server's stderr at `err_log` so far, read by `kinds`, the record
/// kinds its binary registered ([`Log`]); its errors name the file.
pub fn server_log(err_log: &Path, kinds: &[&'static Kind]) -> Result<Log, GateError> {
    let text =
        std::fs::read_to_string(err_log).map_err(|e| format!("{}: {e}", err_log.display()))?;
    Ok(Log::of(&text, kinds).named(err_log.display().to_string()))
}

/// The usable bytes of a `plan` record's stage card (`record::PLAN` or
/// `record::PLAN38`): the last term of its first `devices` item
/// (`stage:<name>:<ordinal>:<usable bytes>`, `record::plan_devices`).
pub fn stage_usable(plan: &Fields) -> Result<u64, GateError> {
    let devices = plan.csv("devices")?;
    let stage = devices
        .first()
        .ok_or_else(|| format!("a `plan` record with no device: {}", plan.line()))?;
    let usable = stage
        .rsplit(':')
        .next()
        .and_then(|n| n.parse().ok())
        .ok_or_else(|| format!("a `plan` record's stage device {stage:?} names no bytes"))?;
    Ok(usable)
}

/// The JSON of a 200 response; any other status is an error naming it.
pub fn json_of(what: &str, status: u16, body: &str) -> Result<Value, GateError> {
    if status != 200 {
        return Err(format!("{what}: HTTP {status}: {body}").into());
    }
    Ok(serde_json::from_str(body).map_err(|e| format!("{what}: {e}: {body}"))?)
}

/// The ids of a JSON array (entries that are not ids are left out).
pub fn ids_of(v: &Value) -> Vec<u32> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().and_then(|i| u32::try_from(i).ok()))
                .collect()
        })
        .unwrap_or_default()
}

/// `a,b,c` or `[a, b, c]` as ids.
pub fn parse_ids(s: &str) -> Result<Vec<u32>, GateError> {
    s.trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<u32>()
                .map_err(|e| format!("id {t:?}: {e}").into())
        })
        .collect()
}

/// One clause's line, `check <name>: PASS|FAIL`, folded into `ok`.
pub fn check(ok: &mut bool, name: &str, pass: bool) {
    println!("check {name}: {}", verdict(pass));
    *ok &= pass;
}

/// The binary `name` beside the running one.
pub fn beside(name: &str) -> Result<PathBuf, GateError> {
    Ok(std::env::current_exe()?.with_file_name(name))
}

/// `bloomery-serve` beside the running binary, `BLOOMERY_REF_MODEL` removed
/// from its environment: the file is named once, by `-m`.
pub fn serve_cmd() -> Result<Command, GateError> {
    let mut cmd = Command::new(beside("bloomery-serve")?);
    cmd.env_remove("BLOOMERY_REF_MODEL");
    Ok(cmd)
}

/// `ids` agree with `reference`, a run that did not stop at the
/// end-of-generation id: all of them, or, when `stop` is `eos`, a prefix
/// ending there.
pub fn agree(ids: &[u32], stop: &str, reference: &[u32]) -> bool {
    match stop {
        "eos" => !ids.is_empty() && reference.starts_with(ids),
        _ => ids == reference,
    }
}

/// `/tokenize` of `text` without BOS.
pub fn tokenized(url: &dyn Fn(&str) -> String, text: &str) -> Result<Vec<u32>, GateError> {
    let (st, body) = curl(
        &url("/tokenize"),
        Some(&json!({ "content": text, "add_special": false })),
        false,
    )?;
    Ok(ids_of(&json_of("/tokenize", st, &body)?["tokens"]))
}

/// The ids of `messages` as the server's chat template renders them
/// (`/apply-template`, then [`tokenized`]).
pub fn rendered(url: &dyn Fn(&str) -> String, messages: Value) -> Result<Vec<u32>, GateError> {
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

/// One greedy `/completion` of `ids`, `n` tokens, at temperature 0 — `cache`
/// asks the server to keep the prefix the prompt shares with the slot: the
/// status and the body as the server sent them.
pub fn completion_raw(
    url: &dyn Fn(&str) -> String,
    ids: &[u32],
    n: usize,
    cache: bool,
) -> Result<(u16, String), GateError> {
    let body = json!({
        "prompt": ids, "n_predict": n, "temperature": 0, "return_tokens": true,
        "cache_prompt": cache,
    });
    curl(&url("/completion"), Some(&body), false)
}

/// [`completion_raw`]'s JSON; any status but 200 is an error naming it.
pub fn completion(
    url: &dyn Fn(&str) -> String,
    ids: &[u32],
    n: usize,
    cache: bool,
) -> Result<Value, GateError> {
    let (st, text) = completion_raw(url, ids, n, cache)?;
    json_of("/completion", st, &text)
}

/// What the server answered for one prompt.
pub struct Answer {
    pub prompt: Vec<u32>,
    pub tokens: Vec<u32>,
    pub stop: String,
    /// `cache_n`: the positions the request kept of what the slot held.
    pub cache_n: u64,
}

/// [`completion_raw`] of `prompt`, its body in `<dir>/<name>.json` and one
/// line of its tokens, stop and `cache_n` printed.
pub fn greedy(
    url: &dyn Fn(&str) -> String,
    prompt: Vec<u32>,
    n: usize,
    dir: &Path,
    name: &str,
    cache: bool,
) -> Result<(Answer, Value), GateError> {
    let (st, text) = completion_raw(url, &prompt, n, cache)?;
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

/// `/v1/chat/completions` of `messages` at temperature 0 and `max_tokens` `n`
/// is a 200 whose `usage` counts `ids`' prompt and `v`'s tokens (`ids` and
/// `v` from [`greedy`] of the same turn's rendered ids), and whose message
/// text (the reasoning and the content) is inside the text of `ids`' tokens
/// (`/detokenize`). Its body is in `<dir>/chat.json`.
pub fn chat_is_those_ids(
    url: &dyn Fn(&str) -> String,
    dir: &Path,
    messages: Value,
    n: usize,
    (ids, v): (&Answer, &Value),
) -> Result<bool, GateError> {
    let body = json!({
        "messages": messages, "max_tokens": n, "temperature": 0, "stream": false,
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
        Some(&json!({ "tokens": ids.tokens })),
        false,
    )?;
    let detok = json_of("/detokenize", st, &text)?["content"]
        .as_str()
        .unwrap_or("")
        .to_owned();
    let usage = &chat["usage"];
    let counted = usage["prompt_tokens"].as_u64() == Some(ids.prompt.len() as u64)
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
    Ok(counted && !said.trim().is_empty() && parts_inside)
}

/// The prefix clauses of a seat that keeps the prefix a request shares with
/// the slot, on the user turn `a` answered and resent as `b`.
pub struct PrefixTurns<'a> {
    /// The turn answered first, and the same turn with its user message
    /// changed: the resend diverges inside the first turn's prompt rows.
    pub a: &'a str,
    pub b: &'a str,
    /// The tokens each request makes.
    pub n: usize,
    /// The ids the changed turn holds past the divergence, at least.
    pub min_after: usize,
}

/// The edit and extension clauses of a seat that keeps a shared prefix: the
/// turn of `t.a` answered, then resent with its user message changed to
/// `t.b`, which diverges at `j` inside the rows the first run's prompt call
/// wrote with at least `t.min_after` ids after it, keeps `j` (`cache_n`) and
/// answers the ids of the same ids fed fresh (`cache_prompt: false`); and the
/// turn resent with its reply and `later`, the ids the template renders past
/// a reply's end into a later user turn, keeps every position the slot held
/// (`cache_n == held`), its ids printed only.
pub fn edit_and_extension(
    url: &dyn Fn(&str) -> String,
    dir: &Path,
    ok: &mut bool,
    t: &PrefixTurns,
    later: &[u32],
) -> Result<(), GateError> {
    let p = rendered(url, json!([{ "role": "user", "content": t.a }]))?;
    let q = rendered(url, json!([{ "role": "user", "content": t.b }]))?;
    let j = p.iter().zip(&q).take_while(|(a, b)| a == b).count();
    if j == 0 || j >= p.len() || q.len() < j + t.min_after + 1 {
        return Err(format!(
            "the edit clause's turns diverge at {j} of {} and {} ids; the divergence must \
             sit inside the first turn's prompt rows with at least {} ids after it",
            p.len(),
            q.len(),
            t.min_after
        )
        .into());
    }
    // The edit clause: the turn answered fresh, resent with the user
    // message changed, and the changed turn fed fresh.
    let (first, _) = greedy(url, p.clone(), t.n, dir, "edit_first", false)?;
    let (edit, _) = greedy(url, q.clone(), t.n, dir, "edit", true)?;
    let (fresh, _) = greedy(url, q, t.n, dir, "edit_fresh", false)?;
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

    // The extension clause: the turn answered again, then resent with its
    // reply and a later user turn.
    let (held_run, _) = greedy(url, p, t.n, dir, "extend_first", false)?;
    let held = (held_run.prompt.len() + held_run.tokens.len()).saturating_sub(1) as u64;
    let mut extend = held_run.prompt.clone();
    extend.extend_from_slice(&held_run.tokens);
    extend.extend_from_slice(later);
    let (resend, _) = greedy(url, extend, t.n, dir, "extend", true)?;
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
