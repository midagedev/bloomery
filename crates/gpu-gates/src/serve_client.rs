//! The HTTP side of the binaries that drive a server (`gate_ds41_serve`,
//! `soak_ds41_serve` on `bloomery-serve-ds41`, `gate_qwen3_serve`,
//! `gate_glm5next_serve` and `gate_mimo2_serve` on `bloomery-serve`): the
//! server started beside the calling binary and killed by its handle on every
//! way out, its address read from its stderr (with the `server` record of the
//! wall the spawn took to get there), one request through curl, a
//! `/metrics` value, the server's stderr read by its binary's record kinds
//! ([`server_log`]), and the requests the seat gates share: the chat turn's
//! ids ([`rendered`], [`tokenized`]), a greedy completion ([`greedy`]), the
//! chat answer held to a completion's ids ([`chat_is_those_ids`]), an image
//! file held to its digest and sent as base64 ([`sha256`], [`base64`]) and the
//! prefix clauses of a seat that keeps a shared prefix
//! ([`edit_and_extension`]) and of one whose prompt cache holds a saved
//! state's checkpoints ([`prefix_restore`]).

use std::fs::File;
use std::io::Write;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::record::{self, Fields, Kind, Log, Record};
use crate::{GateError, verdict};

/// The longest gap between two reads of the server's stderr in
/// [`Served::address`]: the `server` record's wall ends at the read that found
/// the line, so a gate's own `poll` (seconds) must not set its resolution.
const READ_GAP: Duration = Duration::from_millis(100);

/// The server this binary started; killed and reaped on every way out.
pub struct Served {
    pub child: Child,
    /// The server binary's file name, as the `server` record names it.
    exe: String,
    /// The instant just before the child was spawned, until
    /// [`Served::address`] prints the `server` record: it prints once.
    spawned: Option<Instant>,
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
        let spawned = Instant::now();
        let child = cmd
            .spawn()
            .map_err(|e| format!("spawn {}: {e}", exe.display()))?;
        let exe = exe
            .file_name()
            .map_or_else(|| exe.display().to_string(), |n| n.to_string_lossy().into());
        Ok(Served {
            child,
            exe,
            spawned: Some(spawned),
        })
    }

    /// The server's path: `bloomery-serve-ds41` beside this binary.
    pub fn exe() -> Result<PathBuf, GateError> {
        Ok(std::env::current_exe()?.with_file_name("bloomery-serve-ds41"))
    }

    /// Waits for the `listening on http://<addr>` line in the server's stderr:
    /// `polls` polls of `poll` bound the wait, and the reads come at most
    /// [`READ_GAP`] apart within them. The first time it finds the line it
    /// prints the `server` record ([`record::SERVER_SPAWN`]) on stdout: the
    /// wall from the spawn to the read that found the line.
    pub fn address(
        &mut self,
        err_log: &Path,
        polls: usize,
        poll: Duration,
    ) -> Result<String, GateError> {
        let gap = poll.min(READ_GAP);
        let per_poll = poll.as_nanos().div_ceil(gap.as_nanos().max(1)).max(1);
        let reads = usize::try_from(per_poll)
            .unwrap_or(usize::MAX)
            .saturating_mul(polls);
        for _ in 0..reads {
            let text = std::fs::read_to_string(err_log).unwrap_or_default();
            if let Some(addr) = text
                .lines()
                .find_map(|l| l.split_once("listening on http://").map(|(_, a)| a.trim()))
            {
                if let Some(spawned) = self.spawned.take() {
                    Record::new(&record::SERVER_SPAWN)
                        .w("exe", &self.exe)
                        .u("pid", self.child.id())
                        .f("listening_s", spawned.elapsed().as_secs_f64())
                        .print();
                }
                return Ok(addr.to_owned());
            }
            if let Some(status) = self.child.try_wait()? {
                return Err(format!(
                    "the server exited ({status}) before listening; {}:\n{text}",
                    err_log.display()
                )
                .into());
            }
            std::thread::sleep(gap);
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

/// The bytes as standard base64 with padding, the alphabet `serve::media`'s strict decoder
/// reads.
pub fn base64(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let n = u32::from(b[0]) << 16 | u32::from(b[1]) << 8 | u32::from(b[2]);
        out.push(A[(n >> 18) as usize & 63] as char);
        out.push(A[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 {
            A[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if c.len() > 2 {
            A[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// The sha256 of the file at `path`, as `sha256sum` prints it; a failed `sha256sum` is an error
/// naming the file.
pub fn sha256(path: &Path) -> Result<String, GateError> {
    let out = Command::new("sha256sum").arg(path).output()?;
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    match text.split_whitespace().next() {
        Some(h) if out.status.success() => Ok(h.to_owned()),
        _ => Err(format!("sha256sum {}: {:?}", path.display(), text.trim()).into()),
    }
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

/// The clause of a seat whose prompt cache carries checkpoints a request can
/// restore: the prompt `a` answered and saved, another request that takes the
/// slot, then a request that shares only the first `shared` ids of `a`.
pub struct PrefixRestore<'a> {
    /// The texts the long prompts repeat: prompt A's, B's (another prompt,
    /// whose first id is not A's) and the ids C ends with in place of A's.
    pub a: &'a str,
    pub b: &'a str,
    pub tail: &'a str,
    /// The ids of A, and the first ones C shares with it.
    pub a_len: usize,
    pub shared: usize,
    /// `cache_n` C keeps: the nearest checkpoint the saved state carries at
    /// or below `shared`.
    pub kept: u64,
    /// The tokens each request makes.
    pub n: usize,
}

/// `text` repeated until it holds `len` ids, cut to them.
fn repeated_ids(
    url: &dyn Fn(&str) -> String,
    text: &str,
    len: usize,
) -> Result<Vec<u32>, GateError> {
    let one = tokenized(url, text)?.len().max(1);
    let mut ids = tokenized(url, &format!("{text} ").repeat(len / one + 2))?;
    if ids.len() < len {
        return Err(format!(
            "{:?} repeated holds {} ids, short of {len}",
            text,
            ids.len()
        )
        .into());
    }
    ids.truncate(len);
    Ok(ids)
}

/// A shared prefix restores from the host cache: A of `t.a_len` ids is
/// answered, B takes the slot (so the slot no longer holds A and A's state is
/// saved), and C — A's first `t.shared` ids and then another tail — keeps
/// `t.kept` ids (`cache_n`), restored from the host cache at the nearest
/// checkpoint of A's state at or below `t.shared`, and answers the ids of C
/// fed fresh on a dropped slot (`erase`). The cache keeps a state's
/// checkpoints at the doubling multiples of the spacing and the last one
/// (`runtime::seqstate::save_points`), so a prefix inside the prompt, below
/// the last checkpoint, still restores.
///
/// Mutant: the save carries the last checkpoint alone (C keeps nothing).
pub fn prefix_restore(
    url: &dyn Fn(&str) -> String,
    dir: &Path,
    ok: &mut bool,
    t: &PrefixRestore,
    erase: &dyn Fn() -> Result<(), GateError>,
) -> Result<(), GateError> {
    let a = repeated_ids(url, t.a, t.a_len)?;
    let b = repeated_ids(url, t.b, t.a_len / 4)?;
    let tail = tokenized(url, t.tail)?;
    if t.shared >= t.a_len || tail.is_empty() || a[t.shared] == tail[0] || a[0] == b[0] {
        return Err(format!(
            "the prefix clause's prompts: C shares {} of A's {} ids and then {:?}, A's next id {:?};              B starts {:?} against A's {:?}: C must diverge at {} and B at 0",
            t.shared,
            t.a_len,
            tail.first(),
            a.get(t.shared),
            b.first(),
            a.first(),
            t.shared
        )
        .into());
    }
    let c: Vec<u32> = a[..t.shared].iter().chain(&tail).copied().collect();
    erase()?;
    let (first, _) = greedy(url, a, t.n, dir, "prefix_a", true)?;
    let (other, _) = greedy(url, b, t.n, dir, "prefix_b", true)?;
    let (shared, _) = greedy(url, c.clone(), t.n, dir, "prefix_c", true)?;
    erase()?;
    let (cold, _) = greedy(url, c, t.n, dir, "prefix_c_cold", false)?;
    println!(
        "prefix: A {} ids, B {} ids, C shares {} of A and keeps {} (want {}); its ids {:?}, cold {:?}",
        first.prompt.len(),
        other.prompt.len(),
        t.shared,
        shared.cache_n,
        t.kept,
        shared.tokens,
        cold.tokens
    );
    check(
        ok,
        "prefix_c_restores_the_nearest_checkpoint_from_the_host_cache",
        !first.tokens.is_empty() && !other.tokens.is_empty() && shared.cache_n == t.kept,
    );
    check(
        ok,
        "prefix_c_ids_are_a_fresh_runs",
        !cold.tokens.is_empty() && shared.tokens == cold.tokens && cold.cache_n == 0,
    );
    Ok(())
}
