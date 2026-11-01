//! The HTTP side of the binaries that drive a server (`gate_ds41_serve`,
//! `soak_ds41_serve` on `bloomery-serve-ds41`, `gate_qwen3_serve`,
//! `gate_glm5next_serve`, `gate_mimo2_serve` and `gate_serve_start` on
//! `bloomery-serve`): the server started beside the calling binary and killed
//! by its handle on every way out, its address read from its stderr (with the
//! `server` record of the wall the spawn took to get there), one request
//! through curl, a `/metrics` value, the server's stderr read by its binary's
//! record kinds ([`server_log`]), and the requests the seat gates share: the
//! chat turn's ids ([`rendered`], [`tokenized`]), a greedy completion ([`greedy`]), the
//! ids the decode CLI prints for the same prompt ([`generate_arms`]), the
//! chat answer held to a completion's ids ([`chat_is_those_ids`]), a greedy chat with its
//! runner-up logprobs read against an oracle's ids ([`greedy_chat`], [`answered`],
//! [`ChatReply::follows`]), an image
//! file held to its digest and sent as base64 ([`gguf::digest::sha256_file`], [`base64`]) and the
//! prefix clauses of a seat that keeps a shared prefix
//! ([`edit_and_extension`]) and of one whose prompt cache holds a saved
//! state's checkpoints ([`prefix_restore`]).

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::record::{self, Kind, Log, Record};
use crate::{GateError, die_with_parent, verdict};

/// An image file's bytes as the base64 a `data:` URL carries.
pub use serve::media::encode_base64 as base64;

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
        die_with_parent(&mut cmd);
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
        record::require(
            record::groups::serve_client::SPAWN,
            "serve_client::Served::address",
        );
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

/// A slot action (`POST /slots/{slot}?action=...`) through [`curl`], printed:
/// its status and its JSON (`Null` when the body is not JSON).
pub fn slot_action(
    url: &dyn Fn(&str) -> String,
    slot: u64,
    action: &str,
    body: &Value,
) -> Result<(u16, Value), GateError> {
    let (st, text) = curl(
        &url(&format!("/slots/{slot}?action={action}")),
        Some(body),
        false,
    )?;
    println!("slots {action} of slot {slot}: HTTP {st}: {text}");
    Ok((st, serde_json::from_str(&text).unwrap_or(Value::Null)))
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

/// The JSON of a 200 response; any other status is an error naming it.
pub fn json_of(what: &str, status: u16, body: &str) -> Result<Value, GateError> {
    if status != 200 {
        return Err(format!("{what}: HTTP {status}: {body}").into());
    }
    Ok(serde_json::from_str(body).map_err(|e| format!("{what}: {e}: {body}"))?)
}

/// The ids of a JSON array of token ids. Refused by name: a value that is
/// no array, and an entry that is no integer in `u32`'s range, so a
/// malformed answer never reaches a comparison as a shorter vector.
pub fn ids_of(v: &Value) -> Result<Vec<u32>, GateError> {
    let a = v
        .as_array()
        .ok_or_else(|| format!("token ids: {v} is no array"))?;
    a.iter()
        .enumerate()
        .map(|(i, x)| {
            x.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| format!("token ids: entry {i} ({x}) is no u32 id in {v}").into())
        })
        .collect()
}

/// A completion's draft counts, `timings.draft_n` and
/// `draft_n_accepted`: both absent is a request that proposed nothing,
/// as the server omits them then. Refused by name: no `timings` object,
/// one count present without the other, a count that is no unsigned
/// integer, no proposal, and more ids accepted than proposed.
pub fn draft_counts(answer: &Value) -> Result<(u64, u64), GateError> {
    let t = &answer["timings"];
    if !t.is_object() {
        return Err(format!("a completion with no timings object: {answer}").into());
    }
    let (d, a) = (&t["draft_n"], &t["draft_n_accepted"]);
    if t.get("draft_n").is_none() && t.get("draft_n_accepted").is_none() {
        return Ok((0, 0));
    }
    match (d.as_u64(), a.as_u64()) {
        (Some(d), Some(a)) if d > 0 && a <= d => Ok((d, a)),
        _ => Err(format!(
            "timings draft_n={d} draft_n_accepted={a}: a completion that proposed carries \
             both counts, at most every proposed id accepted"
        )
        .into()),
    }
}

/// The completion evidence used by the lookup gate.
pub struct LookupCompletion {
    pub tokens: Vec<u32>,
    pub stop: String,
    pub slot: u64,
    pub cache_n: u64,
    pub proposed: u64,
    pub accepted: u64,
}

impl LookupCompletion {
    /// Read every required field before the gate compares completions.
    pub fn read(answer: &Value, slots: u64, ctx: u64) -> Result<Self, GateError> {
        let tokens = ids_of(&answer["tokens"]).map_err(|e| format!("lookup completion: {e}"))?;
        if tokens.is_empty() {
            return Err("lookup completion: tokens is empty".into());
        }
        let stop = answer["stop_type"]
            .as_str()
            .ok_or("lookup completion: stop_type is missing or not a string")?;
        if !matches!(stop, "eos" | "word" | "limit") {
            return Err(format!("lookup completion: stop_type is unknown: {stop}").into());
        }
        let slot = answer["id_slot"]
            .as_u64()
            .filter(|&n| n < slots)
            .ok_or("lookup completion: id_slot is missing, not unsigned or outside slots")?;
        let cache_n = answer["timings"]["cache_n"]
            .as_u64()
            .filter(|&n| n <= ctx)
            .ok_or("lookup completion: timings.cache_n is missing, not unsigned or exceeds ctx")?;
        let (proposed, accepted) = draft_counts(answer)?;
        Ok(Self {
            tokens,
            stop: stop.to_owned(),
            slot,
            cache_n,
            proposed,
            accepted,
        })
    }
}

/// An erase response names the requested slot and an unsigned count within its context.
pub fn slot_erased(answer: &Value, slot: u64, ctx: u64) -> Result<(), GateError> {
    if answer["id_slot"].as_u64() != Some(slot) {
        return Err(
            "lookup erase: id_slot is missing, not unsigned or not the requested slot".into(),
        );
    }
    answer["n_erased"]
        .as_u64()
        .filter(|&n| n <= ctx)
        .ok_or("lookup erase: n_erased is missing, not unsigned or exceeds ctx")?;
    Ok(())
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

pub use crate::beside;

/// One `--plan` run of a seat: `cmd` (its binary and environment) with
/// `args`, then `--plan`; stdin and stdout closed, stderr to `err`. Under
/// `--plan` a seat prints the lines its decisions before the load make and
/// exits before it opens a card, so the run needs no stop. Its exit status
/// and its stderr's lines.
pub fn plan_run(
    mut cmd: Command,
    args: &[&str],
    err: &Path,
) -> Result<(std::process::ExitStatus, Vec<String>), GateError> {
    let program = cmd.get_program().to_string_lossy().into_owned();
    let status = cmd
        .args(args)
        .arg("--plan")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(File::create(err)?)
        .status()
        .map_err(|e| format!("spawn {program}: {e}"))?;
    let lines = std::fs::read_to_string(err)?
        .lines()
        .map(str::to_owned)
        .collect();
    Ok((status, lines))
}

/// `bloomery-serve` beside the running binary, `BLOOMERY_REF_MODEL` removed
/// from its environment: the file is named once, by `-m`.
pub fn serve_cmd() -> Result<Command, GateError> {
    let mut cmd = Command::new(beside("bloomery-serve")?);
    cmd.env_remove("BLOOMERY_REF_MODEL");
    Ok(cmd)
}

/// The file's path as UTF-8, the command line's `-m` value.
pub fn utf8(p: &Path) -> Result<&str, GateError> {
    Ok(p.to_str()
        .ok_or_else(|| format!("{} is not UTF-8", p.display()))?)
}

/// A serve gate's seat: the model word the one server binary takes and the
/// record kinds its server prints. The gate bins' `--dir`, their server command
/// and their `--plan` run are one path over it, whatever seat drives them.
pub struct SeatGate {
    /// `--model <word>`.
    pub model: &'static str,
    /// The kinds the server prints, as [`server_log`] reads them.
    pub kinds: &'static [&'static Kind],
}

impl SeatGate {
    /// The `--dir <dir>` flag every serve gate takes; `usage` names the
    /// binary.
    pub fn dir(&self, usage: &str) -> Result<PathBuf, GateError> {
        let mut dir = None;
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} needs a value: {usage}"))?;
            match flag.as_str() {
                "--dir" => dir = Some(PathBuf::from(v)),
                other => return Err(format!("unknown argument {other:?}: {usage}").into()),
            }
        }
        Ok(dir.ok_or(usage)?)
    }

    /// `bloomery-serve --model <model> -m <file>` beside this binary.
    pub fn command(&self, file: &Path) -> Result<(Command, Vec<String>), GateError> {
        Ok((
            serve_cmd()?,
            vec![
                "--model".to_owned(),
                self.model.to_owned(),
                "-m".to_owned(),
                utf8(file)?.to_owned(),
            ],
        ))
    }

    /// `--plan` of `args`: its exit status, the log of its records and its
    /// text, the server's logs in `<dir>/<name>/`.
    pub fn plan(
        &self,
        dir: &Path,
        name: &str,
        file: &Path,
        args: &[&str],
    ) -> Result<(std::process::ExitStatus, Log, String), GateError> {
        self.plan_under((dir, name), file, args, &[])
    }

    /// [`SeatGate::plan`] with each of `levers` set on the server to its
    /// value, or removed from its environment where the value is `None`, so a
    /// value in this binary's environment cannot stand in for unset; the
    /// text printed when the run fails. A gate names each lever at its call,
    /// a `(lever, value)` pair, where `tools/check-defaults.py` reads its pins.
    pub fn plan_under(
        &self,
        (dir, name): (&Path, &str),
        file: &Path,
        args: &[&str],
        levers: &[(&str, Option<&str>)],
    ) -> Result<(std::process::ExitStatus, Log, String), GateError> {
        let d = dir.join(name);
        std::fs::create_dir_all(&d)?;
        let (mut cmd, mut all) = self.command(file)?;
        for &(lever, value) in levers {
            match value {
                Some(v) => cmd.env(lever, v),
                None => cmd.env_remove(lever),
            };
        }
        all.extend(args.iter().map(|a| (*a).to_owned()));
        let all: Vec<&str> = all.iter().map(String::as_str).collect();
        let err = d.join("server.err");
        let (status, lines) = plan_run(cmd, &all, &err)?;
        println!("{name}: {} --plan: {status}", args.join(" "));
        let text = lines.join("\n");
        if !status.success() {
            println!("{name} stderr:\n{text}");
        }
        Ok((status, server_log(&err, self.kinds)?, text))
    }
}

/// The `residency lever` record is `word` with why `why`.
pub fn residency_lever_is(log: &Log, word: &str, why: &str) -> Result<bool, GateError> {
    Ok(match log.first(&record::RESIDENCY_LEVER)? {
        Some(l) => l.word("residency")? == word && l.word("why")? == why,
        None => false,
    })
}

/// No `residency host` record: the load holds no churn pool.
pub fn no_residency_host(log: &Log) -> Result<bool, GateError> {
    Ok(log.first(&record::RESIDENCY_HOST)?.is_none())
}

/// The `residency lever` record of an unset word on a serving placement: a word the plan
/// resolved (its why is neither `set` nor `fixed_place`), ahead of the plan, and the churn pool's
/// `residency host` record after the plan exactly where the word is not `off`; the word is the
/// serving word ([`bloomery_levers::RESIDENCY_SERVING`]) where its why is `place`.
pub fn residency_lever_resolved(log: &Log) -> Result<bool, GateError> {
    let (lever, plan, host) = (
        log.first(&record::RESIDENCY_LEVER)?,
        log.first(&record::PLAN)?,
        log.first(&record::RESIDENCY_HOST)?,
    );
    let (Some(l), Some(p)) = (&lever, &plan) else {
        return Ok(false);
    };
    let (word, why) = (l.word("residency")?, l.word("why")?);
    let pool = match &host {
        Some(h) => p.at() < h.at(),
        None => false,
    };
    Ok(l.at() < p.at()
        && why != "set"
        && why != "fixed_place"
        && (why != "place" || word == bloomery_levers::RESIDENCY_SERVING)
        && ((word == "off") == host.is_none())
        && (word == "off" || pool))
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

/// `<exe> --arm <prompt>/<n> … --ctx <ctx> --last-step` beside this binary on `model` (`exe`: the decode CLI
/// of the Qwen-family engines), one arm a prompt in `prompts`: the ids each arm prints on its `tokens` line, in
/// order, the CLI's output saved as `<dir>/gen.log` and `<dir>/gen.err`. `--last-step` is the server's cut (the
/// prompt less its last id, then a step), so a server's greedy completion of the same ids is compared with these.
/// `ctx` is the server's loaded context: the CLI's own default plans a different split of the routed experts,
/// and the two legs do not sum in one order.
pub fn generate_arms(
    exe: &str,
    model: &Path,
    prompts: &[&[u32]],
    n: usize,
    ctx: u64,
    dir: &Path,
) -> Result<Vec<Vec<u32>>, GateError> {
    let exe = beside(exe)?;
    let mut args = Vec::new();
    for p in prompts {
        let ids: Vec<String> = p.iter().map(u32::to_string).collect();
        args.push("--arm".to_owned());
        args.push(format!("{}/{n}", ids.join(",")));
    }
    // The server's context, so the CLI's plan is the server's split.
    args.push("--ctx".to_owned());
    args.push(ctx.to_string());
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

/// `/tokenize` of `text` without BOS.
pub fn tokenized(url: &dyn Fn(&str) -> String, text: &str) -> Result<Vec<u32>, GateError> {
    let (st, body) = curl(
        &url("/tokenize"),
        Some(&json!({ "content": text, "add_special": false })),
        false,
    )?;
    ids_of(&json_of("/tokenize", st, &body)?["tokens"])
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

/// The terms of one whole-fit line (`whole fit at ctx …: the whole load
/// on …`): its `need`, its `cache` term (the planes' bytes), that need
/// less the cache (the terms no context moves at or past the floor),
/// what the card had, the census reading inside its parentheses, and
/// whether it says the load fits. A line without its terms is a named
/// error.
pub struct WholeTerms<'a> {
    pub need: u64,
    pub cache: u64,
    pub rest: u64,
    pub budget: u64,
    pub census: &'a str,
    pub fits: bool,
}

/// The terms of `line`, the whole-fit verdict's own words.
pub fn whole_terms(line: &str) -> Result<WholeTerms<'_>, GateError> {
    let term = |after: &str| -> Result<u64, GateError> {
        line.split(after)
            .nth(1)
            .and_then(|t| t.split(" B").next())
            .and_then(|t| t.parse().ok())
            .ok_or_else(|| format!("the whole-fit line names no `{after}… B`: {line}").into())
    };
    let (need, cache) = (term(": need ")?, term(" + cache ")?);
    let (had, census) = line
        .split_once(" B the card had (")
        .ok_or_else(|| format!("the whole-fit line names no bytes the card had: {line}"))?;
    let budget: u64 = had
        .rsplit(", of ")
        .next()
        .and_then(|t| t.parse().ok())
        .ok_or_else(|| format!("the whole-fit line names no bytes the card had: {line}"))?;
    let rest = need
        .checked_sub(cache)
        .ok_or_else(|| format!("the whole-fit line's cache passes its need: {line}"))?;
    Ok(WholeTerms {
        need,
        cache,
        rest,
        budget,
        census: census.trim_end_matches(')'),
        fits: line.contains(": fits: "),
    })
}

/// `body` through `/v1/chat/completions` (a 200), the answer saved as `<dir>/<name>.json`.
pub fn chatted(
    url: &dyn Fn(&str) -> String,
    dir: &Path,
    name: &str,
    body: &Value,
) -> Result<Value, GateError> {
    let (st, text) = curl(&url("/v1/chat/completions"), Some(body), false)?;
    std::fs::write(dir.join(format!("{name}.json")), &text)?;
    json_of("/v1/chat/completions", st, &text)
}

/// The turn the tool-call clauses ask.
const TOOL_ASK: &str = "What is the weather in Seoul right now? Use the tool.";

/// The reply the scripted engine of [`scripted_call`] gives every prompt: the template's XML call
/// format, one call.
const SCRIPTED_CALL: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nSeoul\n</parameter>\n</function>\n</tool_call>";

/// The scripted engine's context, and the tokens its reply may run: its vocabulary is a byte a
/// token, so a tool prompt is a few thousand ids.
const SCRIPT_CTX: usize = 16_384;
const SCRIPT_TOKENS: u64 = 256;

/// The one function the tool clauses list.
fn tools() -> Value {
    json!([{ "type": "function", "function": {
        "name": "get_weather",
        "description": "The current weather of a city.",
        "parameters": {
            "type": "object",
            "properties": { "city": { "type": "string", "description": "the city's name" } },
            "required": ["city"],
        },
    } }])
}

/// The tool turn the clauses ask, thinking off.
pub fn tool_turn() -> Value {
    json!({
        "messages": [{ "role": "user", "content": TOOL_ASK }], "tools": tools(),
        "chat_template_kwargs": { "enable_thinking": false },
    })
}

/// [`tool_turn`] as a chat request at temperature 0 whose reply runs at most `max_tokens`.
pub fn tool_chat(max_tokens: u64) -> Value {
    let mut body = tool_turn();
    body["max_tokens"] = json!(max_tokens);
    body["temperature"] = json!(0);
    body["stream"] = json!(false);
    body
}

/// A server over a scripted engine (`serve::ScriptedEngine`, whose every reply is the template's XML
/// call format) with the chat template `template`, asked [`tool_turn`]: it answers 200 with
/// `finish_reason` `tool_calls` and one call, `get_weather` with the arguments `{"city":"Seoul"}`.
/// It runs the server's own template read, think-span entry, XML scanner and response, with no
/// model behind them, so the clause reads nothing a model chose to reply. The answer is saved as
/// `<dir>/scripted.json`.
pub fn scripted_call(template: &str, dir: &Path) -> Result<bool, GateError> {
    let addr = scripted_server(template)?;
    let url = |p: &str| format!("http://{addr}{p}");
    let answer = chatted(&url, dir, "scripted", &tool_chat(SCRIPT_TOKENS))?;
    let choice = &answer["choices"][0];
    let calls = choice["message"]["tool_calls"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let function = calls.first().map(|c| &c["function"]);
    let name = function.and_then(|f| f["name"].as_str());
    let arguments: Option<Value> = function
        .and_then(|f| f["arguments"].as_str())
        .and_then(|a| serde_json::from_str(a).ok());
    println!(
        "scripted call: finish {:?}, {} call(s), function {name:?}, arguments {arguments:?}; content {:?}",
        choice["finish_reason"],
        calls.len(),
        choice["message"]["content"]
    );
    Ok(choice["finish_reason"] == "tool_calls"
        && calls.len() == 1
        && name == Some("get_weather")
        && arguments == Some(json!({ "city": "Seoul" })))
}

/// A server over a scripted engine whose every reply is [`SCRIPTED_CALL`], with the chat template
/// `template`: its address.
fn scripted_server(template: &str) -> Result<std::net::SocketAddr, GateError> {
    let config = serve::ServerConfig {
        model_alias: "scripted".to_owned(),
        model_path: "scripted.gguf".to_owned(),
        chat_template: template.to_owned(),
        sampler: None,
        fatal_linger: serve::FATAL_LINGER,
        slot_save_path: None,
        api_keys: serve::flag::ApiKeys::default(),
        chat_preparation: None,
    };
    let engine = serve::ScriptedEngine::new(SCRIPT_CTX, SCRIPTED_CALL);
    Ok(serve::Server::bind("127.0.0.1:0", Box::new(engine), config)
        .map_err(|e| format!("the scripted server: {e}"))?
        .spawn()?)
}

/// What a chat answer holds of a cut generation, whole or folded from its stream.
struct Cut {
    /// The error the answer was, empty when it was one.
    error: String,
    finish: String,
    content: String,
    /// Each call's name and arguments, by index.
    calls: Vec<(String, String)>,
}

/// `body` of a chat request as [`Cut`] reads it: a non-stream answer's choice, or the deltas of a stream
/// folded; a status but 200, an error event or text that is not the answer's shape is the `error`.
fn cut_of(status: u16, body: &str, stream: bool) -> Cut {
    let mut cut = Cut {
        error: String::new(),
        finish: String::new(),
        content: String::new(),
        calls: Vec::new(),
    };
    if status != 200 {
        cut.error = format!("HTTP {status}: {body}");
        return cut;
    }
    let events: Vec<Value> = if stream {
        body.split("\n\n")
            .filter_map(|e| e.strip_prefix("data: "))
            .filter(|e| *e != "[DONE]")
            .filter_map(|e| serde_json::from_str(e).ok())
            .collect()
    } else {
        serde_json::from_str(body).into_iter().collect()
    };
    for v in &events {
        if let Some(e) = v["error"]["message"].as_str() {
            cut.error = e.to_owned();
        }
        let Some(choice) = v["choices"].get(0) else {
            continue;
        };
        if let Some(f) = choice["finish_reason"].as_str() {
            cut.finish = f.to_owned();
        }
        let part = if stream {
            &choice["delta"]
        } else {
            &choice["message"]
        };
        cut.content.push_str(part["content"].as_str().unwrap_or(""));
        for call in part["tool_calls"].as_array().into_iter().flatten() {
            let at = call["index"]
                .as_u64()
                .and_then(|i| usize::try_from(i).ok())
                .unwrap_or(cut.calls.len());
            if cut.calls.len() <= at {
                cut.calls.resize(at + 1, (String::new(), String::new()));
            }
            if let Some(n) = call["function"]["name"].as_str() {
                cut.calls[at].0.push_str(n);
            }
            if let Some(a) = call["function"]["arguments"].as_str() {
                cut.calls[at].1.push_str(a);
            }
        }
    }
    if events.is_empty() && cut.error.is_empty() {
        cut.error = format!("no answer in {body:?}");
    }
    cut
}

/// The server of [`scripted_call`] asked [`tool_turn`] with a `max_tokens` that ends the generation inside the call
/// (the scripted vocabulary is a byte a token), whole and as a stream. llama-server parses the end of the text as the
/// end of the input, so a cut call is no fault there: it answers 200 with `finish_reason` `length`, the text read as
/// `content` while the call's name is not whole, and the call with the arguments read (`{`) once it is. The same
/// here, each answer saved as `<dir>/cut-<where>-<whole|stream>.txt`.
pub fn cut_call(template: &str, dir: &Path) -> Result<bool, GateError> {
    let addr = scripted_server(template)?;
    let url = |p: &str| format!("http://{addr}{p}");
    let inside = |tag: &str| {
        SCRIPTED_CALL
            .find(tag)
            .map(|at| at + tag.len() / 2)
            .ok_or_else(|| format!("the scripted call holds no {tag}"))
    };
    let named = ("get_weather".to_owned(), "{".to_owned());
    let cuts = [
        ("before-name", inside("<function=")?, None),
        ("after-name", inside("<parameter=")?, Some(named)),
    ];
    let mut ok = true;
    for (place, bytes, call) in cuts {
        for stream in [false, true] {
            let mut body = tool_chat(u64::try_from(bytes)?);
            body["stream"] = json!(stream);
            let (status, text) = curl(&url("/v1/chat/completions"), Some(&body), stream)?;
            let way = if stream { "stream" } else { "whole" };
            std::fs::write(dir.join(format!("cut-{place}-{way}.txt")), &text)?;
            let cut = cut_of(status, &text, stream);
            let (calls, content) = match &call {
                Some(c) => (vec![c.clone()], ""),
                None => (Vec::new(), &SCRIPTED_CALL[..bytes]),
            };
            let good = cut.error.is_empty()
                && cut.finish == "length"
                && cut.calls == calls
                && cut.content == content;
            println!(
                "cut call {place} ({bytes} tokens, {way}): error {:?}, finish {:?}, calls {:?}, content {:?}; wants \
                 calls {calls:?}, content {content:?} {good}",
                cut.error, cut.finish, cut.calls, cut.content
            );
            ok &= good;
        }
    }
    Ok(ok)
}

/// `POST /v1/chat/completions/input_tokens` of `body`: how many ids the prompt the same chat request renders holds,
/// counted without generating, as llama-server's `handle_count_tokens` does.
pub fn input_tokens(url: &dyn Fn(&str) -> String, body: &Value) -> Result<usize, GateError> {
    const ROUTE: &str = "/v1/chat/completions/input_tokens";
    let (st, text) = curl(&url(ROUTE), Some(body), false)?;
    let n = json_of(ROUTE, st, &text)?["input_tokens"]
        .as_u64()
        .ok_or_else(|| format!("{ROUTE}: no input_tokens in {text}"))?;
    Ok(usize::try_from(n)?)
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

/// Greedy ids on the same loaded server without a speculative pass, and the
/// reference requests' total draft count.
pub fn stepped_completion(
    url: &dyn Fn(&str) -> String,
    ids: &[u32],
    n: usize,
) -> Result<(Vec<u32>, u64), GateError> {
    let mut prefix = ids.to_vec();
    let mut tokens = Vec::new();
    let mut proposed = 0u64;
    for i in 0..n {
        // The first token ends each request before a speculative pass can
        // start. Every later prompt feeds only the preceding answer's id.
        let s = completion(url, &prefix, 1, i > 0)?;
        if i > 0 {
            let kept = u64::try_from(prefix.len() - 1)?;
            if s["timings"]["cache_n"].as_u64() != Some(kept) {
                return Err(format!(
                    "step reference: expected {kept} cached positions before token {i}: {s}"
                )
                .into());
            }
        }
        proposed = proposed
            .checked_add(draft_counts(&s)?.0)
            .ok_or("step reference: draft count overflow")?;
        let ids = ids_of(&s["tokens"]).map_err(|e| format!("step reference: {e}"))?;
        match ids.as_slice() {
            [id] => {
                prefix.push(*id);
                tokens.push(*id);
            }
            [] if s["stop_type"] == "eos" => break,
            _ => return Err(format!("step reference: expected one token: {s}").into()),
        }
        match s["stop_type"].as_str() {
            Some("limit") => {}
            Some("eos") => break,
            _ => return Err(format!("step reference: unexpected stop_type: {s}").into()),
        }
    }
    Ok((tokens, proposed))
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
    let tokens = ids_of(&v["tokens"])?;
    let stop = v["stop_type"].as_str().unwrap_or("").to_owned();
    let cache_n = v["timings"]["cache_n"]
        .as_u64()
        .ok_or("/completion: missing or invalid timings.cache_n")?;
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

/// How long the request `v` answered held its slot, in milliseconds: the
/// server's `timings.prompt_ms` plus `predicted_ms`, the window a request
/// posted beside it has to arrive in to take the other slot. A timings object
/// that lacks either is refused by name.
pub fn held_ms(name: &str, v: &Value) -> Result<f64, GateError> {
    ["prompt_ms", "predicted_ms"]
        .iter()
        .map(|k| {
            v["timings"][k]
                .as_f64()
                .ok_or_else(|| format!("{name}: the answer's timings carry no {k}"))
        })
        .sum::<Result<f64, String>>()
        .map_err(Into::into)
}

/// Two requests in flight on `addr`, both posted at once: the engine thread
/// admits the second while the first holds its slot, so it takes the other
/// slot whatever the body's speed (a gap between the posts outlives a fixture
/// body's whole run, and the second then takes the first's freed slot).
/// `request` runs each, given the server's URL maker, the request's name and
/// its prompt ids; a panic or an error of either is the call's error.
pub fn together<R: Send>(
    addr: &str,
    first: (&str, &[u32]),
    second: (&str, &[u32]),
    request: impl Fn(&dyn Fn(&str) -> String, &str, &[u32]) -> Result<R, GateError> + Sync,
) -> Result<[R; 2], GateError> {
    std::thread::scope(|sc| {
        let a = sc.spawn(|| post(addr, first, &request));
        let b = sc.spawn(|| post(addr, second, &request));
        let a = a.join().map_err(|_| "a round's first request panicked")??;
        let b = b
            .join()
            .map_err(|_| "a round's second request panicked")??;
        Ok([a, b])
    })
}

/// One request of [`together`], its error as text (a [`GateError`] is not
/// `Send`).
fn post<R>(
    addr: &str,
    (name, prompt): (&str, &[u32]),
    request: &impl Fn(&dyn Fn(&str) -> String, &str, &[u32]) -> Result<R, GateError>,
) -> Result<R, String> {
    let url = |p: &str| format!("http://{addr}{p}");
    request(&url, name, prompt).map_err(|e| e.to_string())
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

/// `body` (a chat request's messages and template switches) as a greedy chat
/// of `n` tokens with each step's two best tokens returned, no prompt cache.
pub fn greedy_chat(body: &Value, n: usize) -> Value {
    let mut b = body.clone();
    b["max_tokens"] = json!(n);
    b["temperature"] = json!(0);
    b["cache_prompt"] = json!(false);
    b["logprobs"] = json!(true);
    b["top_logprobs"] = json!(2);
    b
}

/// The ids a greedy chat ([`greedy_chat`]) answered, each with the runner-up
/// of its step and the gap between the top two logprobs (`None` where the
/// answer carried one only), the prompt's token count and the text the ids
/// say (the reasoning, then the content).
pub struct ChatReply {
    pub ids: Vec<u32>,
    pub second: Vec<Option<(u32, f64)>>,
    pub prompt: u64,
    pub text: String,
}

/// The reply in `status`/`text` of a chat of [`greedy_chat`], or a named error.
pub fn answered(status: u16, text: &str) -> Result<ChatReply, GateError> {
    let v = json_of("/v1/chat/completions", status, text)?;
    let content = v["choices"][0]["logprobs"]["content"]
        .as_array()
        .ok_or("the chat answer carries no logprobs content")?;
    let (mut ids, mut second) = (Vec::new(), Vec::new());
    for e in content {
        ids.push(u32::try_from(
            e["id"].as_u64().ok_or("a logprobs entry with no id")?,
        )?);
        second.push(e["top_logprobs"].as_array().and_then(|t| {
            let (a, b) = (t.first()?, t.get(1)?);
            Some((
                u32::try_from(b["id"].as_u64()?).ok()?,
                a["logprob"].as_f64()? - b["logprob"].as_f64()?,
            ))
        }));
    }
    let msg = &v["choices"][0]["message"];
    Ok(ChatReply {
        ids,
        second,
        prompt: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
        text: format!(
            "{}{}",
            msg["reasoning_content"].as_str().unwrap_or(""),
            msg["content"].as_str().unwrap_or("")
        ),
    })
}

impl ChatReply {
    /// An oracle's rule for one prompt, over the prefix `want[..agree]` where
    /// the oracle's own CUDA and CPU runs agree (past it the oracle is two
    /// engines' roundings parting and holds nothing): the server's greedy ids
    /// equal `want`, or at the first difference the oracle's id is the
    /// server's own runner-up within `near_tie` of its top (a near tie, where
    /// two engines' roundings may part) and nothing after it is held. The
    /// step the ids agreed up to, the gap at the first difference when there
    /// is one, and whether the rule holds; a reply shorter than the prefix
    /// fails.
    pub fn follows(&self, want: &[u32], agree: usize, near_tie: f64) -> (usize, Option<f64>, bool) {
        if self.ids.len() < agree {
            return (self.ids.len(), None, false);
        }
        for (k, &wanted) in want.iter().enumerate().take(agree) {
            if self.ids[k] == wanted {
                continue;
            }
            let gap = self.second[k]
                .filter(|&(id, _)| id == wanted)
                .map(|(_, g)| g);
            return (k, gap, gap.is_some_and(|g| g < near_tie));
        }
        (agree, None, true)
    }
}

/// Two questions share a system message in the live slot and answer the ids of a fresh run.
pub fn shared_system_prompt(
    url: &dyn Fn(&str) -> String,
    dir: &Path,
    system: &str,
    users: [&str; 2],
    n: usize,
) -> Result<bool, GateError> {
    let mut ok = true;
    let mut prompts = Vec::new();
    for user in users {
        prompts.push(rendered(
            url,
            json!([
                { "role": "system", "content": system },
                { "role": "user", "content": user },
            ]),
        )?);
    }
    let marker = tokenized(url, "<|im_start|>")?;
    if marker.is_empty() {
        return Err("the message-start marker encodes no ids".into());
    }
    let start = prompts[0]
        .windows(marker.len())
        .enumerate()
        .filter_map(|(i, ids)| (ids == marker).then_some(i))
        .nth(1)
        .ok_or("the shared-system chat has no second message start")?;
    let (first, _) = greedy(url, prompts[0].clone(), n, dir, "system_first", false)?;
    let (reused, _) = greedy(url, prompts[1].clone(), n, dir, "system_reused", true)?;
    let (fresh, _) = greedy(url, prompts[1].clone(), n, dir, "system_fresh", false)?;
    println!(
        "shared system: message_start={start} kept={} reused={:?} fresh={:?}",
        reused.cache_n, reused.tokens, fresh.tokens
    );
    check(
        &mut ok,
        "shared_system_prompt_stays_in_the_slot",
        start > 0 && first.cache_n == 0 && reused.cache_n >= start as u64,
    );
    check(
        &mut ok,
        "shared_system_prompt_ids_are_a_fresh_runs",
        fresh.cache_n == 0 && !reused.tokens.is_empty() && reused.tokens == fresh.tokens,
    );
    Ok(ok)
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
pub fn repeated_ids(
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

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::{
        LookupCompletion, completion, curl, draft_counts, held_ms, ids_of, slot_erased,
        stepped_completion, together,
    };
    use serde_json::json;

    /// The step reference never runs a pass, even on a server that drafts
    /// requests asking for probabilities. Mutant: one multi-token request
    /// with `n_probs` as the reference.
    #[test]
    fn stepped_completion_never_drafts_on_a_drafting_load() {
        let config = serve::ServerConfig {
            model_alias: "mock".into(),
            model_path: "mock.gguf".into(),
            chat_template: include_str!("../../serve/tests/fixtures/v41-chat-template.jinja")
                .into(),
            chat_preparation: None,
            sampler: None,
            fatal_linger: serve::FATAL_LINGER,
            slot_save_path: None,
            api_keys: serve::flag::ApiKeys::default(),
        };
        let server = serve::Server::bind(
            "127.0.0.1:0",
            Box::new(serve::DraftMock::new(4096).with_sampled_draft(3, 2, 3)),
            config,
        )
        .unwrap();
        let addr = server.spawn().unwrap();
        let url = |p: &str| format!("http://{addr}{p}");
        let ids = [97, 98, 97];
        let drafted = completion(&url, &ids, 16, false).unwrap();
        assert!(draft_counts(&drafted).unwrap().0 > 0, "{drafted}");
        let (steps, count) = stepped_completion(&url, &ids, 16).unwrap();
        assert_eq!(count, 0, "the reference drafted {count} ids");
        assert_eq!(steps, ids_of(&drafted["tokens"]).unwrap());
        assert_eq!(stepped_completion(&url, &ids, 0).unwrap(), (vec![], 0));
        assert_eq!(stepped_completion(&url, &ids, 1).unwrap().0.len(), 1);
        assert_eq!(
            curl(&url("/shutdown"), Some(&json!({})), false).unwrap().0,
            200
        );
    }

    /// The two requests of a round start together: the second is posted while the first still
    /// holds its slot, however short the body's whole run is, so the second starts within
    /// `WITHIN` of the first, a fraction of a gap a fixture body outlives. Mutant: a gap between
    /// the posts.
    #[test]
    fn together_posts_both_requests_at_once() {
        const WITHIN: Duration = Duration::from_millis(150);
        let t0 = Instant::now();
        let ran = together(
            "127.0.0.1:0",
            ("first", &[1]),
            ("second", &[2]),
            |url, name, prompt| {
                let start = t0.elapsed();
                assert_eq!(url("/completion"), "http://127.0.0.1:0/completion");
                Ok((name.to_owned(), prompt.to_vec(), start))
            },
        )
        .expect("both requests ran");
        assert_eq!(
            ran.each_ref().map(|r| (r.0.as_str(), r.1.as_slice())),
            [("first", &[1][..]), ("second", &[2][..])]
        );
        let apart = ran[1].2.abs_diff(ran[0].2);
        assert!(apart < WITHIN, "the posts were {apart:?} apart");
    }

    /// An error of either request is the call's error, in the request's own words.
    #[test]
    fn together_returns_the_request_error() {
        let ran = together("a:0", ("one", &[1]), ("two", &[2]), |_, name, _| {
            if name == "two" {
                Err(format!("{name} refused").into())
            } else {
                Ok(name.len())
            }
        });
        assert_eq!(
            ran.expect_err("the second failed").to_string(),
            "two refused"
        );
    }

    /// A request's held time is its prompt and predicted milliseconds; either missing is refused
    /// by name.
    #[test]
    fn held_ms_sums_the_two_timings_and_refuses_a_missing_one() {
        let timings = json!({ "timings": {"prompt_ms": 12.5, "predicted_ms": 30.0} });
        let held = held_ms("r", &timings).expect("both present");
        assert!((held - 42.5).abs() < 1e-9, "{held}");
        for gone in ["prompt_ms", "predicted_ms"] {
            let mut bad = timings.clone();
            bad["timings"].as_object_mut().unwrap().remove(gone);
            let e = held_ms("r", &bad).expect_err("one missing").to_string();
            assert_eq!(e, format!("r: the answer's timings carry no {gone}"));
        }
        assert!(held_ms("r", &json!({})).is_err());
    }

    /// A completion's draft counts: both or neither, unsigned, at least one
    /// proposal and no more accepted than proposed.
    #[test]
    fn draft_counts_need_both_counts_of_a_proposal() {
        let counts = |t| draft_counts(&json!({ "timings": t })).ok();
        assert!(
            draft_counts(&json!({})).is_err(),
            "no timings object was taken"
        );
        assert_eq!(counts(json!({})), Some((0, 0)));
        assert_eq!(
            counts(json!({"draft_n": 10, "draft_n_accepted": 0})),
            Some((10, 0))
        );
        assert_eq!(
            counts(json!({"draft_n": 10, "draft_n_accepted": 4})),
            Some((10, 4))
        );
        for bad in [
            json!({"draft_n": null}),
            json!({"draft_n_accepted": null}),
            json!({"draft_n": null, "draft_n_accepted": null}),
            json!({"draft_n": 10}),
            json!({"draft_n": 10, "draft_n_accepted": null}),
            json!({"draft_n": 10, "draft_n_accepted": "4"}),
            json!({"draft_n": 10, "draft_n_accepted": -1}),
            json!({"draft_n": 10, "draft_n_accepted": 11}),
            json!({"draft_n_accepted": 0}),
            json!({"draft_n": 0, "draft_n_accepted": 0}),
            json!({"draft_n": 1.5, "draft_n_accepted": 1}),
            json!({"draft_n": [], "draft_n_accepted": 0}),
            json!({"draft_n": 1, "draft_n_accepted": {}}),
            json!({"draft_n": true, "draft_n_accepted": 0}),
        ] {
            assert_eq!(counts(bad.clone()), None, "{bad} was taken");
            let mut answer = json!({
                "tokens": [1], "stop_type": "limit", "id_slot": 0, "timings": bad
            });
            answer["timings"]["cache_n"] = json!(0);
            assert!(
                LookupCompletion::read(&answer, 2, 4096).is_err(),
                "{answer}"
            );
        }
    }

    /// A response's ids are every entry of an array of `u32` ids; a value that is no
    /// array and an entry that is no such id are refused by name, never left out.
    #[test]
    fn ids_of_refuses_what_is_no_id() {
        assert_eq!(ids_of(&json!([7, 0, u32::MAX])).unwrap(), [7, 0, u32::MAX]);
        assert_eq!(ids_of(&json!([])).unwrap(), Vec::<u32>::new());
        for (bad, entry) in [
            (json!([7, null, 8]), "entry 1 (null)"),
            (json!([7, u64::from(u32::MAX) + 1]), "entry 1 (4294967296)"),
            (json!([-1]), "entry 0 (-1)"),
            (json!([1.5]), "entry 0 (1.5)"),
            (json!(["7"]), "entry 0 (\"7\")"),
        ] {
            let e = ids_of(&bad)
                .expect_err("an entry that is no id")
                .to_string();
            assert!(e.contains(entry), "{bad}: {e}");
        }
        for bad in [json!(null), json!({}), json!("7,8")] {
            let e = ids_of(&bad).expect_err("no array").to_string();
            assert!(e.ends_with("is no array"), "{bad}: {e}");
        }
    }

    #[test]
    fn lookup_completion_requires_evidence() {
        let valid = json!({
            "tokens": [1, 2], "stop_type": "limit", "id_slot": 0,
            "timings": {"cache_n": 4096, "draft_n": 2, "draft_n_accepted": 1}
        });
        assert!(LookupCompletion::read(&valid, 2, 4096).is_ok());
        for field in ["tokens", "stop_type", "id_slot", "timings"] {
            let mut bad = valid.clone();
            bad.as_object_mut().unwrap().remove(field);
            assert!(
                LookupCompletion::read(&bad, 2, 4096).is_err(),
                "missing {field}"
            );
            for value in [json!(null), json!(false), json!({}), json!("0")] {
                bad[field] = value;
                assert!(LookupCompletion::read(&bad, 2, 4096).is_err(), "{bad}");
            }
        }
        for (field, value) in [
            ("tokens", json!([])),
            ("tokens", json!([1, null, 2])),
            ("tokens", json!([1, "2"])),
            ("tokens", json!([1, -1])),
            ("tokens", json!([1, 1.5])),
            ("tokens", json!([1, 4294967296_u64])),
            ("stop_type", json!(1)),
            ("stop_type", json!("")),
            ("stop_type", json!("unknown")),
            ("id_slot", json!(2)),
            ("id_slot", json!(-1)),
            ("id_slot", json!(0.5)),
        ] {
            let mut bad = valid.clone();
            bad[field] = value;
            assert!(LookupCompletion::read(&bad, 2, 4096).is_err(), "{bad}");
        }
        let mut missing = valid.clone();
        missing["timings"]
            .as_object_mut()
            .unwrap()
            .remove("cache_n");
        assert!(LookupCompletion::read(&missing, 2, 4096).is_err());
        for value in [json!(null), json!("0"), json!(-1), json!(0.5), json!(4097)] {
            let mut bad = valid.clone();
            bad["timings"]["cache_n"] = value;
            assert!(LookupCompletion::read(&bad, 2, 4096).is_err(), "{bad}");
        }
        for stop in ["eos", "word", "limit"] {
            let mut valid = valid.clone();
            valid["stop_type"] = json!(stop);
            valid["id_slot"] = json!(1);
            valid["timings"] = json!({"cache_n": 0});
            assert!(LookupCompletion::read(&valid, 2, 4096).is_ok());
        }
    }

    #[test]
    fn slot_erased_requires_evidence() {
        for n in [0, 4096] {
            assert!(slot_erased(&json!({"id_slot": 1, "n_erased": n}), 1, 4096).is_ok());
        }
        for field in ["id_slot", "n_erased"] {
            let mut bad = json!({"id_slot": 1, "n_erased": 0});
            bad.as_object_mut().unwrap().remove(field);
            assert!(slot_erased(&bad, 1, 4096).is_err(), "missing {field}");
            for value in [json!(null), json!("1"), json!(-1), json!(0.5), json!(4097)] {
                bad[field] = value;
                assert!(slot_erased(&bad, 1, 4096).is_err(), "{bad}");
            }
        }
        assert!(slot_erased(&json!({"id_slot": 0, "n_erased": 0}), 1, 4096).is_err());
    }
}
