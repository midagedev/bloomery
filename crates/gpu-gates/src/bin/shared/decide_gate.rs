//! What the decide seat's card gates share (`gate_clef_serve`, `gate_lev_serve`): the server of
//! `bloomery-serve` beside the gate binary run one process at a time (`spawn`), a refusal read off its
//! stderr (`refused`), its `listening` record (`listening`), a request posted as written (`post_raw`),
//! and a body's tops and probabilities compared with another's (`compare`). A clause's own questions
//! stay in its gate.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use bloomery_gpu_gates::serve_client::Served;
use bloomery_gpu_gates::{GateError, verdict};
use serde_json::Value;

/// How long a refused server may take to exit: it refuses before any
/// load of the backbone.
pub const REFUSE_WITHIN: Duration = Duration::from_secs(120);

pub fn check(ok: &mut bool, name: &str, pass: bool) {
    println!("check {name}: {}", verdict(pass));
    *ok &= pass;
}

pub fn utf8(p: &Path) -> Result<&str, GateError> {
    Ok(p.to_str()
        .ok_or_else(|| format!("{} is not UTF-8", p.display()))?)
}

/// `bloomery-serve` beside this binary with `args`, `BLOOMERY_REF_MODEL`
/// removed, its logs in `<dir>/<name>/`.
pub fn spawn(dir: &Path, name: &str, args: &[&str]) -> Result<(Served, PathBuf), GateError> {
    let d = dir.join(name);
    std::fs::create_dir_all(&d)?;
    let mut cmd = Command::new(std::env::current_exe()?.with_file_name("bloomery-serve"));
    cmd.env_remove("BLOOMERY_REF_MODEL");
    Ok((Served::spawn_cmd(cmd, args, &d)?, d))
}

/// Whether the server of `args` exits non-zero within [`REFUSE_WITHIN`]
/// with every one of `want` in its stderr; a server still running then
/// is killed and fails the clause.
pub fn refused(dir: &Path, name: &str, args: &[&str], want: &[&str]) -> Result<bool, GateError> {
    let (mut s, d) = spawn(dir, name, args)?;
    let t = Instant::now();
    let status = loop {
        if let Some(status) = s.child.try_wait()? {
            break Some(status);
        }
        if t.elapsed() > REFUSE_WITHIN {
            break None;
        }
        std::thread::sleep(Duration::from_millis(200));
    };
    let err = std::fs::read_to_string(d.join("server.err"))?;
    let pass = match status {
        Some(st) => {
            let named = want.iter().all(|w| err.contains(w));
            println!("{name}: exit {st}, names {want:?}: {named}");
            !st.success() && named
        }
        None => {
            println!(
                "{name}: still running after {REFUSE_WITHIN:?}: {}",
                s.stop()?
            );
            false
        }
    };
    if !pass {
        println!("{name} stderr:\n{err}");
    }
    Ok(pass)
}

/// How long the server may take to load and listen, and how often its
/// stderr is read meanwhile.
pub const LISTEN_WITHIN: Duration = Duration::from_secs(300);
pub const LISTEN_POLL: Duration = Duration::from_millis(500);

/// The seat's `listening` record in the server's stderr, read every
/// [`LISTEN_POLL`] for at most [`LISTEN_WITHIN`]; a server that exits
/// first fails by name.
pub fn listening(s: &mut Served, err_log: &Path) -> Result<String, GateError> {
    let t = Instant::now();
    while t.elapsed() < LISTEN_WITHIN {
        let text = std::fs::read_to_string(err_log).unwrap_or_default();
        if let Some(l) = text.lines().find(|l| l.contains("listening on http://")) {
            return Ok(l.to_owned());
        }
        if let Some(st) = s.child.try_wait()? {
            return Err(format!("the server exited ({st}) before listening").into());
        }
        std::thread::sleep(LISTEN_POLL);
    }
    Err(format!("the server did not listen within {LISTEN_WITHIN:?}").into())
}

/// `text` POSTed to `url` as it is (curl's stdin, no re-serialization:
/// the request's key order is its questions' order): the status and the
/// body.
pub fn post_raw(url: &str, text: &str) -> Result<(u16, String), GateError> {
    let mut c = Command::new("curl")
        .args(["-sS", "--max-time", "600", "-w", "\n%{http_code}"])
        .args([
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            "@-",
            url,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    c.stdin
        .take()
        .ok_or("curl has no stdin")?
        .write_all(text.as_bytes())?;
    let out = c.wait_with_output()?;
    let all = String::from_utf8(out.stdout)?;
    let (body, code) = all
        .rsplit_once('\n')
        .ok_or_else(|| format!("curl {url}: no status line"))?;
    Ok((code.trim().parse()?, body.to_owned()))
}

/// `body` with the `timings` object the server appends as its last key
/// taken out, or `None` when it has none.
pub fn untimed(body: &str) -> Option<&str> {
    let at = body.rfind(",\"timings\":")?;
    body.trim_end().ends_with("}}").then_some(&body[..at])
}

/// The option an answer picks: a noul's side, a choice's `choice`, a score's most probable
/// level.
pub fn top(a: &Value) -> Option<String> {
    match a["type"].as_str()? {
        "noul" => Some((a["noul"].as_f64()? >= 0.5).to_string()),
        "choice" => a["choice"].as_str().map(str::to_owned),
        "score" => {
            let mut best: Option<(&String, f64)> = None;
            for (k, v) in a["probabilities"].as_object()? {
                let p = v.as_f64()?;
                if best.is_none_or(|(_, b)| p > b) {
                    best = Some((k, p));
                }
            }
            best.map(|(k, _)| k.clone())
        }
        _ => None,
    }
}

/// Every probability an answer gives, by name: a noul's `noul`, else its `probabilities`.
pub fn probabilities(a: &Value) -> Vec<(String, f64)> {
    match a["probabilities"].as_object() {
        Some(p) => p
            .iter()
            .filter_map(|(k, v)| v.as_f64().map(|v| (k.clone(), v)))
            .collect(),
        None => a["noul"]
            .as_f64()
            .map(|v| vec![("noul".to_owned(), v)])
            .unwrap_or_default(),
    }
}

/// `a`'s answers against `b`'s on every one of `questions`, printed a question a line under
/// `what`: whether every question picks the same option, and the largest |dp| over every
/// probability. A question either lacks, or whose probabilities differ in names, is a
/// difference of 1 and no match.
pub fn compare(what: &str, a: &Value, b: &Value, questions: &[String]) -> (bool, f64) {
    let (mut tops, mut worst) = (true, 0f64);
    for q in questions {
        let (x, y) = (&a["answers"][q], &b["answers"][q]);
        let (tx, ty) = (top(x), top(y));
        let (px, py) = (probabilities(x), probabilities(y));
        let names = px.iter().map(|p| &p.0).eq(py.iter().map(|p| &p.0));
        let dp = if names && !px.is_empty() {
            px.iter()
                .zip(&py)
                .map(|(x, y)| (x.1 - y.1).abs())
                .fold(0.0, f64::max)
        } else {
            1.0
        };
        let same = tx.is_some() && tx == ty;
        println!(
            "{what}: {q}: top {tx:?} / {ty:?} {} max |dp| {dp:.4}",
            if same { "equal" } else { "DIFFERENT" }
        );
        tops &= same;
        worst = worst.max(dp);
    }
    (tops, worst)
}
