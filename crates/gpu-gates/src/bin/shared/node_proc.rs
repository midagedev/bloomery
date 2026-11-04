//! A gate's child process — a `tier_server` it serves a tier from, or the gate binary itself
//! under another environment — whose output the gate prints under a tag and reads back: stdout
//! line by line, stderr kept whole. The child dies with the gate
//! (`bloomery_gpu_gates::die_with_parent`), and dropping its handle kills and reaps a child
//! still running.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::SocketAddr;
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bloomery_gpu_gates::GateError;
use bloomery_levers::{CARD_BUDGET, CARD_DONTNEED, HOST_LOCK, HOST_POPULATE, TIER_NODES};

/// What a `tier_server` prints once bound, before its address.
pub const LISTENING: &str = "listening on ";

/// How long a `tier_server` may take to bind and print its address.
pub const BIND_WITHIN: Duration = Duration::from_secs(120);

/// A child process whose output this process prints, each line under
/// `tag`, and passes on: stdout's lines through [`Proc::wait_line`], stderr
/// kept whole.
pub struct Proc {
    child: Child,
    lines: Receiver<String>,
    err: Arc<Mutex<String>>,
    stderr_reader: Option<std::thread::JoinHandle<()>>,
    tag: &'static str,
}

impl Proc {
    /// `cmd` started with its three streams piped, killed when this process
    /// dies.
    pub fn spawn(mut cmd: Command, tag: &'static str) -> Result<Proc, GateError> {
        bloomery_gpu_gates::die_with_parent(&mut cmd);
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("{tag}: spawn: {e}"))?;
        let (tx, lines) = mpsc::channel();
        let out = child.stdout.take().ok_or("a child with no stdout")?;
        std::thread::spawn(move || {
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                println!("[{tag}] {line}");
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let err = Arc::new(Mutex::new(String::new()));
        let mut stderr = child.stderr.take().ok_or("a child with no stderr")?;
        let kept = Arc::clone(&err);
        let stderr_reader = std::thread::spawn(move || {
            let mut b = [0u8; 4096];
            while let Ok(n) = stderr.read(&mut b) {
                if n == 0 {
                    break;
                }
                let text = String::from_utf8_lossy(&b[..n]).into_owned();
                for l in text.lines() {
                    eprintln!("[{tag}] {l}");
                }
                kept.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push_str(&text);
            }
        });
        Ok(Proc {
            child,
            lines,
            err,
            stderr_reader: Some(stderr_reader),
            tag,
        })
    }

    /// The first stdout line starting with `prefix`, waited for within
    /// `within`.
    pub fn wait_line(&self, prefix: &str, within: Duration) -> Result<String, GateError> {
        let end = Instant::now() + within;
        loop {
            let left = end.saturating_duration_since(Instant::now());
            match self.lines.recv_timeout(left) {
                Ok(l) if l.starts_with(prefix) => return Ok(l),
                Ok(_) => {}
                Err(e) => {
                    return Err(format!("{}: no line {prefix:?} ({e})", self.tag).into());
                }
            }
        }
    }

    /// Every stdout line the process printed and [`Proc::wait_line`] has not
    /// taken, in order, without waiting: once the process has exited, the
    /// rest of its output.
    pub fn take_lines(&self) -> Vec<String> {
        self.lines.try_iter().collect()
    }

    /// The process's exit, waited for within `within`, with its stderr
    /// read to the end; `None` past it.
    pub fn wait_within(&mut self, within: Duration) -> Result<Option<ExitStatus>, GateError> {
        let end = Instant::now() + within;
        loop {
            if let Some(s) = self.child.try_wait()? {
                if let Some(reader) = self.stderr_reader.take() {
                    reader
                        .join()
                        .map_err(|_| format!("{}: stderr reader panicked", self.tag))?;
                }
                return Ok(Some(s));
            }
            if Instant::now() > end {
                return Ok(None);
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// Send `sig` (`kill`'s spelling, `-STOP`, `-KILL`, …) to the process.
    pub fn signal(&self, sig: &str) -> Result<(), GateError> {
        let pid = self.child.id().to_string();
        let s = Command::new("kill").args([sig, &pid]).status()?;
        if !s.success() {
            return Err(format!("kill {sig} {pid}: {s}").into());
        }
        Ok(())
    }

    /// The process's id, which [`Proc::signal`] sends to.
    pub fn id(&self) -> u32 {
        self.child.id()
    }

    /// Kill the process (`SIGKILL`) without reaping it: [`Proc::wait_within`]
    /// reaps it.
    pub fn kill(&mut self) -> Result<(), GateError> {
        self.child.kill()?;
        Ok(())
    }

    /// The next stdout line [`Proc::wait_line`] has not taken, waited for
    /// within `within`: `Disconnected` once the process's stdout has closed
    /// and every line was taken.
    pub fn next_line(&self, within: Duration) -> Result<String, RecvTimeoutError> {
        self.lines.recv_timeout(within)
    }

    /// Write `line` to the process's stdin.
    pub fn say(&mut self, line: &str) -> Result<(), GateError> {
        let stdin = self.child.stdin.as_mut().ok_or("a child with no stdin")?;
        stdin.write_all(line.as_bytes())?;
        stdin.flush()?;
        Ok(())
    }

    /// What the process wrote to stderr so far.
    pub fn stderr(&self) -> String {
        self.err
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl Drop for Proc {
    fn drop(&mut self) {
        if let Ok(None) = self.child.try_wait() {
            drop(self.child.kill());
            drop(self.child.wait());
        }
    }
}

/// The `tier_server` beside this binary on a free loopback port, serving
/// `model` when one is given (else its `$BLOOMERY_REF_MODEL`): the gate's load
/// levers that size a stage (`BLOOMERY_CARD_BUDGET`, the host set's populate and
/// lock, the card's dontneed) and the node list are taken out of its environment.
pub fn tier_server_command(model: Option<&Path>) -> Result<Command, GateError> {
    let mut cmd = Command::new(bloomery_gpu_gates::beside("tier_server")?);
    cmd.args(["--listen", "127.0.0.1:0"])
        .env_remove(TIER_NODES)
        .env_remove(CARD_BUDGET)
        .env_remove(HOST_POPULATE)
        .env_remove(HOST_LOCK)
        .env_remove(CARD_DONTNEED);
    if let Some(m) = model {
        cmd.arg("-m").arg(m);
    }
    Ok(cmd)
}

/// `cmd`, a [`tier_server_command`] and what its caller added, started under
/// `tag`, and the address it printed once bound ([`LISTENING`], within
/// [`BIND_WITHIN`]).
pub fn tier_server(cmd: Command, tag: &'static str) -> Result<(Proc, SocketAddr), GateError> {
    let p = Proc::spawn(cmd, tag)?;
    let line = p.wait_line(LISTENING, BIND_WITHIN)?;
    let addr = line[LISTENING.len()..].parse()?;
    Ok((p, addr))
}
