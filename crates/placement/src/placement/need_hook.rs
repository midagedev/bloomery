//! The need hook: `BLOOMERY_NEED_HOOK`, an executable a placed load calls once,
//! after the plan and before the card is opened or the host set read in, with
//! the load's `need` record on its standard input. The plan's site stages the
//! record it printed ([`stage`]); the placed load consults the hook
//! ([`consult`]) and ends by its answer:
//!
//! * exit 0: the load goes on;
//! * exit 75 ([`REFUSED`]): the load ends, and the process exits 75 naming the
//!   hook's last line of output ([`refusal`] is what `exit_with` reads);
//! * any other status, a signal, or a hook that cannot start: a named error.
//!
//! The hook may wait as long as it likes (a broker queue): the load has opened
//! no card of its own, uploaded no weight and read no host set, and the
//! process holds no lock, while it does. A lane probe before the plan may have
//! made the process's primary context, which stays. Unset, nothing runs and
//! the staged record is never read. The lever takes the hook as an absolute
//! path only, so the program that runs is the file it checked and never one
//! found in `PATH`.

use std::fmt;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Mutex, PoisonError};

/// The hook's exit status that refuses the load (`EX_TEMPFAIL`: try again
/// later, which is what a broker that cannot admit the load now means).
pub const REFUSED: u8 = 75;

/// Why a load did not get the hook's go-ahead.
#[derive(Debug)]
pub enum HookError {
    /// The lever's own refusal ([`bloomery_levers::need_levers`]).
    Levers(String),
    /// The hook is set and no plan staged a record for this load.
    NoRecord { hook: PathBuf },
    /// The hook could not be run, or its output not read.
    Run { hook: PathBuf, source: io::Error },
    /// The hook exited [`REFUSED`]; `line` is its last line of output.
    Refused { hook: PathBuf, line: String },
    /// The hook ended with another status; `status` is as the OS states it.
    Failed {
        hook: PathBuf,
        status: String,
        line: String,
    },
}

impl fmt::Display for HookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let said = |line: &str| {
            if line.is_empty() {
                "it printed nothing".to_string()
            } else {
                format!("it said: {line}")
            }
        };
        match self {
            HookError::Levers(e) => f.write_str(e),
            HookError::NoRecord { hook } => write!(
                f,
                "BLOOMERY_NEED_HOOK names {}, and no plan staged a need record for this load: \
                 the load's plan site prints none",
                hook.display()
            ),
            HookError::Run { hook, source } => {
                write!(
                    f,
                    "the need hook {} could not run: {source}",
                    hook.display()
                )
            }
            HookError::Refused { hook, line } => write!(
                f,
                "the need hook {} refused the load (exit {REFUSED}); {}",
                hook.display(),
                said(line)
            ),
            HookError::Failed { hook, status, line } => write!(
                f,
                "the need hook {} ended with {status}, not 0 or {REFUSED}; {}",
                hook.display(),
                said(line)
            ),
        }
    }
}

impl std::error::Error for HookError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            HookError::Run { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// The last non-empty line of `bytes`, trimmed; empty for none.
fn last_line(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .lines()
        .map(str::trim)
        .rfind(|l| !l.is_empty())
        .unwrap_or_default()
        .to_string()
}

/// Run `hook` with `record` and a newline on its standard input and wait for
/// it. Its standard error is passed on to this process's, so what the hook
/// says reaches the load's log whatever it exits.
fn run(hook: &Path, record: &str) -> Result<(), HookError> {
    let run_err = |source| HookError::Run {
        hook: hook.to_path_buf(),
        source,
    };
    let mut child = Command::new(hook)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(run_err)?;
    let mut stdin = child.stdin.take().expect("the child's stdin was piped");
    // A hook that decides without reading closes its end first: its status
    // is the answer, not the broken pipe.
    match stdin
        .write_all(record.as_bytes())
        .and_then(|()| stdin.write_all(b"\n"))
    {
        Err(e) if e.kind() != io::ErrorKind::BrokenPipe => {
            drop(stdin);
            let _ = child.kill();
            let _ = child.wait();
            return Err(run_err(e));
        }
        _ => drop(stdin),
    }
    let out = child.wait_with_output().map_err(run_err)?;
    if !out.stderr.is_empty() {
        let _ = io::stderr().write_all(&out.stderr);
    }
    let line = match last_line(&out.stdout) {
        l if l.is_empty() => last_line(&out.stderr),
        l => l,
    };
    match out.status.code() {
        Some(0) => Ok(()),
        Some(c) if c == i32::from(REFUSED) => Err(HookError::Refused {
            hook: hook.to_path_buf(),
            line,
        }),
        _ => Err(HookError::Failed {
            hook: hook.to_path_buf(),
            status: out.status.to_string(),
            line,
        }),
    }
}

/// What a process stages for its load and what a refusal leaves behind.
struct Slots {
    /// The need record the plan printed, for the load that follows it.
    staged: Mutex<Option<String>>,
    /// The message of the hook's refusal, once there was one.
    refusal: Mutex<Option<String>>,
}

impl Slots {
    const fn new() -> Slots {
        Slots {
            staged: Mutex::new(None),
            refusal: Mutex::new(None),
        }
    }
}

static SLOTS: Slots = Slots::new();

/// Stage `record`, the `need` line a plan just printed, for the placed load
/// that follows it. A later plan's record replaces it; the load takes it.
pub fn stage(record: &str) {
    stage_in(&SLOTS, record);
}

fn stage_in(slots: &Slots, record: &str) {
    *slots.staged.lock().unwrap_or_else(PoisonError::into_inner) = Some(record.to_string());
}

/// The placed load's consult: the hook `BLOOMERY_NEED_HOOK` names, if any,
/// run with the staged record. Unset, `Ok` and nothing read or run. A refusal
/// is kept for [`refusal`].
pub fn consult() -> Result<(), HookError> {
    let levers = bloomery_levers::need_levers().map_err(|e| HookError::Levers(e.to_string()))?;
    consult_in(&SLOTS, levers.hook.as_deref())
}

fn consult_in(slots: &Slots, hook: Option<&Path>) -> Result<(), HookError> {
    let Some(hook) = hook else {
        return Ok(());
    };
    let record = slots
        .staged
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take()
        .ok_or_else(|| HookError::NoRecord {
            hook: hook.to_path_buf(),
        })?;
    let result = run(hook, &record);
    if let Err(e @ HookError::Refused { .. }) = &result {
        *slots.refusal.lock().unwrap_or_else(PoisonError::into_inner) = Some(e.to_string());
    }
    result
}

/// The message of the hook's refusal in this process, `None` while there was
/// none: a process whose load was refused ends [`REFUSED`] by it, whichever
/// thread or error type carried the refusal up.
#[must_use]
pub fn refusal() -> Option<String> {
    refusal_in(&SLOTS)
}

fn refusal_in(slots: &Slots) -> Option<String> {
    slots
        .refusal
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    /// A directory of the test's own under the system temp dir.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("need-hook-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("a scratch directory");
        dir
    }

    /// An executable shell script in `dir` with `body`.
    fn stub(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        // Only the writer child opens the script for writing, so concurrent spawns cannot
        // inherit a writable descriptor and make the hook's exec fail with ETXTBSY.
        let written = Command::new("/bin/sh")
            .args(["-c", "printf '%s\\n' \"$2\" > \"$1\"", "stub"])
            .arg(&path)
            .arg(format!("#!/bin/sh\n{body}\n"))
            .status()
            .expect("the stub writer runs");
        assert!(written.success(), "the stub is written: {written}");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
            .expect("the stub is executable");
        path
    }

    const RECORD: &str = "need bytes=1 pinned=1 anon=0";

    /// The hook gets the record and a newline on its standard input, and exit
    /// 0 lets the load go on, whatever else the hook prints.
    #[test]
    fn exit_zero_goes_on_and_the_hook_reads_the_record() {
        let dir = scratch("zero");
        let seen = dir.join("seen");
        let hook = stub(
            &dir,
            "hook",
            &format!("cat > {}\necho ignored output", seen.display()),
        );
        run(&hook, RECORD).expect("exit 0 goes on");
        assert_eq!(
            std::fs::read_to_string(&seen).expect("the hook wrote its stdin"),
            format!("{RECORD}\n")
        );
    }

    /// Exit 75 is the refusal, and the error names the hook's last output
    /// line: its standard output's, else its standard error's.
    #[test]
    fn exit_75_refuses_and_names_the_hooks_line() {
        let dir = scratch("refuse");
        let hook = stub(
            &dir,
            "hook",
            "cat > /dev/null\necho first\necho 'the tape holds the box'\necho >&2 noise\nexit 75",
        );
        let Err(HookError::Refused { line, .. }) = run(&hook, RECORD) else {
            panic!("exit 75 is a refusal");
        };
        assert_eq!(line, "the tape holds the box");
        let err = run(&hook, RECORD).unwrap_err().to_string();
        assert!(
            err.contains("refused the load") && err.contains("the tape holds the box"),
            "{err}"
        );
        let quiet = stub(&dir, "quiet", "echo 'only on stderr' >&2\nexit 75");
        let Err(HookError::Refused { line, .. }) = run(&quiet, RECORD) else {
            panic!("exit 75 is a refusal");
        };
        assert_eq!(line, "only on stderr");
        let mute = stub(&dir, "mute", "exit 75");
        let err = run(&mute, RECORD).unwrap_err().to_string();
        assert!(err.contains("it printed nothing"), "{err}");
    }

    /// Any other status, a signal and a hook that is not there or not
    /// executable are named errors that are not the refusal.
    #[test]
    fn every_other_end_is_a_named_error() {
        let dir = scratch("other");
        let failing = stub(&dir, "failing", "echo broker down\nexit 3");
        let Err(HookError::Failed { status, line, .. }) = run(&failing, RECORD) else {
            panic!("exit 3 is a failure");
        };
        assert!(
            status.contains('3') && line == "broker down",
            "{status} {line}"
        );
        let killed = stub(&dir, "killed", "kill -9 $$");
        let Err(HookError::Failed { status, .. }) = run(&killed, RECORD) else {
            panic!("a signal is a failure");
        };
        assert!(status.contains("signal"), "{status}");
        let missing = dir.join("missing");
        assert!(matches!(run(&missing, RECORD), Err(HookError::Run { .. })));
        let plain = dir.join("plain");
        std::fs::write(&plain, "not a program").expect("a plain file");
        let err = run(&plain, RECORD).unwrap_err();
        assert!(matches!(err, HookError::Run { .. }), "{err}");
        assert!(err.to_string().contains("could not run"), "{err}");
    }

    /// A hook that answers without reading its standard input is answered by
    /// its status, not by a broken pipe, however long the record.
    #[test]
    fn a_hook_that_does_not_read_is_still_answered() {
        let dir = scratch("noread");
        let hook = stub(&dir, "hook", "exit 75");
        let long = "x".repeat(1 << 20);
        assert!(matches!(run(&hook, &long), Err(HookError::Refused { .. })));
        let go = stub(&dir, "go", "exit 0");
        run(&go, &long).expect("exit 0 goes on");
    }

    /// Unset, nothing runs and nothing is read; set, the staged record is
    /// taken by the one consult, a second finds none, and a refusal is kept
    /// for the exit.
    #[test]
    fn a_consult_takes_the_staged_record_once() {
        let dir = scratch("consult");
        let seen = dir.join("seen");
        let hook = stub(&dir, "hook", &format!("cat > {}\nexit 75", seen.display()));
        let slots = Slots::new();
        stage_in(&slots, RECORD);
        consult_in(&slots, None).expect("unset runs nothing");
        assert!(!seen.exists(), "nothing ran");
        assert_eq!(refusal_in(&slots), None);
        let Err(HookError::Refused { .. }) = consult_in(&slots, Some(&hook)) else {
            panic!("exit 75 refuses");
        };
        assert_eq!(
            std::fs::read_to_string(&seen).expect("the hook ran"),
            format!("{RECORD}\n")
        );
        let kept = refusal_in(&slots).expect("the refusal is kept");
        assert!(kept.contains("refused the load"), "{kept}");
        let Err(HookError::NoRecord { .. }) = consult_in(&slots, Some(&hook)) else {
            panic!("the record was taken");
        };
        // A later plan stages again.
        stage_in(&slots, "need bytes=2");
        stage_in(&slots, "need bytes=3");
        let go = stub(&dir, "go", "exit 0");
        consult_in(&slots, Some(&go)).expect("exit 0 goes on");
        assert!(matches!(
            consult_in(&slots, Some(&go)),
            Err(HookError::NoRecord { .. })
        ));
    }

    /// Only the refusal leaves a trace for the exit: a failure and a go-ahead
    /// do not.
    #[test]
    fn only_a_refusal_is_kept_for_the_exit() {
        let dir = scratch("kept");
        let slots = Slots::new();
        let failing = stub(&dir, "failing", "exit 3");
        stage_in(&slots, RECORD);
        assert!(matches!(
            consult_in(&slots, Some(&failing)),
            Err(HookError::Failed { .. })
        ));
        assert_eq!(refusal_in(&slots), None);
    }
}
