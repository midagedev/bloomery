//! The HTTP side of the binaries that drive `bloomery-serve-ds41`
//! (`gate_ds41_serve`, `soak_ds41_serve`): the server started beside the
//! calling binary and killed by its handle on every way out, its address read
//! from its stderr, and one request through curl.

use std::fs::File;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use serde_json::Value;

use crate::GateError;

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
        let exe = Self::exe()?;
        let mut cmd = Command::new(&exe);
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

/// One request through curl: the status and the body.
pub fn curl(url: &str, body: Option<&Value>, stream: bool) -> Result<(u16, String), GateError> {
    let mut c = Command::new("curl");
    c.args(["-sS", "--max-time", "600", "-w", "\n%{http_code}"]);
    if stream {
        c.arg("-N");
    }
    if let Some(b) = body {
        c.args(["-H", "Content-Type: application/json", "-d", &b.to_string()]);
    }
    let out = c.arg(url).output()?;
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
