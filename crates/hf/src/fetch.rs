//! The network part: `curl` for the listing and for each file, then the
//! checks of what it wrote. curl because it is on every machine that runs
//! the server and keeps a TLS stack out of the build; it is asked for
//! `--fail --location --continue-at -`, so an HTTP error is an exit code and
//! a partial file resumes where it stopped.
//!
//! A file lands at its cache path ([`crate::cache_path`]) — the MTP draft
//! beside the set's first shard instead — only after its size and its
//! digest ([`crate::Check`]) match the listing's: curl writes
//! `<file>.part`, the check reads it whole, and the rename makes it the
//! file. Beside it, `<file>.verified` holds the digest it was checked
//! against, so a later start trusts a file of the right size whose marker
//! names the listing's digest without reading it again. A file of the
//! right size with no such marker is read and checked once; one whose
//! content is not the listing's (the repo moved it on) is removed and
//! fetched again (`stale`); one shorter than the listing's is moved back to
//! `.part` and resumed; one longer is refused. A `.part` whose check fails
//! is removed, and the refusal names both digests.
//!
//! A listing that fails at the network — curl cannot resolve the host,
//! connect, finish TLS or hear back in time ([`network_failure`]), not an
//! HTTP status — leaves the cache to stand in: the files under the repo's
//! cache directory that carry a `.verified` marker are the listing
//! ([`verified_entries`]), the same pick runs over them, and exactly one
//! complete set the quant names is used, each file `offline-cached`, after an
//! `Offline` event with the network's reason. No such set, two, or a set
//! missing a verified shard is [`HfError::Offline`] with the cache's own
//! refusal; an HTTP error is refused as it is.
//!
//! The token for a gated repo is `$HF_TOKEN`, else the text of
//! `~/.cache/huggingface/token`. It reaches curl on its standard input as a
//! config line, never in its arguments, and is never printed. curl sends a
//! custom `Authorization` header to the first host only, not to the CDN a
//! download redirects to.

use std::ffi::OsString;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use sha1::Digest as _;

use crate::{Check, Entry, GgufSet, HfError, RepoRef, cache_path, exact, gguf_sets, listing};

/// How often a running download prints its progress.
const PROGRESS_EVERY: Duration = Duration::from_secs(5);

/// Seconds curl waits for the connection alone, not the transfer, so an
/// offline start reaches the cache without curl's default wait.
const CONNECT_TIMEOUT: &str = "15";

/// What a file was found as before its fetch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    /// In the cache, checked: nothing is fetched.
    Cached,
    /// Not there: fetched from byte 0.
    Fetch,
    /// A partial file: fetched from its end.
    Resume,
    /// In the cache with content that is not the listing's: removed and
    /// fetched again.
    Stale,
    /// In the cache with its `.verified` marker, used without a listing:
    /// the network refused the listing.
    OfflineCached,
}

impl State {
    pub fn name(self) -> &'static str {
        match self {
            State::Cached => "cached",
            State::Fetch => "fetch",
            State::Resume => "resume",
            State::Stale => "stale",
            State::OfflineCached => "offline-cached",
        }
    }
}

/// What a resolve says as it goes, for the caller to print.
#[derive(Debug)]
pub enum Event<'a> {
    /// The set a quant picked.
    Set { repo: &'a str, set: &'a GgufSet },
    /// A file before its fetch: its size, its state, and the byte the
    /// fetch starts at (its size when cached).
    File {
        file: &'a str,
        bytes: u64,
        state: State,
        from: u64,
    },
    /// A running download's bytes on disk.
    Progress { file: &'a str, have: u64, of: u64 },
    /// The listing failed at the network and the cache stands in: why.
    Offline { repo: &'a str, why: &'a str },
    /// A file checked: the bytes this run fetched, and the digest.
    Done {
        file: &'a str,
        fetched: u64,
        check: Check,
        digest: &'a str,
    },
}

/// How a resolve reaches the hub and where it caches.
#[derive(Clone, Debug)]
pub struct Client {
    /// `https://huggingface.co`, or another base with the same paths.
    pub base: String,
    pub token: Option<String>,
    pub root: PathBuf,
    pub curl: OsString,
}

/// `$HF_TOKEN` (`env`) when set and not blank, else the token file's text
/// (`file`), trimmed; `None` when neither holds one.
pub fn token_of(env: Option<String>, file: Option<String>) -> Option<String> {
    [env, file]
        .into_iter()
        .flatten()
        .map(|t| t.trim().to_owned())
        .find(|t| !t.is_empty())
}

/// The HTTP status a `--fail` curl names on its stderr (`The requested URL
/// returned error: 404`), if it names one.
pub fn http_status(stderr: &str) -> Option<u16> {
    let at = stderr.find("returned error: ")? + "returned error: ".len();
    stderr[at..]
        .split(|c: char| !c.is_ascii_digit())
        .next()?
        .parse()
        .ok()
}

/// Whether curl's exit `code` is a failure of the network rather than of the
/// hub: a proxy or host it cannot resolve (5, 6), a connection refused (7), a
/// timeout (28), a TLS handshake that failed (35), no reply or a broken one
/// (52, 55, 56). An HTTP status (`--fail`'s 22) is the hub's answer.
pub fn network_failure(code: Option<i32>) -> bool {
    matches!(code, Some(5 | 6 | 7 | 28 | 35 | 52 | 55 | 56))
}

/// curl's failure as a sentence: an HTTP 401 or 403 says the repo may be
/// gated, a 404 that it or the file is not there.
fn curl_why(code: Option<i32>, stderr: &str) -> String {
    let said = stderr.trim().lines().collect::<Vec<_>>().join("; ");
    let hint = match http_status(stderr) {
        Some(401 | 403) => {
            " (no such repo, or a gated or private one: accept its terms on huggingface.co and \
             set HF_TOKEN, or write the token to ~/.cache/huggingface/token)"
        }
        Some(404) => " (no such repo or file)",
        _ => "",
    };
    match code {
        Some(c) => format!("exit {c}: {said}{hint}"),
        None => format!("ended by a signal: {said}"),
    }
}

impl Client {
    /// The hub at `https://huggingface.co`, the cache root `root`, the token
    /// from `$HF_TOKEN` or `$HOME/.cache/huggingface/token`, and `curl` from
    /// `PATH`.
    pub fn new(root: PathBuf) -> Client {
        let file = std::env::var_os("HOME")
            .and_then(|h| fs::read_to_string(Path::new(&h).join(".cache/huggingface/token")).ok());
        Client {
            base: "https://huggingface.co".to_owned(),
            token: token_of(std::env::var("HF_TOKEN").ok(), file),
            root,
            curl: "curl".into(),
        }
    }

    /// A curl command with the token on its standard input (none when
    /// there is no token) and `args`; its stdout and stderr captured.
    fn curl(&self, what: &str, args: &[&std::ffi::OsStr]) -> Result<Vec<u8>, HfError> {
        let mut cmd = Command::new(&self.curl);
        cmd.args(["--fail", "--location", "--silent", "--show-error"]);
        cmd.args(["--connect-timeout", CONNECT_TIMEOUT]);
        if self.token.is_some() {
            cmd.args(["--config", "-"]);
        }
        cmd.args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = cmd.spawn().map_err(|e| HfError::Curl {
            what: what.to_owned(),
            why: format!("{e} (the fetch shells out to curl, which must be on PATH)"),
        })?;
        self.send_token(&mut child, what)?;
        let out = child.wait_with_output().map_err(|e| HfError::Curl {
            what: what.to_owned(),
            why: e.to_string(),
        })?;
        if !out.status.success() {
            let (what, why) = (
                what.to_owned(),
                curl_why(out.status.code(), &String::from_utf8_lossy(&out.stderr)),
            );
            return Err(if network_failure(out.status.code()) {
                HfError::Network { what, why }
            } else {
                HfError::Curl { what, why }
            });
        }
        Ok(out.stdout)
    }

    /// The token as curl's config line on the child's stdin, which is then
    /// closed.
    fn send_token(&self, child: &mut std::process::Child, what: &str) -> Result<(), HfError> {
        let Some(mut stdin) = child.stdin.take() else {
            return Ok(());
        };
        if let Some(t) = &self.token {
            // A token is letters, digits and `_`; a quote or a newline in
            // it would end the config line early.
            if t.contains(['"', '\n', '\r', '\\']) {
                return Err(HfError::Curl {
                    what: what.to_owned(),
                    why: "the Hugging Face token holds a quote, a backslash or a newline".into(),
                });
            }
            writeln!(stdin, "header = \"Authorization: Bearer {t}\"").map_err(|e| {
                HfError::Curl {
                    what: what.to_owned(),
                    why: format!("writing the token to curl: {e}"),
                }
            })?;
        }
        Ok(())
    }

    /// Every file of `repo`'s main branch, each page of the listing
    /// followed to its last.
    pub fn listing(&self, repo: &str) -> Result<Vec<Entry>, HfError> {
        let headers = self
            .root
            .join(format!(".listing-{}.headers", std::process::id()));
        fs::create_dir_all(&self.root).map_err(|e| HfError::io(&self.root, e))?;
        let mut url = format!("{}/api/models/{repo}/tree/main?recursive=true", self.base);
        let mut out = Vec::new();
        loop {
            let body = self.curl(
                &format!("listing {url}"),
                &["--dump-header".as_ref(), headers.as_os_str(), url.as_ref()],
            );
            let head = fs::read_to_string(&headers).unwrap_or_default();
            let _ = fs::remove_file(&headers);
            let body = String::from_utf8(body?).map_err(|e| HfError::Listing {
                repo: repo.to_owned(),
                why: e.to_string(),
            })?;
            out.extend(listing(repo, &body)?);
            match crate::next_link(&head) {
                Some(next) => url = next,
                None => return Ok(out),
            }
        }
    }

    /// `entry` of `repo` in the cache, fetched and checked as the module
    /// says; its path.
    pub fn fetch(
        &self,
        repo: &str,
        entry: &Entry,
        events: &mut dyn FnMut(&Event<'_>),
    ) -> Result<PathBuf, HfError> {
        let dest = cache_path(&self.root, repo, &entry.path)?;
        self.fetch_to(repo, entry, &dest, events)
    }

    /// `entry` of `repo` fetched to `dest` — the MTP draft's `dest` is
    /// beside the set's first shard, not its cache path — checked and marked
    /// as the module says; `dest`.
    fn fetch_to(
        &self,
        repo: &str,
        entry: &Entry,
        dest: &Path,
        events: &mut dyn FnMut(&Event<'_>),
    ) -> Result<PathBuf, HfError> {
        let part = with_suffix(dest, ".part");
        let marker = with_suffix(dest, ".verified");
        let file = entry.path.as_str();
        let mut state = State::Fetch;
        if let Some(len) = size_of(dest)? {
            if len > entry.size {
                return Err(HfError::Size {
                    file: dest.display().to_string(),
                    bytes: len,
                    want: entry.size,
                });
            }
            if len == entry.size {
                let marked = fs::read_to_string(&marker).ok();
                if marked.as_deref().map(str::trim) == Some(entry.digest.as_str()) {
                    events(&Event::File {
                        file,
                        bytes: entry.size,
                        state: State::Cached,
                        from: entry.size,
                    });
                    return Ok(dest.to_path_buf());
                }
                let got = digest(dest, entry.check, entry.size)?;
                if got == entry.digest {
                    events(&Event::File {
                        file,
                        bytes: entry.size,
                        state: State::Cached,
                        from: entry.size,
                    });
                    write_marker(&marker, &got)?;
                    events(&Event::Done {
                        file,
                        fetched: 0,
                        check: entry.check,
                        digest: &got,
                    });
                    return Ok(dest.to_path_buf());
                }
                remove(dest)?;
                state = State::Stale;
            } else {
                rename(dest, &part)?;
            }
            let _ = fs::remove_file(&marker);
        }
        let from = size_of(&part)?.unwrap_or(0);
        if from > entry.size {
            return Err(HfError::Size {
                file: part.display().to_string(),
                bytes: from,
                want: entry.size,
            });
        }
        if from > 0 && state == State::Fetch {
            state = State::Resume;
        }
        events(&Event::File {
            file,
            bytes: entry.size,
            state,
            from,
        });
        if from < entry.size {
            self.download(repo, entry, &part, events)?;
        }
        let len = size_of(&part)?.unwrap_or(0);
        if len != entry.size {
            return Err(HfError::Size {
                file: part.display().to_string(),
                bytes: len,
                want: entry.size,
            });
        }
        let got = digest(&part, entry.check, entry.size)?;
        if got != entry.digest {
            remove(&part)?;
            return Err(HfError::Hash {
                file: dest.display().to_string(),
                check: entry.check,
                got,
                want: entry.digest.clone(),
            });
        }
        write_marker(&marker, &got)?;
        rename(&part, dest)?;
        events(&Event::Done {
            file,
            fetched: entry.size - from,
            check: entry.check,
            digest: &got,
        });
        Ok(dest.to_path_buf())
    }

    /// curl from the end of `part` to the end of the file, a `Progress`
    /// event every [`PROGRESS_EVERY`] while it runs.
    fn download(
        &self,
        repo: &str,
        entry: &Entry,
        part: &Path,
        events: &mut dyn FnMut(&Event<'_>),
    ) -> Result<(), HfError> {
        if let Some(dir) = part.parent() {
            fs::create_dir_all(dir).map_err(|e| HfError::io(dir, e))?;
        }
        let url = format!("{}/{repo}/resolve/main/{}", self.base, entry.path);
        let what = format!("download {url}");
        let mut cmd = Command::new(&self.curl);
        cmd.args(["--fail", "--location", "--silent", "--show-error"]);
        cmd.args(["--connect-timeout", CONNECT_TIMEOUT]);
        if self.token.is_some() {
            cmd.args(["--config", "-"]);
        }
        cmd.args(["--continue-at", "-", "--output"])
            .arg(part)
            .arg(&url)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        let mut child = cmd.spawn().map_err(|e| HfError::Curl {
            what: what.clone(),
            why: format!("{e} (the fetch shells out to curl, which must be on PATH)"),
        })?;
        self.send_token(&mut child, &what)?;
        let mut last = Instant::now();
        let status = loop {
            match child.try_wait() {
                Ok(Some(s)) => break s,
                Ok(None) => {}
                Err(e) => {
                    return Err(HfError::Curl {
                        what,
                        why: e.to_string(),
                    });
                }
            }
            if last.elapsed() >= PROGRESS_EVERY {
                last = Instant::now();
                events(&Event::Progress {
                    file: &entry.path,
                    have: size_of(part)?.unwrap_or(0),
                    of: entry.size,
                });
            }
            std::thread::sleep(Duration::from_millis(100));
        };
        if !status.success() {
            let mut err = String::new();
            if let Some(mut s) = child.stderr.take() {
                let _ = s.read_to_string(&mut err);
            }
            return Err(HfError::Curl {
                what,
                why: curl_why(status.code(), &err),
            });
        }
        Ok(())
    }

    /// The set `r` picks in its repo, every file fetched and checked; the
    /// local paths in shard order (the first shard first). `draft`, the MTP
    /// draft file name a model family runs beside its set, is fetched too
    /// when the repo holds a file of the name ([`crate::mtp_draft`]):
    /// beside the set's first shard, under its own name, not at its repo
    /// path. A repo holding none is fetched as it is, and offline the draft
    /// is not looked for: the file an earlier fetch landed beside the
    /// shards is found by the engine's own rule.
    pub fn resolve(
        &self,
        r: &RepoRef,
        draft: Option<&str>,
        events: &mut dyn FnMut(&Event<'_>),
    ) -> Result<Vec<PathBuf>, HfError> {
        let entries = match self.listing(&r.repo) {
            Err(HfError::Network { why, .. }) => return self.offline_set(r, &why, events),
            other => other?,
        };
        let sets = gguf_sets(&r.repo, &entries)?;
        let set = crate::pick(&r.repo, &sets, r.quant.as_deref())?;
        events(&Event::Set { repo: &r.repo, set });
        let mut files = Vec::with_capacity(set.files.len());
        for e in &set.files {
            files.push(self.fetch(&r.repo, e, events)?);
        }
        if let Some(name) = draft
            && let Some(e) = crate::mtp_draft(&r.repo, &entries, name)?
        {
            // Beside the first shard, under the name the engine opens it by:
            // wherever the repo keeps the draft, the run finds it there.
            let dest = files[0].with_file_name(name);
            self.fetch_to(&r.repo, e, &dest, events)?;
        }
        Ok(files)
    }

    /// The one verified set of the cache `r` picks, the network having
    /// refused the listing for `why`.
    fn offline_set(
        &self,
        r: &RepoRef,
        why: &str,
        events: &mut dyn FnMut(&Event<'_>),
    ) -> Result<Vec<PathBuf>, HfError> {
        let offline = |cache: HfError| HfError::Offline {
            repo: r.repo.clone(),
            why: why.to_owned(),
            cache: Box::new(cache),
        };
        let entries = verified_entries(&self.root, &r.repo).map_err(offline)?;
        let sets = gguf_sets(&r.repo, &entries).map_err(offline)?;
        let set = crate::pick(&r.repo, &sets, r.quant.as_deref()).map_err(offline)?;
        events(&Event::Offline { repo: &r.repo, why });
        events(&Event::Set { repo: &r.repo, set });
        self.offline_files(&r.repo, &set.files, events)
    }

    /// The cache paths of `files`, each an `offline-cached` event.
    fn offline_files(
        &self,
        repo: &str,
        files: &[Entry],
        events: &mut dyn FnMut(&Event<'_>),
    ) -> Result<Vec<PathBuf>, HfError> {
        files
            .iter()
            .map(|e| {
                events(&Event::File {
                    file: &e.path,
                    bytes: e.size,
                    state: State::OfflineCached,
                    from: e.size,
                });
                cache_path(&self.root, repo, &e.path)
            })
            .collect()
    }

    /// The files of `repo` at exactly `paths`, fetched and checked; their
    /// local paths in that order.
    pub fn resolve_exact(
        &self,
        repo: &str,
        paths: &[&str],
        events: &mut dyn FnMut(&Event<'_>),
    ) -> Result<Vec<PathBuf>, HfError> {
        let entries = match self.listing(repo) {
            Err(HfError::Network { why, .. }) => {
                let offline = |cache: HfError| HfError::Offline {
                    repo: repo.to_owned(),
                    why: why.clone(),
                    cache: Box::new(cache),
                };
                let cached = verified_entries(&self.root, repo).map_err(offline)?;
                let files: Vec<Entry> = exact(repo, &cached, paths)
                    .map_err(offline)?
                    .into_iter()
                    .cloned()
                    .collect();
                events(&Event::Offline { repo, why: &why });
                return self.offline_files(repo, &files, events);
            }
            other => other?,
        };
        exact(repo, &entries, paths)?
            .into_iter()
            .map(|e| self.fetch(repo, e, events))
            .collect()
    }
}

/// The files under `repo`'s cache directory in `root` that a fetch checked:
/// each one beside its `.verified` marker, its path relative to the repo's
/// directory, its size on disk and the digest the marker names. A file
/// with no marker (a `.part`, or one a check never passed) is left out; no
/// directory is no file.
pub fn verified_entries(root: &Path, repo: &str) -> Result<Vec<Entry>, HfError> {
    let top = root.join(repo);
    let mut out = Vec::new();
    let mut dirs = vec![top.clone()];
    while let Some(d) = dirs.pop() {
        let read = match fs::read_dir(&d) {
            Ok(r) => r,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(HfError::io(&d, e)),
        };
        for item in read {
            let path = item.map_err(|e| HfError::io(&d, e))?.path();
            let meta = fs::metadata(&path).map_err(|e| HfError::io(&path, e))?;
            if meta.is_dir() {
                dirs.push(path);
                continue;
            }
            let marker = with_suffix(&path, ".verified");
            let Ok(text) = fs::read_to_string(&marker) else {
                continue;
            };
            let digest = text.trim().to_owned();
            let check = match digest.len() {
                64 => Check::Sha256,
                40 => Check::GitSha1,
                _ => continue,
            };
            let rel = path
                .strip_prefix(&top)
                .map_err(|_| HfError::Path {
                    repo: repo.to_owned(),
                    path: path.display().to_string(),
                })?
                .to_string_lossy()
                .into_owned();
            out.push(Entry {
                path: rel,
                size: meta.len(),
                check,
                digest,
            });
        }
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

/// `path` with `suffix` added to its file name.
fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

/// The size of the file at `path`, `None` when there is none.
fn size_of(path: &Path) -> Result<Option<u64>, HfError> {
    match fs::metadata(path) {
        Ok(m) => Ok(Some(m.len())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(HfError::io(path, e)),
    }
}

fn rename(from: &Path, to: &Path) -> Result<(), HfError> {
    fs::rename(from, to).map_err(|e| HfError::io(from, e))
}

fn remove(path: &Path) -> Result<(), HfError> {
    fs::remove_file(path).map_err(|e| HfError::io(path, e))
}

fn write_marker(path: &Path, digest: &str) -> Result<(), HfError> {
    fs::write(path, format!("{digest}\n")).map_err(|e| HfError::io(path, e))
}

/// The digest of the `size`-byte file at `path` by `check`, lowercase hex.
pub fn digest(path: &Path, check: Check, size: u64) -> Result<String, HfError> {
    let mut f = fs::File::open(path).map_err(|e| HfError::io(path, e))?;
    let mut buf = vec![0u8; 8 << 20];
    let mut sha256 = sha2::Sha256::new();
    let mut sha1 = sha1::Sha1::new();
    if check == Check::GitSha1 {
        sha1.update(format!("blob {size}\0").as_bytes());
    }
    loop {
        let n = f.read(&mut buf).map_err(|e| HfError::io(path, e))?;
        if n == 0 {
            break;
        }
        match check {
            Check::Sha256 => sha256.update(&buf[..n]),
            Check::GitSha1 => sha1.update(&buf[..n]),
        }
    }
    let bytes: Vec<u8> = match check {
        Check::Sha256 => sha256.finalize().to_vec(),
        Check::GitSha1 => sha1.finalize().to_vec(),
    };
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch directory under the target dir, fresh for `name`.
    fn scratch(name: &str) -> PathBuf {
        let d = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/hf-tests")
            .join(name);
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).expect("scratch dir");
        d
    }

    fn hex(b: &[u8]) -> String {
        b.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// A served repo under `dir/hub` as `file://` resolves it, one file
    /// `path` of `content`, and a client caching under `dir/cache`.
    fn served(dir: &Path, repo: &str, path: &str, content: &[u8]) -> (Client, Entry) {
        let src = dir.join("hub").join(repo).join("resolve/main").join(path);
        fs::create_dir_all(src.parent().expect("parent")).expect("hub dir");
        fs::write(&src, content).expect("hub file");
        let client = Client {
            base: format!("file://{}", dir.join("hub").display()),
            token: None,
            root: dir.join("cache"),
            curl: "curl".into(),
        };
        let entry = Entry {
            path: path.to_owned(),
            size: content.len() as u64,
            check: Check::Sha256,
            digest: hex(&sha2::Sha256::digest(content)),
        };
        (client, entry)
    }

    fn content() -> Vec<u8> {
        (0..300_000u32).map(|i| (i * 7 % 251) as u8).collect()
    }

    /// A `file://` hub under `dir/hub` as `Client::listing` reads it: the
    /// `files` (repo path → content) served at `resolve/main`, and the tree
    /// listing that names them (`curl` reads `tree/main`, the query of the
    /// API's URL dropped); a client caching under `dir/cache`.
    fn hub(dir: &Path, repo: &str, files: &[(&str, &[u8])]) -> Client {
        let mut json = Vec::new();
        for (path, content) in files {
            let src = dir.join("hub").join(repo).join("resolve/main").join(path);
            fs::create_dir_all(src.parent().expect("parent")).expect("hub dir");
            fs::write(&src, content).expect("hub file");
            json.push(format!(
                "{{\"type\":\"file\",\"path\":\"{path}\",\"size\":{},\"lfs\":{{\"oid\":\"{}\",\"size\":{}}}}}",
                content.len(),
                hex(&sha2::Sha256::digest(content)),
                content.len()
            ));
        }
        let tree = dir.join("hub/api/models").join(repo).join("tree");
        fs::create_dir_all(&tree).expect("tree dir");
        fs::write(tree.join("main"), format!("[{}]", json.join(","))).expect("listing");
        Client {
            base: format!("file://{}", dir.join("hub").display()),
            token: None,
            root: dir.join("cache"),
            curl: "curl".into(),
        }
    }

    /// `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`: the qwen4exp family's
    /// draft file name, what a resolve is given and what the engine's draft
    /// rule looks for beside the target.
    const DRAFT: &str = "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";

    fn states(events: &[String]) -> Vec<String> {
        events
            .iter()
            .filter(|e| e.starts_with("file"))
            .cloned()
            .collect()
    }

    fn run(c: &Client, repo: &str, e: &Entry) -> (Result<PathBuf, HfError>, Vec<String>) {
        let mut seen = Vec::new();
        let r = c.fetch(repo, e, &mut |ev| {
            seen.push(match ev {
                Event::File { state, from, .. } => format!("file {} {from}", state.name()),
                Event::Done { fetched, .. } => format!("done {fetched}"),
                Event::Progress { .. } => "progress".to_owned(),
                Event::Set { .. } => "set".to_owned(),
                Event::Offline { .. } => "offline".to_owned(),
            })
        });
        (r, seen)
    }

    #[test]
    fn a_fetch_lands_checked_and_a_second_fetches_nothing() {
        let d = scratch("fetch");
        let body = content();
        let (c, e) = served(&d, "a/b", "sub/m-Q4_K_M.gguf", &body);
        let (r, seen) = run(&c, "a/b", &e);
        let p = r.expect("fetch");
        assert_eq!(p, d.join("cache/a/b/sub/m-Q4_K_M.gguf"));
        assert_eq!(fs::read(&p).expect("file"), body);
        assert_eq!(seen, ["file fetch 0", &format!("done {}", body.len())]);
        assert!(!with_suffix(&p, ".part").exists());
        let (r, seen) = run(&c, "a/b", &e);
        r.expect("cached");
        assert_eq!(states(&seen), [format!("file cached {}", body.len())]);
        assert_eq!(seen.len(), 1, "a marked file was read again: {seen:?}");
    }

    #[test]
    fn a_truncated_file_resumes_from_its_end() {
        let d = scratch("resume");
        let body = content();
        let (c, e) = served(&d, "a/b", "m-Q8_0.gguf", &body);
        let p = run(&c, "a/b", &e).0.expect("fetch");
        let f = fs::OpenOptions::new().write(true).open(&p).expect("open");
        f.set_len(100_000).expect("truncate");
        let (r, seen) = run(&c, "a/b", &e);
        r.expect("resume");
        assert_eq!(
            seen,
            [
                "file resume 100000",
                &format!("done {}", body.len() - 100_000)
            ]
        );
        assert_eq!(fs::read(&p).expect("file"), body);
    }

    #[test]
    fn a_size_or_digest_mismatch_is_refused_and_the_part_never_used() {
        let d = scratch("mismatch");
        let body = content();
        let (c, mut e) = served(&d, "a/b", "m.gguf", &body);
        e.digest = "0".repeat(64);
        let (r, _) = run(&c, "a/b", &e);
        match r {
            Err(HfError::Hash { got, want, .. }) => {
                assert_eq!(want, "0".repeat(64));
                assert_eq!(got, hex(&sha2::Sha256::digest(&body)));
            }
            other => panic!("{other:?}"),
        }
        let dest = d.join("cache/a/b/m.gguf");
        assert!(!dest.exists() && !with_suffix(&dest, ".part").exists());
        let (c, mut e) = served(&d, "a/b", "m.gguf", &body);
        e.size += 1;
        let (r, _) = run(&c, "a/b", &e);
        assert!(matches!(r, Err(HfError::Size { .. })), "{r:?}");
        assert!(!dest.exists(), "a short file took the final name");
    }

    #[test]
    fn a_cached_file_the_repo_moved_on_is_fetched_again() {
        let d = scratch("stale");
        let body = content();
        let (c, e) = served(&d, "a/b", "m.gguf", &body);
        let p = run(&c, "a/b", &e).0.expect("fetch");
        let mut other = body.clone();
        other[5] ^= 1;
        fs::write(&p, &other).expect("overwrite");
        fs::remove_file(with_suffix(&p, ".verified")).expect("marker");
        let (r, seen) = run(&c, "a/b", &e);
        r.expect("refetch");
        assert_eq!(states(&seen), ["file stale 0"]);
        assert_eq!(fs::read(&p).expect("file"), body);
    }

    #[test]
    fn a_missing_file_is_curls_named_error() {
        let d = scratch("missing");
        let (c, mut e) = served(&d, "a/b", "m.gguf", b"x");
        e.path = "absent.gguf".into();
        match run(&c, "a/b", &e).0 {
            Err(HfError::Curl { what, .. }) => assert!(what.contains("absent.gguf"), "{what}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_git_file_checks_by_its_blob_sha1() {
        let d = scratch("git");
        let p = d.join("README.md");
        fs::write(&p, "hello\n").expect("file");
        // `git hash-object` of "hello\n".
        assert_eq!(
            digest(&p, Check::GitSha1, 6).expect("digest"),
            "ce013625030ba8dba906f756967f9e9ca394464a"
        );
    }

    /// A client whose hub refuses the connection: curl's exit 7.
    fn unreachable(root: &Path) -> Client {
        Client {
            base: "http://127.0.0.1:1".to_owned(),
            token: None,
            root: root.to_owned(),
            curl: "curl".into(),
        }
    }

    /// A hub on a local port that answers every request with `status`.
    fn answering(status: &'static str, root: &Path) -> Client {
        use std::io::Write as _;
        let l = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = l.local_addr().expect("addr").port();
        std::thread::spawn(move || {
            for c in l.incoming().flatten() {
                let mut c = c;
                let mut buf = [0u8; 4096];
                let _ = c.read(&mut buf);
                let _ = write!(
                    c,
                    "HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                );
            }
        });
        Client {
            base: format!("http://127.0.0.1:{port}"),
            token: None,
            root: root.to_owned(),
            curl: "curl".into(),
        }
    }

    /// `paths` of `repo` fetched into `dir/cache` from a `file://` hub, each
    /// checked and marked.
    fn cached(dir: &Path, repo: &str, paths: &[&str]) {
        for p in paths {
            let (c, e) = served(dir, repo, p, &content());
            c.fetch(repo, &e, &mut |_| {}).expect("fetch");
        }
    }

    fn offline_run(c: &Client, r: &str) -> (Result<Vec<PathBuf>, HfError>, Vec<String>) {
        let r = RepoRef::parse(r).expect("repo");
        let mut seen = Vec::new();
        let got = c.resolve(&r, None, &mut |ev| {
            seen.push(match ev {
                Event::Offline { why, .. } => format!("offline {why}"),
                Event::File { file, state, .. } => format!("file {file} {}", state.name()),
                Event::Set { set, .. } => format!("set {}", set.name),
                _ => "other".to_owned(),
            })
        });
        (got, seen)
    }

    #[test]
    fn offline_the_one_verified_set_stands_in() {
        let d = scratch("offline-one");
        cached(&d, "a/b", &["m-Q4_K_M.gguf", "m-Q8_0.gguf"]);
        let (got, seen) = offline_run(&unreachable(&d.join("cache")), "a/b:Q4_K_M");
        assert_eq!(
            got.expect("offline"),
            [d.join("cache/a/b/m-Q4_K_M.gguf")],
            "{seen:?}"
        );
        assert!(seen[0].starts_with("offline exit 7"), "{seen:?}");
        assert_eq!(
            seen[1..],
            ["set m-Q4_K_M", "file m-Q4_K_M.gguf offline-cached"]
        );
    }

    #[test]
    fn offline_zero_or_two_verified_sets_are_refused() {
        let d = scratch("offline-zero-two");
        cached(
            &d,
            "a/b",
            &["m-Q8_0.gguf", "one/m-Q4_K_M.gguf", "two/m-Q4_K_M.gguf"],
        );
        let c = unreachable(&d.join("cache"));
        match offline_run(&c, "a/b:Q6_K").0 {
            Err(HfError::Offline { cache, .. }) => {
                assert!(matches!(*cache, HfError::NoMatch { .. }), "{cache}")
            }
            other => panic!("{other:?}"),
        }
        match offline_run(&c, "a/b:Q4_K_M").0 {
            Err(HfError::Offline { cache, .. }) => {
                assert!(matches!(*cache, HfError::TwoMatches { .. }), "{cache}")
            }
            other => panic!("{other:?}"),
        }
        match offline_run(&c, "x/y:Q4_K_M").0 {
            Err(HfError::Offline { cache, .. }) => {
                assert!(matches!(*cache, HfError::NoGguf { .. }), "{cache}")
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn offline_a_file_no_check_passed_is_not_used() {
        let d = scratch("offline-unverified");
        cached(&d, "a/b", &["m-Q4_K_M.gguf"]);
        let f = d.join("cache/a/b/m-Q4_K_M.gguf");
        fs::remove_file(with_suffix(&f, ".verified")).expect("marker");
        fs::write(d.join("cache/a/b/m-Q8_0.gguf.part"), b"x").expect("part");
        match offline_run(&unreachable(&d.join("cache")), "a/b:Q4_K_M").0 {
            Err(HfError::Offline { cache, .. }) => {
                assert!(matches!(*cache, HfError::NoGguf { .. }), "{cache}")
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn an_http_error_stays_a_refusal_beside_a_verified_set() {
        let d = scratch("offline-http");
        cached(&d, "a/b", &["m-Q4_K_M.gguf"]);
        match offline_run(&answering("404 Not Found", &d.join("cache")), "a/b:Q4_K_M").0 {
            Err(HfError::Curl { why, .. }) => assert!(why.contains("404"), "{why}"),
            other => panic!("{other:?}"),
        }
        match offline_run(
            &answering("503 Service Unavailable", &d.join("cache")),
            "a/b",
        )
        .0
        {
            Err(HfError::Curl { why, .. }) => assert!(why.contains("503"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn offline_the_head_files_stand_in_by_name() {
        let d = scratch("offline-exact");
        cached(
            &d,
            "c/h",
            &["joint_head.safetensors", "joint_head_config.json"],
        );
        let c = unreachable(&d.join("cache"));
        let mut seen = Vec::new();
        let got = c
            .resolve_exact(
                "c/h",
                &["joint_head.safetensors", "joint_head_config.json"],
                &mut |ev| {
                    if let Event::File { state, .. } = ev {
                        seen.push(state.name());
                    }
                },
            )
            .expect("offline");
        assert_eq!(got[1], d.join("cache/c/h/joint_head_config.json"));
        assert_eq!(seen, ["offline-cached", "offline-cached"]);
        assert!(matches!(
            c.resolve_exact("c/h", &["nope.json"], &mut |_| {}),
            Err(HfError::Offline { .. })
        ));
    }

    #[test]
    fn a_resolve_fetches_the_set_and_the_draft_beside_its_first_shard() {
        let d = scratch("draft-beside");
        let body = content();
        let c = hub(
            &d,
            "a/b",
            &[
                ("UD-Q4_K_XL/m-UD-Q4_K_XL-00001-of-00002.gguf", &body),
                ("UD-Q4_K_XL/m-UD-Q4_K_XL-00002-of-00002.gguf", &body),
                ("MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf", b"draft"),
            ],
        );
        let r = RepoRef::parse("a/b:UD-Q4_K_XL").expect("repo");
        let mut seen = Vec::new();
        let got = c
            .resolve(&r, Some(DRAFT), &mut |ev| {
                if let Event::File { file, state, .. } = ev {
                    seen.push(format!("file {file} {}", state.name()));
                }
            })
            .expect("resolve");
        assert_eq!(got.len(), 2, "{got:?}");
        assert_eq!(
            got[0],
            d.join("cache/a/b/UD-Q4_K_XL/m-UD-Q4_K_XL-00001-of-00002.gguf")
        );
        // The draft lands beside the first shard under its own name — where
        // the engine's draft rule looks for it — and nowhere else: not at
        // its repo path under MTP/.
        let beside = got[0].with_file_name(DRAFT);
        assert_eq!(beside, d.join("cache/a/b/UD-Q4_K_XL").join(DRAFT));
        assert_eq!(fs::read(&beside).expect("draft"), b"draft");
        assert!(
            !d.join("cache/a/b/MTP").exists(),
            "the repo path was fetched"
        );
        assert_eq!(
            seen,
            [
                "file UD-Q4_K_XL/m-UD-Q4_K_XL-00001-of-00002.gguf fetch".to_owned(),
                "file UD-Q4_K_XL/m-UD-Q4_K_XL-00002-of-00002.gguf fetch".to_owned(),
                format!("file MTP/{DRAFT} fetch"),
            ]
        );
        // A second resolve fetches nothing: the draft too is `cached`.
        let mut again = Vec::new();
        c.resolve(&r, Some(DRAFT), &mut |ev| {
            if let Event::File { state, .. } = ev {
                again.push(state.name());
            }
        })
        .expect("again");
        assert_eq!(again, ["cached", "cached", "cached"]);
    }

    #[test]
    fn a_repo_without_the_drafts_file_fetches_only_the_set() {
        let d = scratch("draft-none");
        let body = content();
        let c = hub(&d, "a/b", &[("m-Q4_K_M.gguf", &body)]);
        let r = RepoRef::parse("a/b:Q4_K_M").expect("repo");
        let mut seen = Vec::new();
        let got = c
            .resolve(&r, Some(DRAFT), &mut |ev| {
                if let Event::File { file, .. } = ev {
                    seen.push(file.to_string());
                }
            })
            .expect("resolve");
        assert_eq!(got, [d.join("cache/a/b/m-Q4_K_M.gguf")]);
        assert_eq!(seen, ["m-Q4_K_M.gguf"]);
    }

    #[test]
    fn offline_the_draft_is_not_looked_for_but_stands_beside_the_shards() {
        let d = scratch("draft-offline");
        let body = content();
        let online = hub(
            &d,
            "a/b",
            &[("Q4_K_XL/m-Q4_K_XL.gguf", &body), (DRAFT, b"draft")],
        );
        let r = RepoRef::parse("a/b:Q4_K_XL").expect("repo");
        online.resolve(&r, Some(DRAFT), &mut |_| {}).expect("fetch");
        assert!(
            d.join("cache/a/b/Q4_K_XL").join(DRAFT).is_file(),
            "the online fetch landed the draft beside the set"
        );
        // The network gone: the set stands in from the cache and the draft
        // is not looked for — the engine finds the fetched file by its name.
        let c = unreachable(&d.join("cache"));
        let mut seen = Vec::new();
        let got = c
            .resolve(&r, Some(DRAFT), &mut |ev| {
                if let Event::File { file, state, .. } = ev {
                    seen.push(format!("file {file} {}", state.name()));
                }
            })
            .expect("offline");
        assert_eq!(got, [d.join("cache/a/b/Q4_K_XL/m-Q4_K_XL.gguf")]);
        assert_eq!(seen, ["file Q4_K_XL/m-Q4_K_XL.gguf offline-cached"]);
    }

    #[test]
    fn network_failures_are_curls_connect_codes_not_http() {
        for c in [5, 6, 7, 28, 35, 52, 55, 56] {
            assert!(network_failure(Some(c)), "{c}");
        }
        for c in [22, 37, 23, 0] {
            assert!(!network_failure(Some(c)), "{c}");
        }
        assert!(!network_failure(None));
    }

    #[test]
    fn tokens_and_curl_errors_read_as_said() {
        assert_eq!(
            token_of(Some(" hf_x \n".into()), Some("hf_y".into())).as_deref(),
            Some("hf_x")
        );
        assert_eq!(
            token_of(Some("".into()), Some("hf_y\n".into())).as_deref(),
            Some("hf_y")
        );
        assert_eq!(token_of(None, None), None);
        let err = "curl: (22) The requested URL returned error: 401";
        assert_eq!(http_status(err), Some(401));
        assert!(curl_why(Some(22), err).contains("gated"));
        assert_eq!(http_status("curl: (6) Could not resolve host"), None);
    }
}
