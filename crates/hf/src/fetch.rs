//! The network part: `curl` for the listing and for each file, then the
//! checks of what it wrote. curl because it is on every machine that runs
//! the server and keeps a TLS stack out of the build; it is asked for
//! `--fail --location`, so an HTTP error is an exit code.
//!
//! A file lands at its cache path ([`crate::cache_path`]) — the MTP draft
//! beside the set's first shard instead — only after its size and its
//! digest ([`crate::Check`]) match the listing's: curl transfers write
//! `<file>.part` by ranges, several at once, each resumed where it stopped
//! (`ranges`), the digest is read in order behind them, and the rename makes
//! it the file. Beside it, `<file>.verified` holds the digest it was checked
//! against, so a later start trusts a file of the right size whose marker
//! names the listing's digest without reading it again. A file of the
//! right size with no such marker is read and checked once; one whose
//! content is not the listing's (the repo moved it on) is removed and
//! fetched again (`stale`); one shorter than the listing's is moved back to
//! `.part` and resumed, unless its marker names another digest: then it is
//! the repo's earlier version (an upload that grew), removed and fetched
//! again (`stale`); one longer, or a `.part` longer than the file, is not
//! the listing's either: removed and fetched again (`stale`). A `.part`
//! whose check fails is removed, and the refusal names both digests.
//! Each destination has a persistent `.lock` file: an exclusive OS lock serializes the state
//! check, hashing, download and publication across threads and processes. The verified marker
//! is atomically replaced after the completed file, so an interrupted publication is rechecked.
//! A listed file whose name ends in `.lock`, `.part` or `.verified`, in any case, is refused
//! when its own fetch starts, files listed before it being fetched already: it would be another
//! listed file's lock, part or marker. The cache's discovery skips such names.
//!
//! A resolve whose files are all on disk at their listed sizes reads the
//! unmarked ones in parallel before the fetches, to mark the ones that match.
//! That pass gives no verdict: a file that does not match is left to its
//! fetch, which removes it and fetches it again.
//!
//! A listing that fails at the network — curl cannot resolve the host,
//! connect, finish TLS or hear back in time ([`network_failure`]) — or that
//! the hub answers busy or down (HTTP 429 or 5xx, [`hub_unavailable`]), and
//! every listing under `HF_HUB_OFFLINE` ([`hub_offline`]), which asks the hub
//! nothing, leaves the cache to stand in: the files under the repo's cache
//! directory that carry a `.verified` marker are the listing
//! ([`verified_entries`]), the same pick runs over them, and exactly one
//! complete set the quant names is used, each file `offline-cached`, after an
//! `Offline` event with the reason. No such set, two, or a set missing a
//! verified shard is [`HfError::Offline`] with the cache's own refusal; any
//! other HTTP error is refused as it is.
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
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use sha1::Digest as _;

use crate::{Check, Entry, GgufSet, HfError, RepoRef, cache_path, exact, gguf_sets, listing};

mod ranges;

use ranges::{Ranges, Split};

/// How often a running download prints its progress.
const PROGRESS_EVERY: Duration = Duration::from_secs(5);

/// How many files of one snapshot [`mark_snapshot`] hashes at once.
const SNAPSHOT_HASH_THREADS: usize = 4;

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
    /// A partial file: each of its ranges fetched from where it stopped.
    Resume,
    /// In the cache with content that is not the listing's: removed and
    /// fetched again.
    Stale,
    /// In the cache with its `.verified` marker, used without a listing:
    /// the network or the hub refused the listing, or `HF_HUB_OFFLINE` asked
    /// for none.
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
    /// The safetensors checkpoint a repo with no GGUF set holds (`crate::checkpoint_manifest`):
    /// its files and their bytes.
    Checkpoint {
        repo: &'a str,
        files: usize,
        bytes: u64,
    },
    /// A file before its fetch: its size, its state, and the bytes of it
    /// already on disk that the fetch keeps (its size when cached).
    File {
        file: &'a str,
        bytes: u64,
        state: State,
        from: u64,
    },
    /// A running download's bytes on disk, whichever ranges hold them.
    Progress { file: &'a str, have: u64, of: u64 },
    /// No listing — the network or the hub refused it, or `HF_HUB_OFFLINE`
    /// asked for none — and the cache stands in: why.
    Offline { repo: &'a str, why: &'a str },
    /// The repo's MTP draft `file` left unfetched: the set cannot run it,
    /// for `why` ([`Draft::usable`]).
    DraftSkip { file: &'a str, why: &'a str },
    /// A file checked: the bytes this run fetched, and the digest.
    Done {
        file: &'a str,
        fetched: u64,
        check: Check,
        digest: &'a str,
    },
}

/// The MTP draft a resolve fetches beside the set it picks
/// ([`Client::resolve`], [`Client::resolve_drafts`]): the file name the
/// family's draft goes by, and whether the set can run it — `usable` reads the
/// set's first shard, once it is fetched, and says why not.
pub struct Draft<'a> {
    pub name: &'a str,
    pub usable: &'a dyn Fn(&Path) -> Result<(), String>,
}

/// The model sources the caller can open, checked before any model file is fetched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sources {
    /// Only GGUF model files, including split sets.
    Gguf,
    /// GGUF model files or a safetensors checkpoint folder.
    GgufOrCheckpoint,
}

/// How a resolve reaches the hub and where it caches.
#[derive(Clone, Debug)]
pub struct Client {
    /// `https://huggingface.co`, or another base with the same paths.
    pub base: String,
    pub token: Option<String>,
    pub root: PathBuf,
    pub curl: OsString,
    /// `HF_HUB_OFFLINE` ([`hub_offline`]): no listing is asked for; the
    /// cache's verified files stand in.
    pub offline: bool,
    /// The revision of the repo a resolve lists and downloads: `main`, or a
    /// commit hash, a tag or another branch a caller pins.
    pub revision: String,
}

/// Why a resolve under `HF_HUB_OFFLINE` asked the hub for no listing.
const HUB_OFFLINE_WHY: &str = "HF_HUB_OFFLINE is set";

/// Whether `HF_HUB_OFFLINE` (`v`, unset as `None`) asks for no network:
/// `1`, `true`, `yes` and `on` do, `0`, `false`, `no`, `off` and the empty
/// value do not (any case, as `huggingface_hub` reads it); any other value is
/// [`HfError::HubOffline`].
pub fn hub_offline(v: Option<&str>) -> Result<bool, HfError> {
    let Some(v) = v else {
        return Ok(false);
    };
    match v.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "" | "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(HfError::HubOffline(v.to_owned())),
    }
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

/// Whether curl's exit `code` and the HTTP `status` it named are the hub
/// answering busy or down: 429 (too many requests) or any 5xx. Every other
/// status is the hub's answer about the request.
pub fn hub_unavailable(code: Option<i32>, status: Option<u16>) -> bool {
    code == Some(22) && matches!(status, Some(429 | 500..=599))
}

/// curl's failure at `what` as its error: [`HfError::Network`] for the
/// network ([`network_failure`]), [`HfError::Unavailable`] for a busy or
/// down hub ([`hub_unavailable`]), [`HfError::Curl`] for everything else.
fn failure(what: &str, code: Option<i32>, stderr: &str) -> HfError {
    let (what, why) = (what.to_owned(), curl_why(code, stderr));
    if network_failure(code) {
        HfError::Network { what, why }
    } else if hub_unavailable(code, http_status(stderr)) {
        HfError::Unavailable { what, why }
    } else {
        HfError::Curl { what, why }
    }
}

/// A repo's listing, or why there is none and the cache stands in.
enum Listed {
    Hub(Vec<Entry>),
    Cache(String),
}

/// What [`curl_why`] says of a 401 or 403; `gpu-gates` reads the class of a refusal from it.
pub const DENIED_HINT: &str = "a gated or private one";

/// What [`curl_why`] says of a 404; `gpu-gates` reads the class of a refusal from it.
pub const NOT_THERE_HINT: &str = "no such repo or file";

/// curl's failure as a sentence: an HTTP 401 or 403 says the repo may be
/// gated ([`DENIED_HINT`]), a 404 that it or the file is not there
/// ([`NOT_THERE_HINT`]).
fn curl_why(code: Option<i32>, stderr: &str) -> String {
    let said = stderr.trim().lines().collect::<Vec<_>>().join("; ");
    let hint = match http_status(stderr) {
        Some(401 | 403) => format!(
            " (no such repo, or {DENIED_HINT}: accept its terms on huggingface.co and \
             set HF_TOKEN, or write the token to ~/.cache/huggingface/token)"
        ),
        Some(404) => format!(" ({NOT_THERE_HINT})"),
        _ => String::new(),
    };
    match code {
        Some(c) => format!("exit {c}: {said}{hint}"),
        None => format!("ended by a signal: {said}"),
    }
}

impl Client {
    /// The hub at `https://huggingface.co`, the cache root `root`, the token
    /// from `$HF_TOKEN` or `$HOME/.cache/huggingface/token`, `curl` from
    /// `PATH`, and offline when `$HF_HUB_OFFLINE` says so ([`hub_offline`],
    /// which refuses a value it does not read).
    pub fn new(root: PathBuf) -> Result<Client, HfError> {
        let file = std::env::var_os("HOME")
            .and_then(|h| fs::read_to_string(Path::new(&h).join(".cache/huggingface/token")).ok());
        let offline = match std::env::var_os("HF_HUB_OFFLINE") {
            None => false,
            Some(v) => {
                hub_offline(Some(v.to_str().ok_or_else(|| {
                    HfError::HubOffline(v.to_string_lossy().into_owned())
                })?))?
            }
        };
        Ok(Client {
            base: "https://huggingface.co".to_owned(),
            token: token_of(std::env::var("HF_TOKEN").ok(), file),
            root,
            curl: "curl".into(),
            offline,
            revision: "main".to_owned(),
        })
    }

    /// curl with the flags every call takes, reading the token's config line
    /// from its standard input when there is a token; its stderr captured.
    fn command(&self) -> Command {
        let mut cmd = Command::new(&self.curl);
        cmd.args(["--fail", "--location", "--silent", "--show-error"]);
        cmd.args(["--connect-timeout", CONNECT_TIMEOUT]);
        if self.token.is_some() {
            cmd.args(["--config", "-"]);
        }
        cmd.stdin(Stdio::piped()).stderr(Stdio::piped());
        cmd
    }

    /// `cmd`, a [`Client::command`], started and given the token.
    fn spawn(&self, mut cmd: Command, what: &str) -> Result<Child, HfError> {
        let mut child = cmd.spawn().map_err(|e| HfError::Curl {
            what: what.to_owned(),
            why: format!("{e} (the fetch shells out to curl, which must be on PATH)"),
        })?;
        self.send_token(&mut child, what)?;
        Ok(child)
    }

    /// A header file for curl's `--dump-header` under the cache root, its own
    /// for each call: `kind` calls run at once in one process.
    fn headers_file(&self, kind: &str) -> PathBuf {
        static CALLS: AtomicUsize = AtomicUsize::new(0);
        self.root.join(format!(
            ".{kind}-{}-{}.headers",
            std::process::id(),
            CALLS.fetch_add(1, Ordering::Relaxed)
        ))
    }

    /// A [`Client::command`] with `args`; its stdout.
    fn curl(&self, what: &str, args: &[&std::ffi::OsStr]) -> Result<Vec<u8>, HfError> {
        let mut cmd = self.command();
        cmd.args(args).stdout(Stdio::piped());
        let child = self.spawn(cmd, what)?;
        let out = child.wait_with_output().map_err(|e| HfError::Curl {
            what: what.to_owned(),
            why: e.to_string(),
        })?;
        if !out.status.success() {
            return Err(failure(
                what,
                out.status.code(),
                &String::from_utf8_lossy(&out.stderr),
            ));
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

    /// `repo`'s listing ([`Client::listing`]), or why the cache stands in
    /// for it: `HF_HUB_OFFLINE` asks the hub nothing, and a listing the
    /// network refuses or the hub answers busy or down is none. Any other
    /// failure is the refusal. Here and in every other call that takes a
    /// `repo`, it is the repo at its revision, [`RepoRef::id`].
    fn listed(&self, repo: &str) -> Result<Listed, HfError> {
        if self.offline {
            return Ok(Listed::Cache(HUB_OFFLINE_WHY.to_owned()));
        }
        match self.listing(repo) {
            Ok(entries) => Ok(Listed::Hub(entries)),
            Err(HfError::Network { why, .. } | HfError::Unavailable { why, .. }) => {
                Ok(Listed::Cache(why))
            }
            Err(e) => Err(e),
        }
    }

    /// Every file of `repo` at its revision ([`Client::hub_at`]), each page
    /// of the listing followed to its last.
    pub fn listing(&self, repo: &str) -> Result<Vec<Entry>, HfError> {
        let (repo, rev) = self.hub_at(repo);
        self.listing_at(repo, rev)
    }

    /// The repo and the revision the hub serves for `id` ([`RepoRef::id`]):
    /// the revision `id` names, else the client's [`Client::revision`].
    pub fn hub_at<'a>(&'a self, id: &'a str) -> (&'a str, &'a str) {
        id.split_once('@').unwrap_or((id, &self.revision))
    }

    /// [`Client::listing`] of `repo` at revision `rev` (a branch, a tag or a
    /// commit hash).
    pub fn listing_at(&self, repo: &str, rev: &str) -> Result<Vec<Entry>, HfError> {
        let headers = self.headers_file("listing");
        fs::create_dir_all(&self.root).map_err(|e| HfError::io(&self.root, e))?;
        let mut url = format!("{}/api/models/{repo}/tree/{rev}?recursive=true", self.base);
        let mut out = Vec::new();
        loop {
            let page = self
                .curl(
                    &format!("listing {url}"),
                    &["--dump-header".as_ref(), headers.as_os_str(), url.as_ref()],
                )
                .and_then(|body| {
                    let head =
                        fs::read_to_string(&headers).map_err(|e| HfError::io(&headers, e))?;
                    Ok((body, head))
                });
            let _ = fs::remove_file(&headers);
            let (body, head) = page?;
            let body = String::from_utf8(body).map_err(|e| HfError::Listing {
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
        self.fetch_to(repo, entry, &dest, Split::HUB, events)
    }

    /// `entry` of `repo` fetched to `dest` — the MTP draft's `dest` is
    /// beside the set's first shard, not its cache path — in ranges `split`
    /// cuts, checked and marked as the module says; `dest`.
    fn fetch_to(
        &self,
        repo: &str,
        entry: &Entry,
        dest: &Path,
        split: Split,
        events: &mut dyn FnMut(&Event<'_>),
    ) -> Result<PathBuf, HfError> {
        let _lock = lock_file(dest)?;
        let part = with_suffix(dest, ".part");
        let marker = with_suffix(dest, ".verified");
        let file = entry.path.as_str();
        let mut state = State::Fetch;
        if let Some(len) = size_of(dest)? {
            let marked = fs::read_to_string(&marker).ok();
            // A file takes its name only after its check, so a marker naming
            // another digest makes a shorter file an earlier version, not a cut
            // download.
            let earlier = marked
                .as_deref()
                .is_some_and(|m| m.trim() != entry.digest.as_str());
            if len > entry.size || (len < entry.size && earlier) {
                remove(dest)?;
                state = State::Stale;
            } else if len == entry.size {
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
                // The file's first bytes, whatever ranges an earlier part held.
                ranges::forget(&part)?;
                rename(dest, &part)?;
            }
            let _ = fs::remove_file(&marker);
        }
        let (ranges, stale) = Ranges::open(entry, &part, split)?;
        if stale {
            state = State::Stale;
        }
        let from = ranges.held();
        if from > 0 && state == State::Fetch {
            state = State::Resume;
        }
        events(&Event::File {
            file,
            bytes: entry.size,
            state,
            from,
        });
        let got = self.download(repo, entry, &part, ranges, split, events)?;
        if got != entry.digest {
            remove(&part)?;
            ranges::forget(&part)?;
            return Err(HfError::Hash {
                file: dest.display().to_string(),
                check: entry.check,
                got,
                want: entry.digest.clone(),
            });
        }
        ranges::forget(&part)?;
        if size_of(&marker)?.is_some() {
            remove(&marker)?;
        }
        rename(&part, dest)?;
        write_marker(&marker, &got)?;
        events(&Event::Done {
            file,
            fetched: entry.size - from,
            check: entry.check,
            digest: &got,
        });
        Ok(dest.to_path_buf())
    }

    /// The `len` bytes of `path` in `repo` at revision `rev` (`main`, or a
    /// commit hash) from byte `from`, read by HTTP Range: the same curl,
    /// token and errors as every other call, nothing cached and nothing
    /// checked against the listing's digest. A hub that
    /// ignores the range and answers the whole file is stopped by its
    /// `Content-Length` (`--max-filesize`, [`range_limit`]) instead of being read; a body of
    /// another length than `len` is [`HfError::Range`], so the caller clips
    /// `len` to the file's listed size first.
    pub fn fetch_range(
        &self,
        repo: &str,
        rev: &str,
        path: &str,
        from: u64,
        len: u64,
    ) -> Result<Vec<u8>, HfError> {
        let url = format!("{}/{repo}/resolve/{rev}/{path}", self.base);
        let what = format!("range {from}+{len} of {url}");
        if len == 0 {
            return Ok(Vec::new());
        }
        let range = format!("{from}-{}", from + len - 1);
        let max = range_limit(len).to_string();
        let body = self.curl(
            &what,
            &[
                "--range".as_ref(),
                range.as_ref(),
                "--max-filesize".as_ref(),
                max.as_ref(),
                url.as_ref(),
            ],
        )?;
        if body.len() as u64 != len {
            return Err(HfError::Range {
                file: format!("{repo}/{path}"),
                from,
                want: len,
                got: body.len() as u64,
            });
        }
        Ok(body)
    }

    /// The set `r` picks in its repo, every file fetched and checked; the
    /// local paths in shard order (the first shard first). A repo with no
    /// GGUF set and a safetensors checkpoint ([`crate::checkpoint_manifest`])
    /// gives the checkpoint's folder, its files fetched and checked, only when `sources`
    /// includes checkpoints. Otherwise it is refused before any file is fetched. `draft`, the MTP
    /// draft a model family runs beside its set, is fetched too when the
    /// repo holds a file of its name ([`crate::mtp_draft`]) and the fetched
    /// set can run it ([`Draft::usable`]; a `DraftSkip` event says why not):
    /// beside the set's first shard, under its own name, not at its repo
    /// path. A repo holding none is fetched as it is, and offline the draft
    /// is not looked for: the file an earlier fetch landed beside the
    /// shards is found by the engine's own rule.
    pub fn resolve(
        &self,
        r: &RepoRef,
        sources: Sources,
        draft: Option<&Draft<'_>>,
        events: &mut dyn FnMut(&Event<'_>),
    ) -> Result<Vec<PathBuf>, HfError> {
        self.resolve_drafts(r, sources, draft.map_or(&[], std::slice::from_ref), events)
    }

    /// [`Client::resolve`] for a caller that does not know the set's family
    /// before it is fetched: `drafts` holds each family's draft, and each
    /// whose file name the repo holds is fetched or skipped by its own
    /// [`Draft::usable`], which reads the fetched first shard and refuses a
    /// set of another family by name. No draft, nothing fetched and nothing
    /// said.
    pub fn resolve_drafts(
        &self,
        r: &RepoRef,
        sources: Sources,
        drafts: &[Draft<'_>],
        events: &mut dyn FnMut(&Event<'_>),
    ) -> Result<Vec<PathBuf>, HfError> {
        let id = r.id();
        let entries = match self.listed(&id)? {
            Listed::Hub(entries) => entries,
            Listed::Cache(why) => return self.offline_set(r, sources, &why, events),
        };
        let sets = gguf_sets(&id, &entries)?;
        if sets.is_empty() && entries.iter().any(|e| e.path == safetensors::Folder::INDEX) {
            let dir = checkpoint(r, sources, &self.root)?;
            let index = exact(&id, &entries, &[safetensors::Folder::INDEX])?[0];
            self.mark_snapshot_if_ready(&id, std::slice::from_ref(index))?;
            let index_path = self.fetch(&id, index, &mut |_| {})?;
            let files = crate::checkpoint_manifest(&id, &entries, &index_path)?;
            checkpoint_event(&id, &files, events);
            self.mark_snapshot_if_ready(&id, &files)?;
            for e in &files {
                self.fetch(&id, e, events)?;
            }
            return Ok(vec![dir]);
        }
        let set = crate::pick(&id, &sets, r.quant.as_deref())?;
        events(&Event::Set { repo: &id, set });
        self.mark_snapshot_if_ready(&id, &set.files)?;
        let mut files = Vec::with_capacity(set.files.len());
        for e in &set.files {
            files.push(self.fetch(&id, e, events)?);
        }
        for d in drafts {
            if let Some(e) = crate::mtp_draft(&id, &entries, d.name)? {
                match (d.usable)(&files[0]) {
                    // Beside the first shard, under the name the engine opens
                    // it by: wherever the repo keeps the draft, the run finds
                    // it there.
                    Ok(()) => {
                        let dest = files[0].with_file_name(d.name);
                        self.fetch_to(&id, e, &dest, Split::HUB, events)?;
                    }
                    Err(why) => events(&Event::DraftSkip {
                        file: &e.path,
                        why: &why,
                    }),
                }
            }
        }
        Ok(files)
    }

    /// The path in `repo` of the image projector that goes with the model file at `model` (its
    /// path in the repo, a set's first shard): [`crate::mmproj_sibling`] of the listing, `None`
    /// when the repo holds none. Where the cache stands in for the listing, as for a set, the
    /// verified files are the entries.
    pub fn mmproj_path(&self, repo: &str, model: &str) -> Result<Option<String>, HfError> {
        let entries = match self.listed(repo)? {
            Listed::Hub(entries) => entries,
            Listed::Cache(_) => verified_entries(&self.root, repo)?,
        };
        Ok(crate::mmproj_sibling(&entries, model).map(|e| e.path.clone()))
    }

    /// The one verified set of the cache `r` picks, there being no listing
    /// for `why`.
    fn offline_set(
        &self,
        r: &RepoRef,
        sources: Sources,
        why: &str,
        events: &mut dyn FnMut(&Event<'_>),
    ) -> Result<Vec<PathBuf>, HfError> {
        let id = r.id();
        let offline = |cache: HfError| HfError::Offline {
            repo: id.clone(),
            why: why.to_owned(),
            cache: Box::new(cache),
        };
        let entries = verified_entries(&self.root, &id).map_err(offline)?;
        let sets = gguf_sets(&id, &entries).map_err(offline)?;
        if sets.is_empty() && entries.iter().any(|e| e.path == safetensors::Folder::INDEX) {
            let dir = checkpoint(r, sources, &self.root).map_err(offline)?;
            let index = dir.join(safetensors::Folder::INDEX);
            let files = crate::checkpoint_manifest(&id, &entries, &index).map_err(offline)?;
            events(&Event::Offline { repo: &id, why });
            checkpoint_event(&id, &files, events);
            self.offline_files(&id, &files, events)?;
            return Ok(vec![dir]);
        }
        let set = crate::pick(&id, &sets, r.quant.as_deref()).map_err(offline)?;
        events(&Event::Offline { repo: &id, why });
        events(&Event::Set { repo: &id, set });
        self.offline_files(&id, &set.files, events)
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
        let entries = match self.listed(repo)? {
            Listed::Cache(why) => {
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
            Listed::Hub(entries) => entries,
        };
        let files: Vec<Entry> = exact(repo, &entries, paths)?.into_iter().cloned().collect();
        self.mark_snapshot_if_ready(repo, &files)?;
        files.iter().map(|e| self.fetch(repo, e, events)).collect()
    }

    /// When every `files` entry is already on disk at its listed size, hashes in parallel the ones
    /// whose `.verified` marker does not name their digest and marks each that matches, so
    /// [`Client::fetch`] trusts it without re-reading. A file that does not match is left to
    /// [`Client::fetch`], which removes it and fetches it again.
    fn mark_snapshot_if_ready(&self, repo: &str, files: &[Entry]) -> Result<(), HfError> {
        if snapshot_on_disk(&self.root, repo, files) {
            mark_snapshot(&self.root, repo, &unmarked(&self.root, repo, files))?;
        }
        Ok(())
    }
}

/// Every `files` entry of `repo` under `root` is present at its listed size.
fn snapshot_on_disk(root: &Path, repo: &str, files: &[Entry]) -> bool {
    files.iter().all(|e| {
        cache_path(root, repo, &e.path)
            .ok()
            .and_then(|p| size_of(&p).ok().flatten())
            .is_some_and(|n| n == e.size)
    })
}

/// The files of `files` whose `.verified` marker does not name their digest: the ones a fetch would hash.
fn unmarked(root: &Path, repo: &str, files: &[Entry]) -> Vec<Entry> {
    files
        .iter()
        .filter(|e| {
            cache_path(root, repo, &e.path)
                .ok()
                .and_then(|p| fs::read_to_string(with_suffix(&p, ".verified")).ok())
                .is_none_or(|m| m.trim() != e.digest.as_str())
        })
        .cloned()
        .collect()
}

/// Each of `entries` of `repo` under `root`, hashed up to [`SNAPSHOT_HASH_THREADS`] at a time, and
/// marked when it hashes to its listing digest. A file that does not match gets no marker and no
/// verdict here: [`Client::fetch`] removes it and fetches it again. A read that fails is the error.
fn mark_snapshot(root: &Path, repo: &str, entries: &[Entry]) -> Result<(), HfError> {
    let next = AtomicUsize::new(0);
    thread::scope(|scope| {
        let mut workers = Vec::new();
        for _ in 0..SNAPSHOT_HASH_THREADS.min(entries.len()) {
            workers.push(scope.spawn(|| mark_from(&next, root, repo, entries)));
        }
        workers
            .into_iter()
            .try_for_each(|w| w.join().unwrap_or_else(|p| std::panic::resume_unwind(p)))
    })
}

/// One of [`mark_snapshot`]'s workers: takes the next entry from `next` until none is left. A
/// failed read ends it, and every other worker after the entry it is on.
fn mark_from(
    next: &AtomicUsize,
    root: &Path,
    repo: &str,
    entries: &[Entry],
) -> Result<(), HfError> {
    while let Some(entry) = entries.get(next.fetch_add(1, Ordering::Relaxed)) {
        if let Err(e) = mark_if_match(root, repo, entry) {
            next.store(entries.len(), Ordering::Relaxed);
            return Err(e);
        }
    }
    Ok(())
}

/// `entry`'s marker, written when its file hashes to the listing's digest; a file of any other
/// digest is left as it is.
fn mark_if_match(root: &Path, repo: &str, entry: &Entry) -> Result<(), HfError> {
    let path = cache_path(root, repo, &entry.path)?;
    let _lock = lock_file(&path)?;
    let got = digest(&path, entry.check, entry.size)?;
    if got == entry.digest {
        write_marker(&with_suffix(&path, ".verified"), &got)?;
    }
    Ok(())
}

/// The folder in `root`'s cache of the checkpoint of `r`'s repo, which a resolve gives
/// in place of a set's first shard; a quant is refused by name, as the
/// checkpoint is the repo's one model. A caller that accepts only GGUF is refused before the event.
fn checkpoint(r: &RepoRef, sources: Sources, root: &Path) -> Result<PathBuf, HfError> {
    let id = r.id();
    if let Some(quant) = &r.quant {
        return Err(HfError::CheckpointQuant {
            repo: id,
            quant: quant.clone(),
        });
    }
    if sources == Sources::Gguf {
        return Err(HfError::CheckpointUnsupported { repo: id });
    }
    let index = cache_path(root, &id, safetensors::Folder::INDEX)?;
    Ok(index.parent().unwrap_or(root).to_path_buf())
}

fn checkpoint_event(repo: &str, files: &[Entry], events: &mut dyn FnMut(&Event<'_>)) {
    events(&Event::Checkpoint {
        repo,
        files: files.len(),
        bytes: files.iter().map(|f| f.size).sum(),
    });
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
            let item = item.map_err(|e| HfError::io(&d, e))?;
            let path = item.path();
            // A download can rename its partial between readdir and metadata.
            let Some(meta) = metadata_of(&path)? else {
                continue;
            };
            if meta.is_dir() {
                dirs.push(path);
                continue;
            }
            let marker = with_suffix(&path, ".verified");
            if sidecar_suffix(&path).is_some() || !marker.is_file() {
                continue;
            }
            let _lock = lock_file(&path)?;
            let Some(size) = size_of(&path)? else {
                continue;
            };
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
                size,
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

/// A bounded record name beside `path`, keyed by its basename. `suffix` ends in `.lock.lock`,
/// which no listed file or listed file's lock can have (see [`lock_file`]).
fn record_path(path: &Path, suffix: &str) -> PathBuf {
    let name = path.file_name().expect("a cache record has a file name");
    let key = gguf::digest::sha256_hex(name.as_encoded_bytes());
    path.with_file_name(format!(".{key}{suffix}"))
}

/// The size of the file at `path`, `None` when there is none.
fn size_of(path: &Path) -> Result<Option<u64>, HfError> {
    Ok(metadata_of(path)?.map(|m| m.len()))
}

/// Metadata of `path`, following symlinks; `None` when its target is absent.
fn metadata_of(path: &Path) -> Result<Option<fs::Metadata>, HfError> {
    match fs::metadata(path) {
        Ok(m) => Ok(Some(m)),
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

fn remove_if_there(path: &Path) -> Result<(), HfError> {
    match fs::remove_file(path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(HfError::io(path, e)),
        _ => Ok(()),
    }
}

fn write_marker(path: &Path, digest: &str) -> Result<(), HfError> {
    write_text(path, &format!("{digest}\n"))
}

/// `text` at `path`, published by rename so a reader sees the old text or the new. The text is
/// written beside it under a name ending in `.lock.lock`, which is no listed file's (see
/// [`lock_file`]).
fn write_text(path: &Path, text: &str) -> Result<(), HfError> {
    let part = record_path(path, ".lock.lock");
    fs::write(&part, text).map_err(|e| HfError::io(&part, e))?;
    rename(&part, path)
}

/// The suffix of a destination's lock, part or marker that `path` ends in, matched without case
/// as a case-insensitive file system does: a listed file of such a name would be another listed
/// file's sidecar.
fn sidecar_suffix(path: &Path) -> Option<&'static str> {
    let bytes = path.as_os_str().as_encoded_bytes();
    [".lock", ".part", ".verified"].into_iter().find(|s| {
        bytes.len() >= s.len() && bytes[bytes.len() - s.len()..].eq_ignore_ascii_case(s.as_bytes())
    })
}

/// A persistent lock file, never removed: replacing its inode could admit another writer. A cache
/// this process cannot write and that has no lock file yet is read without one: this process
/// cannot publish into it, and a writer publishes each file and its marker by rename.
fn lock_file(dest: &Path) -> Result<Option<fs::File>, HfError> {
    if let Some(suffix) = sidecar_suffix(dest) {
        return Err(HfError::io(
            dest,
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("the {suffix} suffix is reserved for cache sidecars"),
            ),
        ));
    }
    let path = with_suffix(dest, ".lock");
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(|e| HfError::io(dir, e))?;
    }
    let file = match fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&path)
    {
        Ok(file) => file,
        Err(e)
            if matches!(
                e.kind(),
                std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::ReadOnlyFilesystem
            ) =>
        {
            match fs::File::open(&path) {
                Ok(file) => file,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(e) => return Err(HfError::io(&path, e)),
            }
        }
        Err(e) => return Err(HfError::io(&path, e)),
    };
    file.lock().map_err(|e| HfError::io(&path, e))?;
    Ok(Some(file))
}

/// The bytes `--max-filesize` allows a range read of `len` bytes: curl holds the redirect the hub
/// answers first (its body counts) to the same limit, and a hub's redirect body is a few hundred
/// bytes whatever the range, so a short range gets this much room beyond `len`.
fn range_limit(len: u64) -> u64 {
    const REDIRECT_BODY: u64 = 4096;
    len.saturating_add(REDIRECT_BODY)
}

/// A file's digest by its [`Check`], fed its bytes in order.
enum Hasher {
    Sha256(gguf::digest::Sha256),
    GitSha1(sha1::Sha1),
}

impl Hasher {
    /// For a Git blob, `size` is the header's size.
    fn new(check: Check, size: u64) -> Hasher {
        match check {
            Check::Sha256 => Hasher::Sha256(gguf::digest::Sha256::new()),
            Check::GitSha1 => {
                let mut sha1 = sha1::Sha1::new();
                // "blob ", at most 20 decimal digits for a u64, and NUL.
                let mut header = [0u8; 26];
                let mut remaining = &mut header[..];
                write!(remaining, "blob {size}\0").expect("a u64 Git blob header fits");
                let used = 26 - remaining.len();
                sha1.update(&header[..used]);
                Hasher::GitSha1(sha1)
            }
        }
    }

    fn update(&mut self, bytes: &[u8]) {
        match self {
            Hasher::Sha256(h) => h.update(bytes),
            Hasher::GitSha1(h) => h.update(bytes),
        }
    }

    /// The digest, lowercase hex.
    fn finish(self) -> String {
        match self {
            Hasher::Sha256(h) => gguf::digest::hex(&h.finish()),
            Hasher::GitSha1(h) => gguf::digest::hex(&h.finalize()),
        }
    }
}

/// The digest of the whole file at `path` by `check`, lowercase hex.
/// For a Git blob, `size` supplies the header's size, not a read limit.
pub fn digest(path: &Path, check: Check, size: u64) -> Result<String, HfError> {
    let mut f = fs::File::open(path).map_err(|e| HfError::io(path, e))?;
    let mut hasher = Hasher::new(check, size);
    let mut buf = vec![0u8; 64 << 10];
    loop {
        let n = f.read(&mut buf).map_err(|e| HfError::io(path, e))?;
        if n == 0 {
            return Ok(hasher.finish());
        }
        hasher.update(&buf[..n]);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A scratch directory under the target dir, fresh for `name`.
    pub(crate) fn scratch(name: &str) -> PathBuf {
        let d = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/hf-tests")
            .join(name);
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).expect("scratch dir");
        d
    }

    pub(super) fn hex(b: &[u8]) -> String {
        b.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// A served repo under `dir/hub` as `file://` resolves it, one file
    /// `path` of `content`, and a client caching under `dir/cache`.
    pub(super) fn served(dir: &Path, repo: &str, path: &str, content: &[u8]) -> (Client, Entry) {
        let src = dir.join("hub").join(repo).join("resolve/main").join(path);
        fs::create_dir_all(src.parent().expect("parent")).expect("hub dir");
        fs::write(&src, content).expect("hub file");
        let client = Client {
            base: format!("file://{}", dir.join("hub").display()),
            token: None,
            root: dir.join("cache"),
            curl: "curl".into(),
            offline: false,
            revision: "main".to_owned(),
        };
        let entry = Entry {
            path: path.to_owned(),
            size: content.len() as u64,
            check: Check::Sha256,
            digest: hex(&sha2::Sha256::digest(content)),
        };
        (client, entry)
    }

    pub(super) fn content() -> Vec<u8> {
        (0..300_000u32).map(|i| (i * 7 % 251) as u8).collect()
    }

    /// A `file://` hub under `dir/hub` as `Client::listing` reads it: the
    /// `files` (repo path → content) served at `resolve/main`, and the tree
    /// listing that names them (`curl` reads `tree/main`, the query of the
    /// API's URL dropped); a client caching under `dir/cache`.
    fn hub(dir: &Path, repo: &str, files: &[(&str, &[u8])]) -> Client {
        hub_at(dir, repo, "main", files)
    }

    /// [`hub`] at the revision `rev`: `resolve/<rev>` and `tree/<rev>`.
    fn hub_at(dir: &Path, repo: &str, rev: &str, files: &[(&str, &[u8])]) -> Client {
        let mut json = Vec::new();
        for (path, content) in files {
            let src = dir
                .join("hub")
                .join(repo)
                .join("resolve")
                .join(rev)
                .join(path);
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
        fs::write(tree.join(rev), format!("[{}]", json.join(","))).expect("listing");
        Client {
            base: format!("file://{}", dir.join("hub").display()),
            token: None,
            root: dir.join("cache"),
            curl: "curl".into(),
            offline: false,
            revision: "main".to_owned(),
        }
    }

    /// `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`: the qwen4exp family's
    /// draft file name, what a resolve is given and what the engine's draft
    /// rule looks for beside the target.
    const DRAFT: &str = "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";

    /// The draft of [`DRAFT`]'s name, which every set can run.
    const RUNS: Draft<'static> = Draft {
        name: DRAFT,
        usable: &|_| Ok(()),
    };

    fn states(events: &[String]) -> Vec<String> {
        events
            .iter()
            .filter(|e| e.starts_with("file"))
            .cloned()
            .collect()
    }

    fn run(c: &Client, repo: &str, e: &Entry) -> (Result<PathBuf, HfError>, Vec<String>) {
        recorded(|events| c.fetch(repo, e, events))
    }

    /// [`run`] in the ranges `split` cuts.
    pub(super) fn run_split(
        c: &Client,
        e: &Entry,
        split: Split,
    ) -> (Result<PathBuf, HfError>, Vec<String>) {
        let dest = cache_path(&c.root, "a/b", &e.path).expect("cache path");
        recorded(|events| c.fetch_to("a/b", e, &dest, split, events))
    }

    /// What `fetch` gives, and each event it says as a line.
    fn recorded(
        fetch: impl FnOnce(&mut dyn FnMut(&Event<'_>)) -> Result<PathBuf, HfError>,
    ) -> (Result<PathBuf, HfError>, Vec<String>) {
        let mut seen = Vec::new();
        let r = fetch(&mut |ev| {
            seen.push(match ev {
                Event::File { state, from, .. } => format!("file {} {from}", state.name()),
                Event::Done { fetched, .. } => format!("done {fetched}"),
                Event::DraftSkip { .. } => "draft-skip".to_owned(),
                Event::Progress { .. } => "progress".to_owned(),
                Event::Set { .. } => "set".to_owned(),
                Event::Checkpoint { .. } => "checkpoint".to_owned(),
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
    fn fetch_refuses_reserved_sidecar_destinations_before_touching_sidecars() {
        let d = scratch("reserved-lock");
        for suffix in [".lock", ".LOCK", ".Lock", ".part", ".Verified"] {
            let name = format!("weights/x.safetensors{suffix}");
            let (c, e) = served(&d, "a/b", &name, b"checkpoint bytes");
            let dest = cache_path(&c.root, "a/b", &name).unwrap();
            fs::create_dir_all(dest.parent().unwrap()).unwrap();
            fs::write(&dest, b"").unwrap();
            let held = fs::File::open(&dest).unwrap();
            let (result, events) = run(&c, "a/b", &e);
            let err = result.expect_err("a cache entry cannot replace a sidecar");
            assert!(
                matches!(&err, HfError::Io { err, .. }
                if err.kind() == std::io::ErrorKind::InvalidInput),
                "{err}"
            );
            assert!(
                err.to_string()
                    .contains(&format!("{} suffix is reserved", suffix.to_lowercase())),
                "{err}"
            );
            assert!(events.is_empty(), "{events:?}");
            assert_eq!(fs::read(&dest).unwrap(), b"");
            assert_eq!(held.metadata().unwrap().len(), 0);
            for sidecar in [".lock", ".part", ".verified"] {
                assert!(!with_suffix(&dest, sidecar).exists(), "{sidecar}");
            }
            assert!(!record_path(&with_suffix(&dest, ".verified"), ".lock.lock").exists());
            let absent = d.join("absent").join(format!("x{suffix}"));
            assert!(lock_file(&absent).is_err());
            assert!(!absent.parent().unwrap().exists());
        }
    }

    #[cfg(unix)]
    #[test]
    fn readable_cache_locks_need_no_write_access() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};

        let d = scratch("readonly-locks");
        let (mut c, e) = served(&d, "a/b", "sub/m-Q4_K_M.gguf", &content());
        let dest = run(&c, "a/b", &e).0.unwrap();
        let lock = with_suffix(&dest, ".lock");
        let inode = fs::metadata(&lock).unwrap().ino();
        fs::set_permissions(&lock, fs::Permissions::from_mode(0o444)).unwrap();
        match fs::OpenOptions::new().write(true).open(&lock) {
            Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::PermissionDenied),
            Ok(_) => {
                eprintln!("read-only lock regression needs an account without permission override");
                return;
            }
        }
        let (hit, seen) = run(&c, "a/b", &e);
        assert_eq!(hit.unwrap(), dest);
        assert_eq!(states(&seen), [format!("file cached {}", e.size)]);
        c.offline = true;
        let (resolved, seen) = offline_run(&c, "a/b:Q4_K_M");
        assert_eq!(resolved.unwrap(), std::slice::from_ref(&dest));
        assert!(
            seen.iter().any(|s| s.contains("offline-cached")),
            "{seen:?}"
        );
        assert_eq!(fs::metadata(&lock).unwrap().ino(), inode);

        // A cache written before lock files existed, in a folder this account cannot write.
        let dir = dest.parent().unwrap();
        fs::remove_file(&lock).unwrap();
        fs::set_permissions(dir, fs::Permissions::from_mode(0o555)).unwrap();
        let (resolved, seen) = offline_run(&c, "a/b:Q4_K_M");
        c.offline = false;
        let (hit, _) = run(&c, "a/b", &e);
        fs::set_permissions(dir, fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(resolved.unwrap(), std::slice::from_ref(&dest));
        assert!(
            seen.iter().any(|s| s.contains("offline-cached")),
            "{seen:?}"
        );
        assert_eq!(hit.unwrap(), dest);
        assert!(!lock.exists());
    }

    /// A listed file named like another one's part sidecar keeps its bytes: the cache's own records
    /// take names no listed file and no listed file's lock can have.
    #[test]
    fn listed_files_named_like_another_files_records_keep_their_bytes() {
        let d = scratch("record-names");
        // PIN(2026-10-31): The two .part-ending cases are now rejected by the reserved-sidecar tests.
        let names = ["x.bin.part.ranges", "x.bin"];
        let bodies: Vec<Vec<u8>> = names.iter().map(|n| n.repeat(3).into_bytes()).collect();
        let files: Vec<_> = names
            .iter()
            .zip(&bodies)
            .map(|(n, b)| (*n, &b[..]))
            .collect();
        let c = hub(&d, "a/b", &files);
        let got = c
            .resolve_exact("a/b", &names, &mut |_| {})
            .expect("resolve");
        for (path, body) in got.iter().zip(&bodies) {
            assert_eq!(
                fs::read(path).ok().as_ref(),
                Some(body),
                "{}",
                path.display()
            );
        }
    }

    /// A listed file named like another listed file's part or marker is refused when its fetch
    /// starts, in either order, so no returned path holds another file's bytes or digest; the
    /// cache's discovery skips such a name an earlier fetch left marked.
    #[test]
    fn listed_files_named_like_another_files_part_or_marker_are_refused() {
        for suffix in [".part", ".verified", ".VERIFIED"] {
            let other = format!("x.bin{suffix}");
            for (i, names) in [[other.as_str(), "x.bin"], ["x.bin", other.as_str()]]
                .into_iter()
                .enumerate()
            {
                let d = scratch(&format!("sidecar-names-{}-{i}", &suffix[1..]));
                let c = hub(
                    &d,
                    "a/b",
                    &[
                        (names[0], names[0].as_bytes()),
                        (names[1], names[1].as_bytes()),
                    ],
                );
                let err = c
                    .resolve_exact("a/b", &names, &mut |_| {})
                    .expect_err("a listed name cannot take another file's sidecar");
                assert!(
                    err.to_string()
                        .contains(&format!("{} suffix is reserved", suffix.to_lowercase())),
                    "{names:?}: {err}"
                );
            }
            let d = scratch(&format!("sidecar-names-{}-cached", &suffix[1..]));
            let (c, e) = served(&d, "a/b", "x.bin", b"x.bin");
            let x = run(&c, "a/b", &e).0.expect("x.bin");
            write_marker(&with_suffix(&x, suffix), &e.digest).unwrap();
            write_marker(
                &with_suffix(&with_suffix(&x, suffix), ".verified"),
                &e.digest,
            )
            .unwrap();
            let found = verified_entries(&c.root, "a/b").expect("discovery");
            assert_eq!(
                found.iter().map(|e| &e.path[..]).collect::<Vec<_>>(),
                ["x.bin"]
            );
        }
    }

    #[test]
    fn long_file_names_leave_room_for_records_and_their_temporaries() {
        let d = scratch("long-record-names");
        for len in [224, 241] {
            let name = format!("{}.gguf", "x".repeat(len - 5));
            let (c, e) = served(&d, "a/b", &name, &content());
            let (got, seen) = run(&c, "a/b", &e);
            let path = got.expect("long-name fetch");
            assert_eq!(fs::read(&path).unwrap(), content());
            assert_eq!(seen, ["file fetch 0", "done 300000"]);
            assert_eq!(
                fs::read_to_string(with_suffix(&path, ".verified"))
                    .unwrap()
                    .trim(),
                e.digest
            );
        }
    }

    #[test]
    fn publishing_the_file_precedes_publishing_its_marker() {
        let d = scratch("publish-order");
        let (c, e) = served(&d, "a/b", "m.gguf", &content());
        let dest = cache_path(&c.root, "a/b", &e.path).unwrap();
        // A marker write fails after the verified bytes have taken their final name.
        fs::create_dir_all(record_path(&with_suffix(&dest, ".verified"), ".lock.lock")).unwrap();
        run(&c, "a/b", &e).0.expect_err("marker publication fails");
        assert_eq!(digest(&dest, e.check, e.size).unwrap(), e.digest);
        assert!(!with_suffix(&dest, ".verified").exists());
    }

    #[test]
    fn publishing_a_marker_replaces_its_inode() {
        let d = scratch("marker-replacement");
        let marker = d.join("m.verified");
        fs::write(&marker, "old\n").unwrap();
        let mut earlier = fs::File::open(&marker).unwrap();
        write_marker(&marker, "new").unwrap();
        let mut old = String::new();
        earlier.read_to_string(&mut old).unwrap();
        assert_eq!(old, "old\n");
        assert_eq!(fs::read_to_string(&marker).unwrap(), "new\n");
        assert!(!record_path(&marker, ".lock.lock").exists());
    }

    #[test]
    fn snapshot_marking_and_discovery_wait_for_the_file_lock() {
        use std::sync::mpsc;
        use std::time::Duration;

        let d = scratch("snapshot-locking");
        let (c, e) = served(&d, "a/b", "m.gguf", &content());
        let dest = run(&c, "a/b", &e).0.unwrap();
        for discover in [false, true] {
            let guard = lock_file(&dest).unwrap();
            let (started, ready) = mpsc::channel();
            let (done, result) = mpsc::channel();
            thread::scope(|scope| {
                scope.spawn(|| {
                    started.send(()).unwrap();
                    let value = if discover {
                        verified_entries(&c.root, "a/b").map(drop)
                    } else {
                        mark_if_match(&c.root, "a/b", &e)
                    };
                    done.send(value).unwrap();
                });
                ready.recv().unwrap();
                let held = result.recv_timeout(Duration::from_millis(200));
                drop(guard);
                assert!(
                    matches!(held, Err(mpsc::RecvTimeoutError::Timeout)),
                    "discover={discover}: {held:?}"
                );
                result
                    .recv_timeout(Duration::from_secs(5))
                    .unwrap()
                    .unwrap();
            });
        }
    }

    #[cfg(unix)]
    #[test]
    fn concurrent_fetches_share_one_download() {
        use std::os::unix::fs::PermissionsExt;

        let d = std::env::current_dir().unwrap();
        if d.join("curl").is_file() {
            let (mut c, e) = served(&d, "a/b", "sub/m.gguf", &content());
            c.curl = d.join("curl").into_os_string();
            run(&c, "a/b", &e).0.expect("concurrent fetch");
            return;
        }
        let d = scratch("concurrent-fetch");
        let script = d.join("curl");
        fs::write(
            &script,
            "#!/bin/sh\ncd -- \"$(dirname -- \"$0\")\" || exit 1\nprintf x >> calls\nsleep 1\nexec curl \"$@\"\n",
        )
        .unwrap();
        fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
        // Both processes use the same partial file, including the same resume offset.
        let (c, e) = served(&d, "a/b", "sub/m.gguf", &content());
        let dest = cache_path(&c.root, "a/b", &e.path).unwrap();
        fs::create_dir_all(dest.parent().unwrap()).unwrap();
        fs::write(with_suffix(&dest, ".part"), &content()[..100_000]).unwrap();
        let spawn = || {
            Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "fetch::tests::concurrent_fetches_share_one_download",
                ])
                .current_dir(&d)
                .spawn()
                .unwrap()
        };
        let mut first = spawn();
        let mut second = spawn();
        let a = first.wait().unwrap();
        let b = second.wait().unwrap();
        assert!(a.success() && b.success(), "fetches: {a}, {b}");
        assert_eq!(fs::read(d.join("calls")).unwrap(), b"x", "downloaded twice");
        assert_eq!(digest(&dest, e.check, e.size).unwrap(), e.digest);
        assert_eq!(
            fs::read_to_string(with_suffix(&dest, ".verified"))
                .unwrap()
                .trim(),
            e.digest
        );
    }

    #[test]
    fn checkpoint_manifest_requires_its_shards_online_and_offline() {
        let d = scratch("checkpoint-manifest");
        let index = br#"{"weight_map":{"a":"weights/a.safetensors","b":"weights/b.safetensors"}}"#;
        let mut c = hub(
            &d,
            "a/b",
            &[
                (safetensors::Folder::INDEX, index),
                ("weights/a.safetensors", b"one"),
                ("weights/b.safetensors", b"two"),
                ("unused.safetensors", b"unused"),
                ("config.json", b"{}"),
                ("chat_template.jinja", b"template"),
                ("extra/config.json", b"unused config"),
            ],
        );
        // The files reported after the checkpoint event make up its bytes, the index included.
        let (mut total, mut after) = (None, 0);
        let dir = open_checkpoint(&c, &mut |ev| match ev {
            Event::Checkpoint { bytes, .. } => total = Some(*bytes),
            Event::File { bytes, .. } if total.is_some() => after += bytes,
            _ => {}
        })
        .unwrap()
        .remove(0);
        assert_eq!(total, Some(after));
        assert_eq!(fs::read(dir.join("weights/a.safetensors")).unwrap(), b"one");
        assert_eq!(fs::read(dir.join("weights/b.safetensors")).unwrap(), b"two");
        assert!(!dir.join("unused.safetensors").exists());
        assert_eq!(fs::read(dir.join("config.json")).unwrap(), b"{}");
        assert_eq!(
            fs::read(dir.join("chat_template.jinja")).unwrap(),
            b"template"
        );
        assert!(!dir.join("extra/config.json").exists());
        // The folder reader follows directory links; discovery must find the same shards.
        #[cfg(unix)]
        {
            let stored = dir.join("stored-weights");
            fs::rename(dir.join("weights"), &stored).unwrap();
            std::os::unix::fs::symlink(stored, dir.join("weights")).unwrap();
        }
        c.offline = true;
        open_checkpoint(&c, &mut |_| {}).expect("complete offline checkpoint");
        for missing in ["weights/b.safetensors", "weights/b.safetensors.verified"] {
            let path = dir.join(missing);
            let saved = with_suffix(&path, ".saved");
            fs::rename(&path, &saved).unwrap();
            let err = open_checkpoint(&c, &mut |_| {}).expect_err("incomplete offline checkpoint");
            assert!(matches!(err, HfError::Offline { .. }), "{err}");
            assert!(err.to_string().contains("weights/b.safetensors"), "{err}");
            fs::rename(saved, path).unwrap();
        }
        c.offline = false;
        let tree = d.join("hub/api/models/a/b/tree/main");
        let json = fs::read_to_string(&tree).unwrap();
        fs::write(
            &tree,
            json.replace("weights/b.safetensors", "weights/c.safetensors"),
        )
        .unwrap();
        let err = open_checkpoint(&c, &mut |_| {}).expect_err("shard absent from listing");
        assert!(matches!(err, HfError::NoFile { .. }), "{err}");
        assert!(err.to_string().contains("weights/b.safetensors"), "{err}");
    }

    #[test]
    fn checkpoint_manifest_refuses_invalid_shard_paths_and_maps() {
        let d = scratch("checkpoint-invalid-manifest");
        for index in [
            "not json",
            "{}",
            r#"{"weight_map":{}}"#,
            r#"{"weight_map":{"a":1}}"#,
            r#"{"weight_map":{"a":"../a.safetensors"}}"#,
            r#"{"weight_map":{"a":"/a.safetensors"}}"#,
            r#"{"weight_map":{"a":"weights//a.safetensors"}}"#,
            r#"{"weight_map":{"a":"C:a.safetensors"}}"#,
        ] {
            let c = hub(&d, "a/b", &[(safetensors::Folder::INDEX, index.as_bytes())]);
            let err = open_checkpoint(&c, &mut |_| {}).expect_err(index);
            assert!(matches!(err, HfError::Listing { .. }), "{err}");
        }
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

    /// A cached file shorter than the repo's whose marker names another
    /// digest is the repo's earlier version (an upload that grew), not a cut
    /// download: fetched again whole, never resumed onto the earlier bytes,
    /// which ends in a digest refusal on the first start after the upload.
    #[test]
    fn a_shorter_file_of_an_earlier_version_is_fetched_again() {
        let d = scratch("earlier");
        let earlier = content();
        let (c, e) = served(&d, "a/b", "m.gguf", &earlier);
        let p = run(&c, "a/b", &e).0.expect("fetch");
        let mut grown = earlier;
        grown[5] ^= 1;
        grown.extend_from_slice(b"grown");
        let (c, e) = served(&d, "a/b", "m.gguf", &grown);
        let (r, seen) = run(&c, "a/b", &e);
        r.expect("the grown upload fetched whole, not resumed onto the earlier version");
        assert_eq!(states(&seen), ["file stale 0"]);
        assert_eq!(fs::read(&p).expect("file"), grown);
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
    fn an_unmarked_same_size_corrupt_cache_is_revalidated_and_fetched_again() {
        let d = scratch("stale");
        let body = content();
        let (c, e) = served(&d, "a/b", "m.gguf", &body);
        let p = run(&c, "a/b", &e).0.expect("fetch");
        let mut other = body.clone();
        other[5] ^= 1;
        fs::write(&p, &other).expect("overwrite");
        fs::remove_file(with_suffix(&p, ".verified")).expect("marker");
        assert_eq!(fs::metadata(&p).expect("cache metadata").len(), e.size);
        let (r, seen) = run(&c, "a/b", &e);
        r.expect("refetch");
        assert_eq!(states(&seen), ["file stale 0"]);
        assert_eq!(fs::read(&p).expect("file"), body);
        assert_eq!(
            fs::read_to_string(with_suffix(&p, ".verified")).expect("new marker"),
            format!("{}\n", e.digest)
        );
        assert!(!with_suffix(&p, ".part").exists());
        let (r, seen) = run(&c, "a/b", &e);
        r.expect("verified cache");
        assert_eq!(states(&seen), [format!("file cached {}", body.len())]);
    }

    /// A cached file longer than the repo's, or a `.part` longer than it, is
    /// not the repo's file: removed and fetched again, never a refusal that
    /// every later start meets.
    #[test]
    fn a_file_longer_than_the_repos_is_fetched_again() {
        let d = scratch("longer");
        let body = content();
        let (c, e) = served(&d, "a/b", "m.gguf", &body);
        let p = run(&c, "a/b", &e).0.expect("fetch");
        let mut longer = body.clone();
        longer.extend_from_slice(b"tail");
        fs::write(&p, &longer).expect("overwrite");
        let (r, seen) = run(&c, "a/b", &e);
        r.expect("refetch");
        assert_eq!(states(&seen), ["file stale 0"]);
        assert_eq!(fs::read(&p).expect("file"), body);
        fs::remove_file(&p).expect("file");
        fs::remove_file(with_suffix(&p, ".verified")).expect("marker");
        fs::write(with_suffix(&p, ".part"), &longer).expect("part");
        let (r, seen) = run(&c, "a/b", &e);
        r.expect("refetch the part");
        assert_eq!(states(&seen), ["file stale 0"]);
        assert_eq!(fs::read(&p).expect("file"), body);
    }

    #[test]
    fn a_range_is_the_files_bytes_at_that_offset() {
        let d = scratch("range");
        let body = content();
        let (c, _) = served(&d, "a/b", "dir/m.gguf", &body);
        let got = c
            .fetch_range("a/b", "main", "dir/m.gguf", 1000, 4096)
            .expect("range");
        assert_eq!(got, body[1000..5096]);
        let last = body.len() as u64 - 7;
        assert_eq!(
            c.fetch_range("a/b", "main", "dir/m.gguf", last, 7)
                .expect("tail"),
            body[body.len() - 7..]
        );
        assert!(
            c.fetch_range("a/b", "main", "dir/m.gguf", 0, 0)
                .expect("none")
                .is_empty()
        );
        // Nothing is cached or marked by a range read.
        assert!(!d.join("cache").join("a/b/dir/m.gguf").exists());
    }

    #[test]
    fn a_range_past_the_end_or_of_a_missing_file_is_a_named_error() {
        let d = scratch("range-bad");
        let body = content();
        let (c, _) = served(&d, "a/b", "m.gguf", &body);
        let at = body.len() as u64 - 10;
        match c.fetch_range("a/b", "main", "m.gguf", at, 100) {
            Err(HfError::Range {
                from, want, got, ..
            }) => assert_eq!((from, want, got), (at, 100, 10)),
            other => panic!("{other:?}"),
        }
        match c.fetch_range("a/b", "main", "absent.gguf", 0, 16) {
            Err(HfError::Curl { what, .. }) => assert!(what.contains("absent.gguf"), "{what}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_revision_names_the_listing_and_the_range() {
        let d = scratch("range-rev");
        let (c, _) = served(&d, "a/b", "m.gguf", b"main bytes");
        let rev = d.join("hub/a/b/resolve/0123abc");
        fs::create_dir_all(&rev).expect("rev dir");
        fs::write(rev.join("m.gguf"), b"older bytes").expect("rev file");
        assert_eq!(
            c.fetch_range("a/b", "0123abc", "m.gguf", 6, 5)
                .expect("range"),
            b"bytes"
        );
        assert_eq!(
            c.fetch_range("a/b", "main", "m.gguf", 5, 5).expect("range"),
            b"bytes"
        );
        let tree = d.join("hub/api/models/a/b/tree");
        fs::create_dir_all(&tree).expect("tree dir");
        let json = |size: u64| {
            format!(
                "[{{\"type\":\"file\",\"path\":\"m.gguf\",\"size\":{size},\"lfs\":{{\"oid\":\"{}\",\"size\":{size}}}}}]",
                "0".repeat(64)
            )
        };
        fs::write(tree.join("main"), json(10)).expect("main listing");
        fs::write(tree.join("0123abc"), json(11)).expect("rev listing");
        assert_eq!(c.listing("a/b").expect("main")[0].size, 10);
        assert_eq!(c.listing_at("a/b", "0123abc").expect("rev")[0].size, 11);
    }

    /// A client's revision is the one its listing and its download read: a pinned revision's file
    /// is fetched, not `main`'s.
    #[test]
    fn a_clients_revision_names_the_listing_and_the_download() {
        let d = scratch("client-rev");
        let (mut c, _) = served(&d, "a/b", "m.bin", b"main bytes");
        let older: &[u8] = b"older bytes";
        let rev = d.join("hub/a/b/resolve/0123abc");
        fs::create_dir_all(&rev).expect("rev dir");
        fs::write(rev.join("m.bin"), older).expect("rev file");
        let tree = d.join("hub/api/models/a/b/tree");
        fs::create_dir_all(&tree).expect("tree dir");
        let listing = |content: &[u8]| {
            format!(
                "[{{\"type\":\"file\",\"path\":\"m.bin\",\"size\":{},\"lfs\":{{\"oid\":\"{}\",\"size\":{}}}}}]",
                content.len(),
                hex(&sha2::Sha256::digest(content)),
                content.len()
            )
        };
        fs::write(tree.join("main"), listing(b"main bytes")).expect("main listing");
        fs::write(tree.join("0123abc"), listing(older)).expect("rev listing");
        c.revision = "0123abc".to_owned();
        let got = c
            .resolve_exact("a/b", &["m.bin"], &mut |_| {})
            .expect("resolve");
        assert_eq!(fs::read(&got[0]).expect("read"), older);
    }

    /// Listings run at once in one process each follow their own `Link` header to their last page:
    /// every first page is held until all have been asked for, so their header files overlap.
    #[test]
    fn concurrent_paginated_listings_each_read_their_own_pages() {
        use std::io::Write as _;
        const REPOS: usize = 8;
        let d = scratch("listing-concurrent");
        let l = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let base = format!("http://127.0.0.1:{}", l.local_addr().expect("addr").port());
        let first_pages = std::sync::Arc::new(std::sync::Barrier::new(REPOS));
        let served = base.clone();
        thread::spawn(move || {
            for c in l.incoming().flatten() {
                let (first_pages, base) = (first_pages.clone(), served.clone());
                thread::spawn(move || {
                    let mut c = c;
                    let mut buf = [0u8; 4096];
                    let n = c.read(&mut buf).expect("request");
                    let request = String::from_utf8_lossy(&buf[..n]);
                    let target = request.split(' ').nth(1).expect("request target");
                    let repo = target.split('/').nth(4).expect("repo");
                    let (page, link) = if target.ends_with("&cursor=2") {
                        (2, String::new())
                    } else {
                        first_pages.wait();
                        (
                            1,
                            format!("Link: <{base}{target}&cursor=2>; rel=\"next\"\r\n"),
                        )
                    };
                    let body = format!(
                        "[{{\"type\":\"file\",\"path\":\"{repo}-{page}.gguf\",\"size\":1,\"lfs\":{{\"oid\":\"{}\",\"size\":1}}}}]",
                        "0".repeat(64)
                    );
                    let _ = write!(
                        c,
                        "HTTP/1.1 200 OK\r\n{link}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                });
            }
        });
        let c = Client {
            base,
            token: None,
            root: d.join("cache"),
            curl: "curl".into(),
            offline: false,
            revision: "main".to_owned(),
        };
        thread::scope(|s| {
            let runs: Vec<_> = (0..REPOS)
                .map(|i| {
                    let c = &c;
                    s.spawn(move || (i, c.listing(&format!("o/r{i}"))))
                })
                .collect();
            for run in runs {
                let (i, got) = run.join().expect("listing thread");
                let paths: Vec<String> =
                    got.expect("listing").into_iter().map(|e| e.path).collect();
                assert_eq!(paths, [format!("r{i}-1.gguf"), format!("r{i}-2.gguf")]);
            }
        });
    }

    /// A page curl answered but whose header file cannot be read is an error, not a last page.
    #[cfg(unix)]
    #[test]
    fn a_listing_page_without_its_headers_is_an_error() {
        use std::os::unix::fs::PermissionsExt;

        let d = scratch("listing-no-headers");
        let script = d.join("curl");
        fs::write(&script, "#!/bin/sh\nprintf '[]'\n").unwrap();
        fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
        let c = Client {
            base: "http://127.0.0.1:1".to_owned(),
            token: None,
            root: d.join("cache"),
            curl: script.into_os_string(),
            offline: false,
            revision: "main".to_owned(),
        };
        let err = c.listing("a/b").expect_err("a page with no headers");
        assert!(
            matches!(&err, HfError::Io { path, .. } if path.ends_with(".headers")),
            "{err}"
        );
    }

    /// The hub answers a request with a redirect whose body is longer than a short range (310
    /// bytes for the 202-byte `generation_config.json` of a checkpoint folder): the limit holds it.
    #[test]
    fn a_short_range_is_given_room_for_a_redirects_body() {
        assert!(range_limit(202) >= 310);
        assert!(range_limit(0) >= 310);
        assert_eq!(range_limit(u64::MAX), u64::MAX);
    }

    #[test]
    fn a_hub_that_ignores_the_range_is_refused_not_read() {
        let d = scratch("range-ignored");
        static WHOLE: [u8; 5000] = [7; 5000];
        // A hub that ignores `Range` answers the whole file, status 200.
        let c = answering_with("200 OK", &WHOLE, &d.join("cache"));
        match c.fetch_range("a/b", "main", "m.gguf", 0, 100) {
            Err(HfError::Curl { why, .. }) => assert!(why.starts_with("exit 63"), "{why}"),
            other => panic!("{other:?}"),
        }
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

    #[test]
    fn an_empty_git_blob_has_the_published_object_id() {
        let d = scratch("git-empty");
        let p = d.join("empty");
        fs::write(&p, b"").expect("file");
        assert_eq!(
            digest(&p, Check::GitSha1, 0).expect("digest"),
            "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
        );
    }

    #[test]
    fn a_git_blob_uses_the_supplied_size_and_reads_the_entire_file() {
        let d = scratch("git-supplied-size");
        let p = d.join("body");
        let body = content();
        fs::write(&p, &body).expect("file");
        for size in [0, 1, body.len() as u64 - 1, body.len() as u64 + 1, u64::MAX] {
            let mut expected = sha1::Sha1::new();
            expected.update(format!("blob {size}\0").as_bytes());
            expected.update(&body);
            assert_eq!(
                digest(&p, Check::GitSha1, size).expect("digest"),
                hex(&expected.finalize()),
                "supplied size {size}"
            );
        }
    }

    #[test]
    fn a_sha256_file_digest_ignores_the_supplied_size() {
        let d = scratch("sha256-size");
        let p = d.join("body");
        let body = content();
        fs::write(&p, &body).expect("file");
        let expected = hex(&sha2::Sha256::digest(&body));
        for size in [0, body.len() as u64, u64::MAX] {
            assert_eq!(digest(&p, Check::Sha256, size).expect("digest"), expected);
        }
    }

    #[test]
    fn a_marked_file_is_not_hashed_again_before_the_fetch() {
        let d = scratch("snapshot-marked");
        let repo = "a/b";
        let (c, marked) = served(&d, repo, "m.bin", b"marked");
        c.fetch(repo, &marked, &mut |_| {}).expect("fetch");
        let (c, bare) = served(&d, repo, "u.bin", b"unmarked");
        c.fetch(repo, &bare, &mut |_| {}).expect("fetch");
        let root = d.join("cache");
        let path = cache_path(&root, repo, &bare.path).expect("path");
        fs::remove_file(with_suffix(&path, ".verified")).expect("marker");
        let left: Vec<String> = unmarked(&root, repo, &[marked, bare])
            .into_iter()
            .map(|e| e.path)
            .collect();
        assert_eq!(left, ["u.bin"]);
    }

    #[test]
    fn concurrent_snapshot_hashing_matches_sequential_hashing() {
        let d = scratch("snapshot-hash");
        let repo = "a/b";
        let root = d.join("cache");
        let mut entries = Vec::new();
        let mut markers = Vec::new();
        for i in 0..6 {
            let name = format!("f{i}.bin");
            let mut body = content();
            body[0] = i;
            let (c, e) = served(&d, repo, &name, &body);
            c.fetch(repo, &e, &mut |_| {}).expect("fetch");
            let path = cache_path(&root, repo, &name).expect("path");
            let marker = with_suffix(&path, ".verified");
            fs::remove_file(&marker).expect("marker");
            assert_eq!(
                digest(&path, e.check, e.size).expect("sequential"),
                e.digest
            );
            entries.push(e);
            markers.push(marker);
        }
        mark_snapshot(&root, repo, &entries).expect("concurrent");
        for (e, marker) in entries.iter().zip(&markers) {
            assert_eq!(
                fs::read_to_string(marker).expect("marker"),
                format!("{}\n", e.digest),
                "{}",
                e.path
            );
        }
    }

    #[test]
    fn a_snapshot_file_that_cannot_be_read_is_an_error_not_a_verdict() {
        let d = scratch("snapshot-unreadable");
        let repo = "a/b";
        let (c, good) = served(&d, repo, "ok.bin", b"ok");
        c.fetch(repo, &good, &mut |_| {}).expect("fetch");
        let root = d.join("cache");
        let marker = with_suffix(
            &cache_path(&root, repo, &good.path).expect("path"),
            ".verified",
        );
        fs::remove_file(&marker).expect("marker");
        let mut gone = good.clone();
        gone.path = "gone.bin".to_owned();
        let r = mark_snapshot(&root, repo, &[gone, good]);
        match r {
            Err(HfError::Io { path, .. }) => assert!(path.contains("gone.bin"), "{path}"),
            other => panic!("{other:?}"),
        }
    }

    /// The two shards of the one GGUF set [`open_set`] resolves.
    const SET_FILES: [&str; 2] = [
        "UD-Q4_K_XL/m-UD-Q4_K_XL-00001-of-00002.gguf",
        "UD-Q4_K_XL/m-UD-Q4_K_XL-00002-of-00002.gguf",
    ];
    /// The files of the checkpoint [`open_checkpoint`] resolves.
    const CHECKPOINT_FILES: [&str; 3] = [
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ];
    /// The files [`open_exact`] asks for.
    const EXACT_FILES: [&str; 2] = ["x.bin", "y.bin"];

    fn open_set(c: &Client, events: &mut dyn FnMut(&Event<'_>)) -> Result<Vec<PathBuf>, HfError> {
        let r = RepoRef::parse("a/b:UD-Q4_K_XL").expect("repo");
        c.resolve(&r, Sources::Gguf, None, events)
    }

    fn open_checkpoint(
        c: &Client,
        events: &mut dyn FnMut(&Event<'_>),
    ) -> Result<Vec<PathBuf>, HfError> {
        let r = RepoRef::parse("a/b").expect("repo");
        c.resolve(&r, Sources::GgufOrCheckpoint, None, events)
    }

    fn open_exact(c: &Client, events: &mut dyn FnMut(&Event<'_>)) -> Result<Vec<PathBuf>, HfError> {
        c.resolve_exact("a/b", &EXACT_FILES, events)
    }

    /// A hub holding `files` of `a/b`, each its own bytes, under `dir`, opened once by `open` so
    /// they are cached and marked; each file's bytes, and each file's cache path.
    fn opened_once(
        dir: &Path,
        files: &[&str],
        open: &impl Fn(&Client, &mut dyn FnMut(&Event<'_>)) -> Result<Vec<PathBuf>, HfError>,
    ) -> (Client, Vec<Vec<u8>>, Vec<PathBuf>) {
        let bodies: Vec<Vec<u8>> = (0..files.len())
            .map(|i| {
                if files[i] == safetensors::Folder::INDEX {
                    return checkpoint_index(files).into_bytes();
                }
                let mut body = content();
                body[0] = i as u8;
                body
            })
            .collect();
        let c = {
            let listed: Vec<(&str, &[u8])> = files
                .iter()
                .copied()
                .zip(bodies.iter().map(Vec::as_slice))
                .collect();
            hub(dir, "a/b", &listed)
        };
        open(&c, &mut |_| {}).expect("first open");
        let paths = files
            .iter()
            .map(|f| dir.join("cache/a/b").join(f))
            .collect();
        (c, bodies, paths)
    }

    fn checkpoint_index(files: &[&str]) -> String {
        let map: serde_json::Map<String, serde_json::Value> = files
            .iter()
            .filter(|f| f.ends_with(".safetensors") && !f.starts_with("extra/"))
            .map(|f| (f.to_string(), serde_json::Value::String(f.to_string())))
            .collect();
        serde_json::json!({"weight_map": map}).to_string()
    }

    /// A cached snapshot whose markers are gone is hashed by the check a resolve runs before its
    /// fetches, which then read none of them: the files' events are `cached` and no `done`.
    fn assert_marked_before_the_fetch(
        name: &str,
        files: &[&str],
        open: impl Fn(&Client, &mut dyn FnMut(&Event<'_>)) -> Result<Vec<PathBuf>, HfError>,
    ) {
        let d = scratch(name);
        let (c, bodies, paths) = opened_once(&d, files, &open);
        for p in &paths {
            fs::remove_file(with_suffix(p, ".verified")).expect("marker");
        }
        let mut seen = Vec::new();
        open(&c, &mut |ev| match ev {
            Event::File { state, .. } => seen.push(state.name()),
            Event::Done { .. } => seen.push("done"),
            _ => {}
        })
        .expect("second open");
        assert_eq!(seen, vec!["cached"; files.len()], "{name}");
        for (p, body) in paths.iter().zip(&bodies) {
            assert_eq!(
                fs::read_to_string(with_suffix(p, ".verified")).expect("marker"),
                format!("{}\n", hex(&sha2::Sha256::digest(body))),
                "{name}"
            );
        }
    }

    /// A cached snapshot with one file of its listed size but other bytes and no marker: the check
    /// a resolve runs before its fetches leaves it alone, and the fetch of that one file removes it
    /// and fetches it again, as a fetch of that file alone does; the resolve gives no error.
    fn assert_corrupt_file_is_fetched_again(
        name: &str,
        files: &[&str],
        open: impl Fn(&Client, &mut dyn FnMut(&Event<'_>)) -> Result<Vec<PathBuf>, HfError>,
    ) {
        let d = scratch(name);
        let (c, bodies, paths) = opened_once(&d, files, &open);
        let last = files.len() - 1;
        let mut other = bodies[last].clone();
        other[5] ^= 1;
        fs::write(&paths[last], &other).expect("overwrite");
        fs::remove_file(with_suffix(&paths[last], ".verified")).expect("marker");
        let mut seen = Vec::new();
        let r = open(&c, &mut |ev| {
            if let Event::File { state, .. } = ev {
                seen.push(state.name());
            }
        });
        r.expect(name);
        seen.sort_unstable();
        let mut want = vec!["cached"; last];
        want.push("stale");
        assert_eq!(seen, want, "{name}");
        assert_eq!(
            fs::read(&paths[last]).expect("file"),
            bodies[last],
            "{name}"
        );
        assert_eq!(
            fs::read_to_string(with_suffix(&paths[last], ".verified")).expect("marker"),
            format!("{}\n", hex(&sha2::Sha256::digest(&bodies[last]))),
            "{name}"
        );
        assert!(!with_suffix(&paths[last], ".part").exists(), "{name}");
    }

    #[test]
    fn a_cached_snapshot_is_marked_before_the_fetch_on_every_resolve_path() {
        assert_marked_before_the_fetch("marked-set", &SET_FILES, open_set);
        assert_marked_before_the_fetch("marked-checkpoint", &CHECKPOINT_FILES, open_checkpoint);
        assert_marked_before_the_fetch("marked-exact", &EXACT_FILES, open_exact);
    }

    #[test]
    fn a_corrupt_same_size_cache_is_fetched_again_by_every_resolve_path() {
        assert_corrupt_file_is_fetched_again("corrupt-set", &SET_FILES, open_set);
        assert_corrupt_file_is_fetched_again(
            "corrupt-checkpoint",
            &CHECKPOINT_FILES,
            open_checkpoint,
        );
        assert_corrupt_file_is_fetched_again("corrupt-exact", &EXACT_FILES, open_exact);
    }

    #[test]
    fn a_missing_digest_file_preserves_its_typed_path_and_io_error() {
        let d = scratch("digest-missing");
        let p = d.join("absent");
        for check in [Check::Sha256, Check::GitSha1] {
            let error = digest(&p, check, 0).expect_err("missing file");
            let HfError::Io { path, err } = &error else {
                panic!("{error:?}");
            };
            assert_eq!(path, &p.display().to_string());
            assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
            let source = std::error::Error::source(&error)
                .expect("I/O source")
                .downcast_ref::<std::io::Error>()
                .expect("original I/O error");
            assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
            assert_eq!(error.to_string(), format!("{}: {source}", p.display()));
            assert!(!source.to_string().contains(path.as_str()));
        }
    }

    /// A client whose hub refuses the connection: curl's exit 7.
    fn unreachable(root: &Path) -> Client {
        Client {
            base: "http://127.0.0.1:1".to_owned(),
            token: None,
            root: root.to_owned(),
            curl: "curl".into(),
            offline: false,
            revision: "main".to_owned(),
        }
    }

    /// A hub on a local port that answers every request with `status`.
    fn answering(status: &'static str, root: &Path) -> Client {
        answering_with(status, b"", root)
    }

    /// A hub on a local port that answers every request with `status` and `body`.
    fn answering_with(status: &'static str, body: &'static [u8], root: &Path) -> Client {
        hub_answering(root, move |_, _| {
            let mut answer = format!(
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .into_bytes();
            answer.extend_from_slice(body);
            answer
        })
    }

    /// A hub on a local port that answers its `n`th request, of text `request`, with the bytes
    /// `answer(n, request)` and closes the connection.
    pub(super) fn hub_answering(
        root: &Path,
        answer: impl Fn(usize, &str) -> Vec<u8> + Send + 'static,
    ) -> Client {
        let l = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = l.local_addr().expect("addr").port();
        std::thread::spawn(move || {
            for (n, c) in l.incoming().flatten().enumerate() {
                let mut c = c;
                let mut buf = [0u8; 4096];
                let got = c.read(&mut buf).unwrap_or(0);
                let _ = c.write_all(&answer(n, &String::from_utf8_lossy(&buf[..got])));
            }
        });
        Client {
            base: format!("http://127.0.0.1:{port}"),
            token: None,
            root: root.to_owned(),
            curl: "curl".into(),
            offline: false,
            revision: "main".to_owned(),
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
        let got = c.resolve(&r, Sources::GgufOrCheckpoint, None, &mut |ev| {
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

    /// `--hf a/b@v1` reads v1's listing and files into its own cache directory, beside main's,
    /// and offline only v1's verified files stand in for it: no revision takes another's set.
    #[test]
    fn a_revision_names_the_listing_the_files_and_the_cache() {
        let d = scratch("revision");
        let file = "m-Q4_K_M.gguf";
        let c = hub(&d, "a/b", &[(file, b"main bytes")]);
        hub_at(
            &d,
            "a/b",
            "v1",
            &[(file, b"v1 bytes"), ("README.md", b"card")],
        );
        let pick = |c: &Client, r: &str| offline_run(c, r).0.expect(r)[0].clone();
        let v1 = pick(&c, "a/b@v1:Q4_K_M");
        assert_eq!(v1, d.join("cache/a/b@v1").join(file));
        let main = pick(&c, "a/b:Q4_K_M");
        assert_eq!(main, d.join("cache/a/b").join(file));
        assert_eq!(pick(&c, "a/b@main:Q4_K_M"), main);
        assert_eq!(fs::read(&v1).expect("v1"), b"v1 bytes");
        assert_eq!(fs::read(&main).expect("main"), b"main bytes");
        let card = c.resolve_exact("a/b@v1", &["README.md"], &mut |_| {});
        assert_eq!(card.expect("card"), [d.join("cache/a/b@v1/README.md")]);
        let off = unreachable(&d.join("cache"));
        assert_eq!(pick(&off, "a/b@v1:Q4_K_M"), v1);
        assert!(matches!(
            offline_run(&off, "a/b@v2:Q4_K_M").0,
            Err(HfError::Offline { .. })
        ));
        let mut pinned = c;
        pinned.revision = "0123abc".to_owned();
        assert_eq!(pinned.hub_at("a/b@v1"), ("a/b", "v1"));
        assert_eq!(pinned.hub_at("a/b"), ("a/b", "0123abc"));
    }

    /// A repo with no GGUF set and a safetensors checkpoint at its top level resolves to the
    /// checkpoint's folder in the cache, its index, shards, configs and template fetched and
    /// nothing else; offline the verified checkpoint stands in. A quant on such a repo is
    /// refused by name, and a repo with neither model is refused naming both.
    #[test]
    fn a_safetensors_repo_resolves_to_its_checkpoint_folder() {
        let d = scratch("checkpoint");
        let files = [
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "config.json",
            "tokenizer.json",
            "chat_template.jinja",
            "README.md",
            "extra/other.safetensors",
        ];
        let body = content();
        let index = checkpoint_index(&files);
        let listed: Vec<(&str, &[u8])> = files
            .iter()
            .map(|f| {
                (
                    *f,
                    if *f == safetensors::Folder::INDEX {
                        index.as_bytes()
                    } else {
                        body.as_slice()
                    },
                )
            })
            .collect();
        let c = hub(&d, "a/b", &listed);
        let (got, seen) = offline_run(&c, "a/b");
        assert_eq!(got.expect("resolve"), [d.join("cache/a/b")], "{seen:?}");
        let fetched: Vec<&str> = files[..6].to_vec();
        for f in &files {
            assert_eq!(
                d.join("cache/a/b").join(f).exists(),
                fetched.contains(f),
                "{f}"
            );
        }
        let (got, seen) = offline_run(&unreachable(&d.join("cache")), "a/b");
        assert_eq!(got.expect("offline"), [d.join("cache/a/b")], "{seen:?}");
        assert_eq!(
            seen.iter()
                .filter(|s| s.ends_with("offline-cached"))
                .count(),
            6
        );
        match offline_run(&c, "a/b:Q4_K_M").0 {
            Err(HfError::CheckpointQuant { quant, .. }) => assert_eq!(quant, "Q4_K_M"),
            other => panic!("{other:?}"),
        }
        let none = hub(&d, "a/c", &[("README.md", b"x")]);
        let e = offline_run(&none, "a/c").0.expect_err("no model");
        assert!(
            matches!(e, HfError::NoGguf { .. }) && e.to_string().contains("safetensors"),
            "{e}"
        );
    }

    #[test]
    fn checkpoint_support_applies_to_online_and_cached_sources() {
        let d = scratch("checkpoint-support");
        let mut c = hub(
            &d,
            "a/b",
            &[
                ("model.safetensors.index.json", br#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#),
                ("model-00001-of-00002.safetensors", b"one"),
                ("model-00002-of-00002.safetensors", b"two"),
            ],
        );
        let r = RepoRef::parse("a/b").unwrap();
        for offline in [false, true] {
            c.offline = offline;
            for single in [false, true] {
                let mut file_events = 0;
                let mut record = |ev: &Event<'_>| {
                    if matches!(ev, Event::File { .. } | Event::Checkpoint { .. }) {
                        file_events += 1;
                    }
                };
                let result = if single {
                    c.resolve(&r, Sources::Gguf, Some(&RUNS), &mut record)
                } else {
                    c.resolve_drafts(&r, Sources::Gguf, &[RUNS], &mut record)
                };
                let error = result.expect_err("checkpoint unsupported");
                let error = match &error {
                    HfError::Offline { cache, .. } => cache.as_ref(),
                    error => error,
                };
                assert!(matches!(error, HfError::CheckpointUnsupported { repo } if repo == "a/b"));
                assert_eq!(file_events, 0);
                if !offline {
                    assert!(!d.join("cache/a/b").exists());
                }
            }
            if !offline {
                c.resolve_drafts(&r, Sources::GgufOrCheckpoint, &[RUNS], &mut |_| {})
                    .expect("supported checkpoint");
            } else {
                assert_eq!(
                    c.resolve_drafts(&r, Sources::GgufOrCheckpoint, &[RUNS], &mut |_| {})
                        .unwrap(),
                    [d.join("cache/a/b")]
                );
            }
        }
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
        for status in ["404 Not Found", "403 Forbidden", "400 Bad Request"] {
            match offline_run(&answering(status, &d.join("cache")), "a/b:Q4_K_M").0 {
                Err(HfError::Curl { why, .. }) => assert!(why.contains(&status[..3]), "{why}"),
                other => panic!("{status}: {other:?}"),
            }
        }
    }

    /// A hub that answers busy (429) or down (5xx) is no listing: the
    /// verified set stands in as it does when the network is gone, and with
    /// none the refusal is the cache's.
    #[test]
    fn a_busy_or_down_hub_leaves_the_verified_set_to_stand_in() {
        let d = scratch("offline-busy");
        cached(&d, "a/b", &["m-Q4_K_M.gguf"]);
        for status in [
            "429 Too Many Requests",
            "500 Internal Server Error",
            "503 Service Unavailable",
        ] {
            let (got, seen) = offline_run(&answering(status, &d.join("cache")), "a/b:Q4_K_M");
            assert_eq!(
                got.expect(status),
                [d.join("cache/a/b/m-Q4_K_M.gguf")],
                "{seen:?}"
            );
            assert!(
                seen[0].starts_with("offline exit 22") && seen[0].contains(&status[..3]),
                "{seen:?}"
            );
            assert_eq!(seen[2], "file m-Q4_K_M.gguf offline-cached");
        }
        match offline_run(
            &answering("502 Bad Gateway", &d.join("cache")),
            "x/y:Q4_K_M",
        )
        .0
        {
            Err(HfError::Offline { why, cache, .. }) => {
                assert!(why.contains("502"), "{why}");
                assert!(matches!(*cache, HfError::NoGguf { .. }), "{cache}");
            }
            other => panic!("{other:?}"),
        }
    }

    /// curl's failures as their kinds: the network's codes and a busy or
    /// down hub let the cache stand in; any other HTTP status, or an exit
    /// that is neither, is the refusal.
    #[test]
    fn a_failure_is_the_networks_the_busy_hubs_or_a_refusal() {
        let said = |s: u16| format!("curl: (22) The requested URL returned error: {s}");
        for s in [429, 500, 502, 503, 504, 599] {
            assert!(hub_unavailable(Some(22), Some(s)), "{s}");
            assert!(
                matches!(
                    failure("l", Some(22), &said(s)),
                    HfError::Unavailable { .. }
                ),
                "{s}"
            );
        }
        for s in [400, 401, 403, 404, 410, 428, 600] {
            assert!(!hub_unavailable(Some(22), Some(s)), "{s}");
            assert!(
                matches!(failure("l", Some(22), &said(s)), HfError::Curl { .. }),
                "{s}"
            );
        }
        assert!(!hub_unavailable(Some(22), None));
        assert!(!hub_unavailable(Some(56), Some(503)));
        assert!(matches!(
            failure("l", Some(7), "refused"),
            HfError::Network { .. }
        ));
        assert!(matches!(
            failure("l", Some(23), "write"),
            HfError::Curl { .. }
        ));
        assert!(matches!(failure("l", None, ""), HfError::Curl { .. }));
    }

    /// `HF_HUB_OFFLINE` as `huggingface_hub` reads it, and nothing else.
    #[test]
    fn hub_offline_reads_the_hubs_words_and_refuses_the_rest() {
        assert!(!hub_offline(None).expect("unset"));
        for v in ["1", "true", "TRUE", "Yes", "on", " 1 "] {
            assert!(hub_offline(Some(v)).expect(v), "{v}");
        }
        for v in ["", "0", "false", "No", "OFF"] {
            assert!(!hub_offline(Some(v)).expect(v), "{v}");
        }
        for v in ["2", "maybe", "offline"] {
            match hub_offline(Some(v)) {
                Err(e @ HfError::HubOffline(_)) => assert!(e.to_string().contains(v), "{e}"),
                other => panic!("{v}: {other:?}"),
            }
        }
    }

    /// Under `HF_HUB_OFFLINE` the hub is not asked: a hub that would answer
    /// with another listing is never read, the verified set stands in, and
    /// with none the refusal names what the cache lacks.
    #[test]
    fn hub_offline_asks_the_hub_nothing() {
        let d = scratch("offline-env");
        cached(&d, "a/b", &["m-Q4_K_M.gguf"]);
        let mut c = hub(&d, "a/b", &[("m-Q8_0.gguf", b"other")]);
        c.offline = true;
        let (got, seen) = offline_run(&c, "a/b:Q4_K_M");
        assert_eq!(got.expect("offline"), [d.join("cache/a/b/m-Q4_K_M.gguf")]);
        assert_eq!(seen[0], format!("offline {HUB_OFFLINE_WHY}"));
        match offline_run(&c, "a/b:Q8_0").0 {
            Err(e @ HfError::Offline { .. }) => {
                let said = e.to_string();
                assert!(
                    said.contains(HUB_OFFLINE_WHY) && said.contains("Q8_0"),
                    "{said}"
                );
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            c.resolve_exact("a/b", &["README.md"], &mut |_| {}),
            Err(HfError::Offline { .. })
        ));
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
            .resolve(&r, Sources::GgufOrCheckpoint, Some(&RUNS), &mut |ev| {
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
        c.resolve(&r, Sources::GgufOrCheckpoint, Some(&RUNS), &mut |ev| {
            if let Event::File { state, .. } = ev {
                again.push(state.name());
            }
        })
        .expect("again");
        assert_eq!(again, ["cached", "cached", "cached"]);
    }

    /// A set that cannot run the repo's draft (the predicate reads its
    /// first shard and says why) leaves the draft unfetched, with one
    /// `DraftSkip` event naming the file and the reason.
    #[test]
    fn a_draft_the_set_cannot_run_is_not_fetched() {
        let d = scratch("draft-unusable");
        let body = content();
        let c = hub(
            &d,
            "a/b",
            &[
                ("UD-Q3_K_XL/m-UD-Q3_K_XL.gguf", &body),
                ("MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf", b"draft"),
            ],
        );
        let first = d.join("cache/a/b/UD-Q3_K_XL/m-UD-Q3_K_XL.gguf");
        let read = std::cell::RefCell::new(Vec::new());
        let usable = |p: &Path| {
            read.borrow_mut().push(p.to_path_buf());
            Err("the target's output.weight is Q6_K".to_owned())
        };
        let draft = Draft {
            name: DRAFT,
            usable: &usable,
        };
        let r = RepoRef::parse("a/b:UD-Q3_K_XL").expect("repo");
        let mut seen = Vec::new();
        let got = c
            .resolve(
                &r,
                Sources::GgufOrCheckpoint,
                Some(&draft),
                &mut |ev| match ev {
                    Event::File { file, .. } => seen.push(format!("file {file}")),
                    Event::DraftSkip { file, why } => seen.push(format!("skip {file}: {why}")),
                    _ => {}
                },
            )
            .expect("resolve");
        assert_eq!(got.as_slice(), std::slice::from_ref(&first));
        assert_eq!(
            read.borrow().as_slice(),
            std::slice::from_ref(&first),
            "the predicate read the first shard"
        );
        assert!(
            !first.with_file_name(DRAFT).exists(),
            "the draft was fetched"
        );
        assert_eq!(
            seen,
            [
                "file UD-Q3_K_XL/m-UD-Q3_K_XL.gguf".to_owned(),
                format!("skip MTP/{DRAFT}: the target's output.weight is Q6_K"),
            ]
        );
    }

    #[test]
    fn a_repo_without_the_drafts_file_fetches_only_the_set() {
        let d = scratch("draft-none");
        let body = content();
        let c = hub(&d, "a/b", &[("m-Q4_K_M.gguf", &body)]);
        let r = RepoRef::parse("a/b:Q4_K_M").expect("repo");
        let mut seen = Vec::new();
        let got = c
            .resolve(&r, Sources::GgufOrCheckpoint, Some(&RUNS), &mut |ev| {
                if let Event::File { file, .. } = ev {
                    seen.push(file.to_string());
                }
            })
            .expect("resolve");
        assert_eq!(got, [d.join("cache/a/b/m-Q4_K_M.gguf")]);
        assert_eq!(seen, ["m-Q4_K_M.gguf"]);
    }

    /// A family with no draft row hands a resolve no draft: the repo's file
    /// of a draft's name is not fetched, and nothing is said of it.
    #[test]
    fn a_family_with_no_draft_row_fetches_no_draft() {
        let d = scratch("draft-no-row");
        let body = content();
        let c = hub(
            &d,
            "a/b",
            &[
                ("m-Q4_K_M.gguf", &body),
                ("MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf", b"draft"),
            ],
        );
        let r = RepoRef::parse("a/b:Q4_K_M").expect("repo");
        let mut seen = Vec::new();
        let got = c
            .resolve_drafts(&r, Sources::Gguf, &[], &mut |ev| match ev {
                Event::File { file, .. } => seen.push(format!("file {file}")),
                Event::DraftSkip { file, .. } => seen.push(format!("skip {file}")),
                _ => {}
            })
            .expect("resolve");
        assert_eq!(got, [d.join("cache/a/b/m-Q4_K_M.gguf")]);
        assert_eq!(seen, ["file m-Q4_K_M.gguf"]);
        assert!(!d.join("cache/a/b/MTP").exists(), "a draft was fetched");
        assert!(
            !got[0].with_file_name(DRAFT).exists(),
            "a draft was fetched"
        );
    }

    /// Each draft of the slice is looked up by its own name: the row the repo
    /// holds no file of fetches nothing, the one it holds lands beside the set.
    #[test]
    fn each_draft_of_the_slice_is_looked_up_by_its_own_name() {
        let d = scratch("draft-rows");
        let body = content();
        let c = hub(
            &d,
            "a/b",
            &[
                ("m-Q4_K_M.gguf", &body),
                (DRAFT, b"draft"),
                ("model.safetensors.index.json", b"index"),
                ("model-00001-of-00001.safetensors", b"checkpoint"),
            ],
        );
        let r = RepoRef::parse("a/b:Q4_K_M").expect("repo");
        let drafts = [
            Draft {
                name: "mtp-another-family.gguf",
                usable: &|_| Err("never asked".to_owned()),
            },
            RUNS,
        ];
        let mut seen = Vec::new();
        let got = c
            .resolve_drafts(&r, Sources::Gguf, &drafts, &mut |ev| match ev {
                Event::File { file, .. } => seen.push(format!("file {file}")),
                Event::DraftSkip { file, .. } => seen.push(format!("skip {file}")),
                _ => {}
            })
            .expect("resolve");
        assert_eq!(
            seen,
            ["file m-Q4_K_M.gguf".to_owned(), format!("file {DRAFT}")]
        );
        assert_eq!(
            fs::read(got[0].with_file_name(DRAFT)).expect("draft"),
            b"draft"
        );
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
        online
            .resolve(&r, Sources::GgufOrCheckpoint, Some(&RUNS), &mut |_| {})
            .expect("fetch");
        assert!(
            d.join("cache/a/b/Q4_K_XL").join(DRAFT).is_file(),
            "the online fetch landed the draft beside the set"
        );
        // The network gone: the set stands in from the cache and the draft
        // is not looked for — the engine finds the fetched file by its name.
        let c = unreachable(&d.join("cache"));
        let mut seen = Vec::new();
        let got = c
            .resolve(&r, Sources::GgufOrCheckpoint, Some(&RUNS), &mut |ev| {
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
        assert!(curl_why(Some(22), err).contains(DENIED_HINT));
        let missing = "curl: (22) The requested URL returned error: 404";
        assert_eq!(http_status(missing), Some(404));
        assert!(curl_why(Some(22), missing).contains(NOT_THERE_HINT));
        assert_eq!(http_status("curl: (6) Could not resolve host"), None);
    }
}
