//! The pick of a start by name: the catalog row of a model that holds on this machine
//! ([`hf::catalog::pick`]), asked what only the machine can say.
//!
//! A row's files come from its repo's listing at the row's pinned revision (one listing a repo and
//! revision), less what the cache holds. A row is planned by the server's own `--plan` on the
//! row's header, in a child process of this binary that runs under the user's flags and
//! environment and answers with one JSON line ([`hf::catalog::read_plan`]): a verified cached set
//! is planned from its first shard, any other from a header-only copy of it under the cache's
//! `headers` directory, which costs the header's bytes and not the download. The rows are taken
//! in rank order and a row that cannot beat the plan in hand is neither listed nor read
//! ([`hf::catalog::pick`]). The children run one after another: each reads the card's free bytes,
//! and a sibling's context would be counted in them.
//!
//! A listing the network refuses leaves the cache to stand in, as `--hf` does; any other failure
//! of a listing ends the pick by name. A row whose header cannot be read or whose child gives no
//! answer is rejected as `plan:header` or `plan:probe`, with the cause on stderr.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use hf::catalog::{self, Avail, CatalogRef, Kind, Machine, Outcome, Plan, Refusal, Row};
use hf::fetch::{Client, NOT_THERE_HINT, verified_entries};
use hf::{Entry, HfError};

use crate::GateError;

/// What a pick is asked.
pub struct Ask<'a> {
    /// The model, and the variant the start named, if any.
    pub target: &'a CatalogRef,
    /// The card class and the cache's room.
    pub machine: &'a Machine,
    /// The user's flags, which every probe's `--plan` runs under (the model flags are out).
    pub flags: &'a [String],
    /// `--revision`: every row is read at it instead of its pinned commit.
    pub revision: Option<&'a str>,
}

/// One repo at one revision, as the pick reads it.
struct Repo {
    /// The listing; `None` where the hub could not be reached and the cache stands in.
    listed: Option<Vec<Entry>>,
    /// The cache's verified files.
    cached: Vec<Entry>,
}

/// What the pick has read of the hub and the cache.
struct Reader<'a> {
    client: &'a Client,
    root: &'a Path,
    revision: Option<&'a str>,
    repos: BTreeMap<(String, String), Repo>,
    /// The first failure of a read that has no row to blame; it ends the pick.
    failure: Option<GateError>,
}

impl Reader<'_> {
    /// The revision `row` is read at.
    fn revision_of(&self, row: &Row) -> String {
        self.revision.unwrap_or(row.revision).to_owned()
    }

    /// `row`'s repo at its revision, listed once.
    fn repo(&mut self, row: &Row) -> Result<&Repo, HfError> {
        let key = (row.repo.to_owned(), self.revision_of(row));
        if !self.repos.contains_key(&key) {
            let cached = verified_entries(self.root, row.repo)?;
            let listed = if self.client.offline {
                None
            } else {
                match self.client.listing_at(row.repo, &key.1) {
                    Ok(entries) => Some(entries),
                    Err(HfError::Network { why, .. } | HfError::Unavailable { why, .. }) => {
                        eprintln!(
                            "bloomery-serve: pick: {}: no listing ({why}); the cache stands in",
                            row.repo
                        );
                        None
                    }
                    // A repo or revision the hub does not have lists no file: its rows are not in the repo.
                    Err(HfError::Curl { why, .. }) if why.contains(NOT_THERE_HINT) => {
                        eprintln!(
                            "bloomery-serve: pick: {} at {}: the hub has no such repo or revision",
                            row.repo, key.1
                        );
                        Some(Vec::new())
                    }
                    Err(e) => return Err(e),
                }
            };
            self.repos.insert(key.clone(), Repo { listed, cached });
        }
        Ok(&self.repos[&key])
    }

    /// What the listing and the cache hold of `row`.
    fn avail(&mut self, row: &Row) -> Result<Avail, HfError> {
        let repo = self.repo(row)?;
        match &repo.listed {
            Some(entries) => catalog::availability(row, entries, &repo.cached),
            None => Ok(catalog::availability_offline(row, &repo.cached)),
        }
    }

    /// The file or folder the seat plans `row` from: the cache's own when it holds the whole set,
    /// else a header-only copy.
    fn header(&mut self, row: &Row) -> Result<PathBuf, GateError> {
        let (client, root) = (self.client, self.root);
        let revision = self.revision_of(row);
        let repo = self.repo(row)?;
        // Where no listing could be read the verified files are all there is.
        let entries: &[Entry] = repo.listed.as_deref().unwrap_or(&repo.cached);
        let files = catalog::row_files(row, entries)?
            .ok_or_else(|| format!("{}: the listing lacks {}", row.repo, row.variant))?;
        if files
            .iter()
            .all(|file| catalog::is_cached(file, &repo.cached))
        {
            return match row.kind {
                Kind::Gguf => {
                    let first = files
                        .first()
                        .ok_or_else(|| format!("{}: the set has no file", row.repo))?;
                    Ok(hf::cache_path(root, row.repo, &first.path)?)
                }
                Kind::Folder => Ok(root.join(row.repo)),
            };
        }
        if repo.listed.is_none() {
            return Err(
                format!("{}: no listing, and the cache holds no whole set", row.repo).into(),
            );
        }
        sparse(client, root, row, &revision, &files)
    }

    /// The seat's plan of `row` as a rejection or a plan.
    fn probe(&mut self, row: &'static Row, exe: &Path, flags: &[String]) -> Result<Plan, Refusal> {
        eprintln!(
            "bloomery-serve: pick: planning {}:{}",
            row.repo, row.variant
        );
        let header = match self.header(row) {
            Ok(path) => path,
            Err(e) => {
                eprintln!("bloomery-serve: pick: {}:{}: {e}", row.repo, row.variant);
                return Err(Refusal {
                    code: "header".to_owned(),
                    need_bytes: None,
                });
            }
        };
        run_plan(exe, &header, flags)
    }
}

/// `files` of `row` as a header-only copy under the cache, and the path the seat opens it by.
#[cfg(feature = "checkpoint")]
fn sparse(
    client: &Client,
    root: &Path,
    row: &Row,
    revision: &str,
    files: &[Entry],
) -> Result<PathBuf, GateError> {
    use model_hf::remote;
    let dir = remote::headers_dir(root, row.repo, revision);
    match row.kind {
        Kind::Gguf => {
            let paths = remote::sparse_set(
                client,
                row.repo,
                revision,
                files,
                &dir,
                &mut |_: &Entry, _: &remote::HeaderStat| {},
            )?;
            paths
                .into_iter()
                .next()
                .ok_or_else(|| format!("{}: the set has no file", row.repo).into())
        }
        Kind::Folder => Ok(remote::sparse_checkpoint(
            client,
            row.repo,
            revision,
            files,
            &dir,
            &mut |_: &Entry| {},
        )?),
    }
}

/// A build without the checkpoint feature reads no remote header.
#[cfg(not(feature = "checkpoint"))]
fn sparse(
    _client: &Client,
    _root: &Path,
    row: &Row,
    _revision: &str,
    _files: &[Entry],
) -> Result<PathBuf, GateError> {
    Err(format!(
        "{}: not in the cache, and this build reads no remote header (it has no checkpoint \
         feature); start it with --hf {}",
        row.repo,
        row.hf()
    )
    .into())
}

/// The last non-empty line of a child's stderr.
fn last_line(stderr: &[u8]) -> String {
    String::from_utf8_lossy(stderr)
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("it printed nothing")
        .to_owned()
}

/// `exe -m header <flags> --plan --json`, and its answer: the plan or its refusal; a child that
/// cannot be run or gives no plan line is the refusal `probe`, with the cause on stderr.
fn run_plan(exe: &Path, header: &Path, flags: &[String]) -> Result<Plan, Refusal> {
    let failed = |why: String| {
        eprintln!("bloomery-serve: pick: {why}");
        Refusal {
            code: "probe".to_owned(),
            need_bytes: None,
        }
    };
    let out = Command::new(exe)
        .arg("-m")
        .arg(header)
        .args(flags)
        .args(["--plan", "--json"])
        .env_remove("BLOOMERY_REF_MODEL")
        .stdin(Stdio::null())
        .output()
        .map_err(|e| failed(format!("{}: {e}", exe.display())))?;
    match catalog::read_plan(&String::from_utf8_lossy(&out.stdout)) {
        Ok(answer) => answer,
        Err(why) => Err(failed(format!(
            "the plan of {} ended {} with no answer ({why}): {}",
            header.display(),
            out.status,
            last_line(&out.stderr)
        ))),
    }
}

/// Why a pick found no row, for the class of refusal it ends in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Why {
    /// No row of the model is in its repo.
    NotInRepo,
    /// No listing could be read and the cache holds none of the rows.
    Offline,
    /// The rows are there and the cache's drive cannot take the least of them.
    DiskShort,
    /// The seat's plan refuses the rows that are there, or they are held only slower than
    /// nothing: the model does not fit as asked.
    NoFit,
    /// The seat's plan refused every considered row for one reason that is a class of its own
    /// (`unsupported`, `usage` or `lever`, as the plan's refusal names it): the start ends in that
    /// class, not as a fit refusal.
    PlanRefused(String),
    /// Every row of the model is one a start by name never takes on its own (an explicit-only
    /// format, a row nobody has verified), so none was asked about the machine: it fits or not,
    /// nobody knows. A start that names a variant takes it.
    NamedOnly,
}

/// The class of an outcome with no picked row: the one reason every rejection of a considered row
/// shares when they share one, else [`Why::NoFit`]. A row the pick filtered by policy (an
/// explicit-only format, an unverified row) was never a candidate, so it does not decide the class;
/// a model with no other row is [`Why::NamedOnly`], not a fit refusal.
#[must_use]
pub fn why_none(outcome: &Outcome) -> Why {
    use catalog::Reason;
    let considered: Vec<&Reason> = outcome
        .rejected
        .iter()
        .map(|r| &r.why)
        .filter(|why| !matches!(why, Reason::ExplicitOnly | Reason::Unverified))
        .collect();
    if considered.is_empty() && !outcome.rejected.is_empty() {
        return Why::NamedOnly;
    }
    let all =
        |want: fn(&Reason) -> bool| !considered.is_empty() && considered.iter().all(|r| want(r));
    if all(|why| matches!(why, Reason::NotInRepo)) {
        Why::NotInRepo
    } else if all(|why| matches!(why, Reason::OfflineUncached)) {
        Why::Offline
    } else if all(|why| matches!(why, Reason::DiskShort)) {
        Why::DiskShort
    } else if let Some(code) = shared_class(&considered) {
        Why::PlanRefused(code)
    } else {
        Why::NoFit
    }
}

/// The plan refusal code every one of `whys` carries, when it is one of the classes a start can
/// end in besides a fit refusal.
fn shared_class(whys: &[&catalog::Reason]) -> Option<String> {
    let mut codes = whys.iter().map(|why| match why {
        catalog::Reason::Plan { code, .. } => Some(code.as_str()),
        _ => None,
    });
    let first = codes.next()??;
    (["unsupported", "usage", "lever"].contains(&first) && codes.all(|code| code == Some(first)))
        .then(|| first.to_owned())
}

/// The pick of `ask`: the row that holds, and every other row with its reason
/// ([`hf::catalog::pick`]). A listing that fails for a reason other than the network ends it
/// by name.
pub fn choose(ask: &Ask<'_>) -> Result<Outcome, GateError> {
    let root = super::cache_root()?;
    let client = Client::new(root.clone())?;
    let exe = std::env::current_exe().map_err(|e| format!("the server's own path: {e}"))?;
    let reader = RefCell::new(Reader {
        client: &client,
        root: &root,
        revision: ask.revision,
        repos: BTreeMap::new(),
        failure: None,
    });
    let mut avail = |row: &'static Row| -> Avail {
        let mut read = reader.borrow_mut();
        match read.avail(row) {
            Ok(avail) => avail,
            Err(e) => {
                if read.failure.is_none() {
                    read.failure = Some(e.into());
                }
                Avail::NotInRepo
            }
        }
    };
    let mut probe = |row: &'static Row| -> Result<Plan, Refusal> {
        reader.borrow_mut().probe(row, &exe, ask.flags)
    };
    let outcome = catalog::pick(ask.target, ask.machine, &mut avail, &mut probe);
    match reader.into_inner().failure {
        Some(e) => Err(e),
        None => Ok(outcome),
    }
}

#[cfg(test)]
#[cfg(unix)]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("bloomery-pick-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch");
        dir
    }

    /// An executable shell script `body` named `name` in `dir`.
    fn script(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).expect("script");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        path
    }

    /// The probe's child is the server itself run as `-m <header> <flags> --plan --json`: its
    /// plan line is read whatever record lines share the output, its refusal is the row's
    /// rejection with the bytes it asked, and a child that prints no plan line, or does not run,
    /// is the rejection `probe`.
    #[test]
    fn a_plan_is_asked_of_a_child_run_on_the_rows_header() {
        let d = scratch("child");
        let plan = r#"{"schema":"bloomery.plan/1","seat":"glm","place":"a","place_why":"w","ctx":4096,"ctx_rule":"target","slots":1,"limit":8192,"host_bytes":0,"paged":0,"need":null}"#;
        let ok = script(
            &d,
            "ok",
            &format!(
                "echo 'plan a card=A6000'\necho '{plan}'\n# $1 $2 are -m and the header\necho \"$@\" >&2"
            ),
        );
        let got = run_plan(
            &ok,
            Path::new("/h/m.gguf"),
            &["--ctx-size".to_owned(), "99".to_owned()],
        );
        assert_eq!(
            got,
            Ok(Plan {
                place: "a".to_owned(),
                ctx: 4096,
                slots: 1,
                host_bytes: 0,
                paged_bytes: 0,
                fit: catalog::Fit::Planned,
            })
        );
        let refused = script(
            &d,
            "refused",
            r#"echo '{"schema":"bloomery.plan/1","refused":{"code":"no-fit","message":"m","need":5}}'; exit 80"#,
        );
        assert_eq!(
            run_plan(&refused, Path::new("/h/m.gguf"), &[]),
            Err(Refusal {
                code: "no-fit".to_owned(),
                need_bytes: Some(5)
            })
        );
        let silent = script(&d, "silent", "echo boom >&2; exit 3");
        let got = run_plan(&silent, Path::new("/h/m.gguf"), &[]).expect_err("no line");
        assert_eq!((got.code.as_str(), got.need_bytes), ("probe", None));
        let missing = run_plan(&d.join("absent"), Path::new("/h/m.gguf"), &[]).expect_err("absent");
        assert_eq!(missing.code, "probe");
        // The child is told the header and the user's flags, then the plan flags.
        let args = script(&d, "args", r#"echo "$@" > "$0.args"; exit 1"#);
        let _ = run_plan(
            &args,
            Path::new("/h/m.gguf"),
            &["--parallel".to_owned(), "2".to_owned()],
        );
        assert_eq!(
            std::fs::read_to_string(d.join("args.args"))
                .expect("args")
                .trim(),
            "-m /h/m.gguf --parallel 2 --plan --json"
        );
        std::fs::remove_dir_all(d).expect("clean");
    }

    /// A pick with no row ends in the class its rejections share: every row missing from the
    /// repo, every row uncached offline, every row too big for the drive, else a fit refusal.
    #[test]
    fn a_pick_with_no_row_ends_in_the_class_its_rejections_share() {
        use catalog::{Reason, Rejected};
        let model = catalog::find("qwen3.8-flash-next").expect("model");
        let rejected = |whys: Vec<Reason>| Outcome {
            picked: None,
            rejected: whys
                .into_iter()
                .zip(model.rows)
                .map(|(why, row)| Rejected { row, why })
                .collect(),
        };
        let plan = |code: &str| Reason::Plan {
            code: code.to_owned(),
            need_bytes: Some(1),
        };
        assert_eq!(why_none(&rejected(vec![])), Why::NoFit);
        assert_eq!(
            why_none(&rejected(vec![Reason::NotInRepo, Reason::NotInRepo])),
            Why::NotInRepo
        );
        assert_eq!(
            why_none(&rejected(vec![
                Reason::OfflineUncached,
                Reason::OfflineUncached
            ])),
            Why::Offline
        );
        assert_eq!(
            why_none(&rejected(vec![Reason::DiskShort, Reason::DiskShort])),
            Why::DiskShort
        );
        assert_eq!(
            why_none(&rejected(vec![Reason::DiskShort, plan("no-fit")])),
            Why::NoFit,
            "a mixed set of reasons is a fit refusal"
        );
        assert_eq!(why_none(&rejected(vec![plan("no-fit")])), Why::NoFit);
        assert_eq!(
            why_none(&rejected(vec![plan("unsupported"), plan("unsupported")])),
            Why::PlanRefused("unsupported".to_owned()),
            "rows the seat refuses as unsupported end as that class"
        );
        assert_eq!(
            why_none(&rejected(vec![plan("unsupported"), plan("no-fit")])),
            Why::NoFit,
            "a mixed set of refusals is a fit refusal"
        );
        assert_eq!(
            why_none(&rejected(vec![plan("refused"), plan("refused")])),
            Why::NoFit,
            "an unnamed refusal is no class of its own"
        );
        assert_eq!(
            why_none(&rejected(vec![
                Reason::OfflineUncached,
                Reason::ExplicitOnly,
                Reason::Unverified
            ])),
            Why::Offline,
            "a row the pick never took does not decide the class"
        );
        assert_eq!(
            why_none(&rejected(vec![Reason::ExplicitOnly, Reason::Unverified])),
            Why::NamedOnly,
            "rows none of which was a candidate were not asked, so they did not fail to fit"
        );
    }

    /// A row whose whole set the cache holds is planned from the cache's first shard, with no
    /// request to the hub; the bytes still to fetch are 0 and a row the listing lacks is
    /// not in the repo.
    #[test]
    fn a_cached_set_is_planned_from_the_cache() {
        let d = scratch("cached");
        let row = catalog::find("qwen3")
            .expect("model")
            .row("Q4_K_M")
            .expect("row");
        let file = "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf";
        let digest = "ab".repeat(32);
        let at = d.join(row.repo).join(file);
        std::fs::create_dir_all(at.parent().expect("parent")).expect("repo dir");
        std::fs::write(&at, b"0123456789").expect("model");
        std::fs::write(d.join(row.repo).join(format!("{file}.verified")), &digest).expect("marker");
        let listed = vec![Entry {
            path: file.to_owned(),
            size: 10,
            check: hf::Check::Sha256,
            digest,
        }];
        let client = Client {
            base: "http://hub.invalid".to_owned(),
            token: None,
            root: d.clone(),
            curl: "curl-is-never-run".into(),
            offline: false,
            revision: "main".to_owned(),
        };
        let mut reader = Reader {
            client: &client,
            root: &d,
            revision: None,
            repos: BTreeMap::new(),
            failure: None,
        };
        reader.repos.insert(
            (row.repo.to_owned(), row.revision.to_owned()),
            Repo {
                listed: Some(listed),
                cached: verified_entries(&d, row.repo).expect("verified"),
            },
        );
        assert_eq!(
            reader.avail(row).expect("avail"),
            Avail::Present { download_bytes: 0 }
        );
        assert_eq!(reader.header(row).expect("header"), at);
        let q8 = catalog::find("qwen3")
            .expect("model")
            .row("Q8_0")
            .expect("row");
        assert_eq!(reader.avail(q8).expect("avail"), Avail::NotInRepo);
        std::fs::remove_dir_all(d).expect("clean");
    }
}
