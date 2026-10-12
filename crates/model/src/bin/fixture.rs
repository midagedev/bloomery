//! fixture — the gate fixture of a model file (`model::fixture`, one spec per family): print its
//! layout, write it, or check a written one against its source; and the variants of a written
//! fixture (`model::fixture::variant`, the table `tools/fixture-variants.tsv`).
//!
//!     fixture plan <real first shard> [--draft <real draft>] [flags]
//!     fixture generate <real first shard> <out dir> [--draft <real draft>] [flags]
//!     fixture fetch <family | repo[:quant]> <out dir> [--rev R] [flags]
//!     fixture verify <fixture first shard> [--source <real first shard>]
//!                    [--draft-source <real draft>]
//!     fixture variant-rules <fixture first shard> --tag T [--table FILE]
//!     fixture variant-check <variant first shard> --source <fixture first shard> --tag T
//!                           [--table FILE]
//!
//! The spec is the family's whose architecture the first file declares (for `verify`, the
//! fixture's, which is its source's); an architecture with no spec is refused by name.
//! Flags of `plan` and `generate`: `--seed N`, `--card-budget B` (bytes, or `nM`/`nG` as
//! `BLOOMERY_CARD_BUDGET` takes them; the spec's budget when absent, a fixed one or the
//! cap its family's plan of the written file chooses), `--shard-bytes B` (tensor data a shard
//! holds at most; the spec's cap when absent), and `--tensors a,b,…` / `--draft-tensors a,b,…`
//! for a file holding only those tensors (it carries `bloomery.fixture.subset`). `plan` writes
//! nothing. `generate` refuses an existing `<out dir>`, writes into `<out dir>.tmp.<pid>` and
//! renames it on success; a whole fixture of a family with an r8 sidecar then gets it, beside the
//! directory (`r8file::sidecar_path`), and a failed sidecar takes the directory down. `fetch` is
//! `generate` with the real files read from a Hugging Face repo by HTTP Range, for a machine that
//! has none: the fixture's bytes are a function of the seed and the source's headers
//! (`model::fixture`), so each shard of the set the quant names becomes a sparse file of its
//! listed size with its header (fetched through the hf crate's client) at the front, in a
//! directory beside `<out dir>` that is removed afterwards; the family's draft is the repo's
//! file its family's source row names, when the row has one (a row can say its draft is not
//! fetchable, and `fetch` then skips it by name). A family name (`v41`, `qwen38`, `qwen35moe`, `glm5next`)
//! reads its row of [`SOURCES`]: repo, quant and revision; a `<repo>:<quant>` takes `main`.
//! After writing, it runs `verify`'s check against the sparse sources it still holds, and a
//! fixture that fails it is removed with its sidecar, so a failed fetch leaves nothing.
//! It takes `generate`'s flags but `--draft`, and `--rev R` (a commit hash, tag or branch of the
//! repo; overrides the row's): a source rewritten upstream has another header, so its fixture
//! is the one of the revision it was made from. `verify` takes the source from `--source`, else the spec's default, and checks
//! the spec's draft fixture when it exists, against `--draft-source`, else the spec's default
//! draft, and a whole fixture's sidecar. `variant-rules` prints what the quantizer is run with to
//! make variant T of the fixture: a `custom-q<TAB>` line (its `--custom-q` argument, which names
//! every tensor) and an `override-kv<TAB>` line (the tag's stamp), after the map's expansion over
//! the fixture; it refuses a row that retypes nothing. `variant-check` holds a written variant to
//! its source: the same header but for the tag, the same shards and shapes, the map's types, and
//! every other tensor's bytes equal. The table is `--table`, else `tools/fixture-variants.tsv`
//! under the working directory. A flag its verb does not take, a flag given twice,
//! `--draft-tensors` without `--draft` and `--draft-source` with no draft fixture are refused. One
//! line per tensor, a summary line, and exit status 1 with the error on any failure.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use gguf::Split;
use hf::fetch::Client;
use hf::{Entry, RepoRef};
use model::arch::deepseek41::fixture as v41;
use model::arch::glm5next::fixture as glm5;
use model::arch::qwen35moe::fixture as qwen38;
use model::arch::qwen35moe::fixture::qwen35moe as qwen36;
use model::fixture::{
    self, DEFAULT_SEED, FilePlan, FixtureError, FixtureSpec, HeaderStat, Options, PlannedTensor,
    RemoteError, Sample, TensorStat, VARIANT_TABLE,
};
use model::placement::card_budget;

const USAGE: &str = "usage: fixture plan <real first shard> [--draft <real draft>] [flags]
       fixture generate <real first shard> <out dir> [--draft <real draft>] [flags]
       fixture fetch <family | repo[:quant]> <out dir> [--rev R] [flags]
       fixture verify <fixture first shard> [--source <real first shard>] [--draft-source <real draft>]
       fixture variant-rules <fixture first shard> --tag T [--table FILE]
       fixture variant-check <variant first shard> --source <fixture first shard> --tag T [--table FILE]
flags of plan and generate: --seed N --card-budget B --shard-bytes B --tensors a,b,... --draft-tensors a,b,...
flags of fetch: the same but --draft, and --rev R";

/// Every family's spec, found by the architecture it declares.
const SPECS: [fn() -> FixtureSpec; 4] = [v41::spec, qwen38::spec, qwen36::spec, glm5::spec];

/// Where a family's draft fixture comes from on the hub.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DraftSource {
    /// The family has no draft file.
    None,
    /// The draft is this file name in the target's repo.
    File(&'static str),
    /// No public file is the draft's source; the reason, said when `fetch` skips it.
    NotFetchable(&'static str),
}

/// The hub source a family's fixture is made from: `fixture fetch <family> <out dir>` reads
/// the row, and an explicit `<repo>:<quant>` overrides it.
struct Source {
    spec: fn() -> FixtureSpec,
    /// The name `fetch` takes: the family's directory under the fixture root.
    family: &'static str,
    repo: &'static str,
    quant: &'static str,
    /// The repo revision the fixture's source header is the header of (`main` when `None`).
    rev: Option<&'static str>,
    draft: DraftSource,
}

/// One row per entry of [`SPECS`].
const SOURCES: [Source; 4] = [
    Source {
        spec: v41::spec,
        family: "v41",
        repo: "vcruz305/DeepSeek-V4.1-Flash-GGUF",
        quant: "Q3_K_M",
        rev: None,
        draft: DraftSource::NotFetchable(
            "the draft fixture is made from our own DSpark copy, and no public file has its header",
        ),
    },
    Source {
        spec: qwen38::spec,
        family: "qwen38",
        repo: "unsloth/Qwen3.8-Flash-Next-GGUF",
        quant: "UD-Q4_K_XL",
        rev: None,
        draft: DraftSource::File("mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"),
    },
    Source {
        spec: qwen36::spec,
        family: "qwen35moe",
        repo: "lmstudio-community/Qwen3.6-35B-A3B-GGUF",
        quant: "Q4_K_M",
        // The commit whose header the spec's map was chosen from, pinned so a rewritten upstream
        // file is a new fixture on purpose.
        rev: Some("68a34855558af61cbef0324d31f411be8a506b08"),
        draft: DraftSource::None,
    },
    Source {
        spec: glm5::spec,
        family: "glm5next",
        repo: "unsloth/GLM-5.3-Flash-GGUF",
        quant: "UD-Q4_K_XL",
        // unsloth's later rewrite of shard 1 renamed `glm5next` to `glm5-next`; this is the
        // commit before, whose header the fixture was made from.
        rev: Some("621d456e93"),
        draft: DraftSource::None,
    },
];

/// The flags `plan` and `generate` take.
const PLAN_FLAGS: [&str; 6] = [
    "--draft",
    "--seed",
    "--card-budget",
    "--shard-bytes",
    "--tensors",
    "--draft-tensors",
];
/// The flags `fetch` takes: `generate`'s but the local draft, which the repo's own file replaces.
const FETCH_FLAGS: [&str; 6] = [
    "--rev",
    "--seed",
    "--card-budget",
    "--shard-bytes",
    "--tensors",
    "--draft-tensors",
];
/// The flags `verify` takes.
const VERIFY_FLAGS: [&str; 2] = ["--source", "--draft-source"];
/// The flags `variant-rules` takes.
const RULES_FLAGS: [&str; 2] = ["--tag", "--table"];
/// The flags `variant-check` takes.
const CHECK_FLAGS: [&str; 3] = ["--source", "--tag", "--table"];

type Res<T> = Result<T, Box<dyn Error>>;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("fixture: error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Res<()> {
    let Some((cmd, rest)) = args.split_first() else {
        return Err(USAGE.into());
    };
    let takes: &[&str] = match cmd.as_str() {
        "plan" | "generate" => &PLAN_FLAGS,
        "fetch" => &FETCH_FLAGS,
        "verify" => &VERIFY_FLAGS,
        "variant-rules" => &RULES_FLAGS,
        "variant-check" => &CHECK_FLAGS,
        _ => return Err(USAGE.into()),
    };
    let a = Args::parse(cmd, takes, rest)?;
    match (cmd.as_str(), a.paths.as_slice()) {
        ("plan", [source]) => plan(source, &a),
        ("generate", [source, out]) => generate(source, out, &a),
        ("fetch", [repo, out]) => fetch(repo, out, &a),
        ("verify", [first]) => verify(first, &a),
        ("variant-rules", [source]) => variant_rules(source, &a),
        ("variant-check", [variant]) => variant_check(variant, &a),
        _ => Err(USAGE.into()),
    }
}

/// The command line after the verb.
struct Args {
    paths: Vec<String>,
    draft: Option<String>,
    source: Option<String>,
    draft_source: Option<String>,
    tag: Option<String>,
    table: Option<String>,
    rev: Option<String>,
    seed: Option<u64>,
    card_budget: Option<u64>,
    shard_bytes: Option<u64>,
    tensors: Option<Vec<String>>,
    draft_tensors: Option<Vec<String>>,
}

impl Args {
    /// `args` after verb `cmd`, which takes the flags `takes`.
    fn parse(cmd: &str, takes: &[&str], args: &[String]) -> Res<Args> {
        let mut a = Args {
            paths: Vec::new(),
            draft: None,
            source: None,
            draft_source: None,
            tag: None,
            table: None,
            rev: None,
            seed: None,
            card_budget: None,
            shard_bytes: None,
            tensors: None,
            draft_tensors: None,
        };
        let mut seen: Vec<&str> = Vec::new();
        let mut it = args.iter();
        while let Some(arg) = it.next() {
            if arg.starts_with("--") {
                if seen.contains(&arg.as_str()) {
                    return Err(format!("{arg} is given twice\n{USAGE}").into());
                }
                seen.push(arg);
            }
            let mut value = || {
                it.next()
                    .cloned()
                    .ok_or_else(|| format!("{arg} needs a value\n{USAGE}"))
            };
            let list = |v: String| v.split(',').map(str::to_string).collect::<Vec<_>>();
            match arg.as_str() {
                "--draft" => a.draft = Some(value()?),
                "--source" => a.source = Some(value()?),
                "--draft-source" => a.draft_source = Some(value()?),
                "--tag" => a.tag = Some(value()?),
                "--table" => a.table = Some(value()?),
                "--rev" => a.rev = Some(value()?),
                "--seed" => {
                    let v = value()?;
                    a.seed = Some(
                        v.parse()
                            .map_err(|_| format!("--seed {v:?} is not a u64"))?,
                    );
                }
                "--card-budget" => a.card_budget = Some(card_budget::parse(&value()?)?),
                "--shard-bytes" => a.shard_bytes = Some(card_budget::parse(&value()?)?),
                "--tensors" => a.tensors = Some(list(value()?)),
                "--draft-tensors" => a.draft_tensors = Some(list(value()?)),
                flag if flag.starts_with("--") => {
                    return Err(format!("unknown flag {flag}\n{USAGE}").into());
                }
                _ => a.paths.push(arg.clone()),
            }
        }
        if let Some(f) = seen.iter().find(|f| !takes.contains(*f)) {
            return Err(format!("{cmd} does not take {f}\n{USAGE}").into());
        }
        if cmd != "fetch" && a.draft_tensors.is_some() && a.draft.is_none() {
            return Err(format!("--draft-tensors needs --draft\n{USAGE}").into());
        }
        Ok(a)
    }

    /// The plan's options: the flags given, the generator's and `spec`'s
    /// defaults for the rest (`--card-budget` absent leaves the budget to the
    /// spec's, `--shard-bytes` its cap).
    fn options(&self, spec: &FixtureSpec) -> Options {
        Options {
            seed: self.seed.unwrap_or(DEFAULT_SEED),
            card_budget: self.card_budget,
            shard_bytes: self.shard_bytes.unwrap_or(spec.shard_bytes),
            tensors: self.tensors.clone(),
            draft_tensors: self.draft_tensors.clone(),
        }
    }
}

fn open(path: &str) -> Res<Split> {
    Split::open(path).map_err(|e| format!("open {path}: {e}").into())
}

/// The spec of the family whose architecture `split` (opened from `path`)
/// declares.
fn spec_of(split: &Split, path: &str) -> Res<FixtureSpec> {
    let arch = split.architecture();
    SPECS
        .iter()
        .map(|spec| spec())
        .find(|spec| Some(spec.arch) == arch)
        .ok_or_else(|| {
            let known: Vec<&str> = SPECS.iter().map(|spec| spec().arch).collect();
            format!("{path}: architecture {arch:?} has no fixture spec (the specs: {known:?})")
                .into()
        })
}

/// One file set's lines: per shard, per layer, and the parts' sum checked
/// against the files' total. Returns the total.
fn print_files(what: &str, p: &FilePlan) -> Res<u64> {
    let layouts = p.layouts()?;
    let mut total = 0u64;
    let mut headers = 0u64;
    for (i, ((name, l), range)) in layouts.iter().zip(&p.shards).enumerate() {
        println!(
            "fixture: {what} shard {}/{} {name} tensors={} data_base={} file_len={}",
            i + 1,
            layouts.len(),
            range.len(),
            l.data_base(),
            l.file_len()
        );
        total += l.file_len();
        headers += l.data_base();
    }
    let mut parts = headers;
    let layers = p
        .tensors
        .iter()
        .filter_map(|t| t.layer)
        .max()
        .map_or(0, |m| m + 1);
    for f in 0..layers {
        let ts: Vec<&PlannedTensor> = p.tensors.iter().filter(|t| t.layer == Some(f)).collect();
        let bytes: u64 = ts.iter().map(|t| p.padded(t)).sum();
        let source = ts.first().map_or(String::new(), |t| t.source.clone());
        println!(
            "fixture: {what} layer {f} <- {} tensors={} bytes={bytes}",
            source.split('.').take(2).collect::<Vec<_>>().join("."),
            ts.len()
        );
        parts += bytes;
    }
    let rest: Vec<&PlannedTensor> = p.tensors.iter().filter(|t| t.layer.is_none()).collect();
    let rest_bytes: u64 = rest.iter().map(|t| p.padded(t)).sum();
    println!(
        "fixture: {what} unlayered tensors={} bytes={rest_bytes}",
        rest.len()
    );
    parts += rest_bytes;
    if parts != total {
        return Err(
            format!("{what}: headers + tensors = {parts}, the files' total {total}").into(),
        );
    }
    println!(
        "fixture: {what} total file_len={total} = headers {headers} + tensors {}",
        total - headers
    );
    Ok(total)
}

fn plan(source: &str, a: &Args) -> Res<()> {
    let split = open(source)?;
    let spec = spec_of(&split, source)?;
    let opts = a.options(&spec);
    let draft = a.draft.as_deref().map(open).transpose()?;
    let p = fixture::plan(&spec, &split, draft.as_ref(), &opts)?;
    println!(
        "fixture: plan source={source} ({} shards) seed={} card_budget={} shard_bytes={} window={}",
        split.shard_count(),
        opts.seed,
        p.card_budget,
        opts.shard_bytes,
        spec.window
    );
    let mut seen: Vec<(String, u64)> = Vec::new();
    for t in p
        .target
        .tensors
        .iter()
        .chain(p.draft.iter().flat_map(|d| d.tensors.iter()))
    {
        let key = (t.ty.to_string(), t.dims[0]);
        if t.dims.len() >= 2 && !seen.contains(&key) {
            println!(
                "fixture: rule {} K={} {}",
                t.ty,
                t.dims[0],
                t.rule.describe()
            );
            seen.push(key);
        }
    }
    for line in &p.notes {
        println!("fixture: {line}");
    }
    println!("fixture: spanning layers {:?}", p.target.spanning_layers());
    let target = print_files("target", &p.target)?;
    println!("fixture: target file_len={target}");
    if let Some(d) = &p.draft {
        let draft = print_files("draft", d)?;
        println!(
            "fixture: draft file_len={draft}; target + draft {}",
            target + draft
        );
    }
    Ok(())
}

/// Peak resident set of this process, in KiB.
fn peak_rss_kib() -> i64 {
    let mut ru = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: `ru` is storage for one `rusage`, which the call fills on success.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, ru.as_mut_ptr()) } != 0 {
        return -1;
    }
    // SAFETY: the call returned 0, so it wrote every field of `ru`.
    unsafe { ru.assume_init() }.ru_maxrss
}

fn generate(source: &str, out: &str, a: &Args) -> Res<()> {
    let split = open(source)?;
    let spec = spec_of(&split, source)?;
    let opts = a.options(&spec);
    let draft = a.draft.as_deref().map(open).transpose()?;
    println!(
        "fixture: generate {source} -> {out} seed={} card_budget={} draft={}",
        opts.seed,
        opts.card_budget
            .map_or("planned".to_string(), |b| b.to_string()),
        a.draft.as_deref().unwrap_or("none")
    );
    write_fixture(&spec, &split, draft.as_ref(), out, &opts)
}

/// `spec`'s fixture of `split` (and `draft`) into `out`, a line per tensor and the summary.
fn write_fixture(
    spec: &FixtureSpec,
    split: &Split,
    draft: Option<&Split>,
    out: &str,
    opts: &Options,
) -> Res<()> {
    let mut line = |t: &TensorStat| {
        println!(
            "fixture: tensor {} type={} bytes={} file={} gen_secs={:.3} write_secs={:.3}",
            t.name, t.ty, t.bytes, t.file, t.gen_secs, t.write_secs
        );
    };
    let s = fixture::generate(spec, split, draft, Path::new(out), opts, &mut line)?;
    println!(
        "fixture: generate done tensors={} bytes={} file_bytes={} card_budget={} gen_secs={:.2} \
         write_secs={:.2} sync_secs={:.2} secs={:.2} peak_rss_kib={} out={}",
        s.tensors,
        s.bytes,
        s.file_bytes,
        s.card_budget,
        s.gen_secs,
        s.write_secs,
        s.sync_secs,
        s.secs,
        peak_rss_kib(),
        s.out.display()
    );
    if let Some(sc) = &s.sidecar {
        println!(
            "fixture: sidecar {} tensors={} file_bytes={} secs={:.2}",
            sc.path.display(),
            sc.tensors,
            sc.bytes,
            sc.secs
        );
    }
    Ok(())
}

/// `generate` from the source `repo` (`<owner>/<name>[:<quant>]`) names, its headers read by
/// range. The sources live in `<out>.fetch.<pid>`, removed whether or not the run succeeds.
fn fetch(target: &str, out: &str, a: &Args) -> Res<()> {
    let (r, rev) = match SOURCES.iter().find(|s| s.family == target) {
        Some(s) => (
            RepoRef {
                repo: s.repo.to_string(),
                quant: Some(s.quant.to_string()),
            },
            a.rev.as_deref().or(s.rev).unwrap_or("main"),
        ),
        None => (RepoRef::parse(target)?, a.rev.as_deref().unwrap_or("main")),
    };
    let out_path = Path::new(out);
    if out_path
        .try_exists()
        .map_err(|e| format!("stat {out}: {e}"))?
    {
        return Err(FixtureError::Exists {
            path: out_path.to_path_buf(),
        }
        .into());
    }
    let mut tmp = out_path.as_os_str().to_owned();
    tmp.push(format!(".fetch.{}", std::process::id()));
    let tmp = PathBuf::from(tmp);
    std::fs::create_dir(&tmp).map_err(|e| format!("create {}: {e}", tmp.display()))?;
    let done = fetch_into(&r, rev, out, a, &tmp);
    match (done, std::fs::remove_dir_all(&tmp)) {
        (done, Ok(())) => done,
        (Ok(()), Err(e)) => Err(format!("remove {}: {e}", tmp.display()).into()),
        (Err(e), Err(c)) => Err(format!("{e}; and removing {} failed: {c}", tmp.display()).into()),
    }
}

/// The body of [`fetch`], the sparse sources under `tmp`.
fn fetch_into(r: &RepoRef, rev: &str, out: &str, a: &Args, tmp: &Path) -> Res<()> {
    let client = Client::new(tmp.to_path_buf())?;
    if client.offline {
        return Err(RemoteError::Offline.into());
    }
    let entries = client.listing_at(&r.repo, rev)?;
    let sets = hf::gguf_sets(&r.repo, &entries)?;
    let set = hf::pick(&r.repo, &sets, r.quant.as_deref())?;
    println!(
        "fixture: fetch {r}@{rev} -> {out} set={} shards={} bytes={}",
        set.name,
        set.files.len(),
        set.bytes()
    );
    let dir = tmp.join("source");
    std::fs::create_dir(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let mut note = |e: &Entry, s: &HeaderStat| {
        println!(
            "fixture: fetch {} size={} header={} reads={}",
            e.path, e.size, s.header, s.reads
        );
    };
    let first = fixture::sparse_set(&client, &r.repo, rev, &set.files, &dir, &mut note)?
        .swap_remove(0)
        .to_string_lossy()
        .into_owned();
    let split = open(&first)?;
    println!(
        "fixture: fetch source header sha256={}",
        fixture::header_sha256(&split)
    );
    let spec = spec_of(&split, &first)?;
    let row = SOURCES
        .iter()
        .find(|s| (s.spec)().arch == spec.arch)
        .ok_or_else(|| format!("{first}: the {} family has no row in SOURCES", spec.arch))?;
    let mut draft_path: Option<String> = None;
    let draft = match row.draft {
        DraftSource::None => None,
        DraftSource::NotFetchable(why) => {
            println!("fixture: fetch draft skipped, not fetchable: {why}");
            None
        }
        DraftSource::File(name) => match hf::mtp_draft(&r.repo, &entries, name)? {
            Some(e) => {
                let one = std::slice::from_ref(e);
                let p = fixture::sparse_set(&client, &r.repo, rev, one, &dir, &mut note)?
                    .swap_remove(0);
                println!("fixture: fetch draft {}", e.path);
                draft_path = Some(p.to_string_lossy().into_owned());
                Some(open(&p.to_string_lossy())?)
            }
            None => {
                return Err(format!(
                    "{}@{rev} holds no {name}, the draft of the {} family",
                    r.repo, row.family
                )
                .into());
            }
        },
    };
    if a.draft_tensors.is_some() && draft.is_none() {
        return Err(format!(
            "--draft-tensors needs a draft, and the {} family has none to fetch",
            row.family
        )
        .into());
    }
    let opts = a.options(&spec);
    println!(
        "fixture: generate {} -> {out} seed={} card_budget={}",
        r.repo,
        opts.seed,
        opts.card_budget
            .map_or("planned".to_string(), |b| b.to_string()),
    );
    write_fixture(&spec, &split, draft.as_ref(), out, &opts)?;
    // The written fixture is held to the sources still on disk, as `verify` would hold it;
    // one that is not leaves nothing behind.
    let checked = first_shard(Path::new(out), spec.stem).and_then(|fx_first| {
        let first_str = fx_first.to_string_lossy().into_owned();
        let fx = open(&first_str)?;
        verify_fixture(&first_str, &fx, &spec, &first, draft_path.as_deref())?;
        Ok(fx_first)
    });
    match checked {
        Ok(_) => Ok(()),
        Err(e) => {
            let sidecar = match (&spec.sidecar, first_shard(Path::new(out), spec.stem)) {
                (Some(_), Ok(f)) => fixture::sidecar_path(&f).ok(),
                _ => None,
            };
            match fixture::remove_fixture(Path::new(out), sidecar.as_deref()) {
                Ok(()) => Err(e),
                Err(c) => Err(format!("{e}; and removing the fixture failed: {c}").into()),
            }
        }
    }
}

/// The first shard of the fixture `<stem>-00001-of-NNNNN.gguf` in `dir`.
fn first_shard(dir: &Path, stem: &str) -> Res<PathBuf> {
    let prefix = format!("{stem}-00001-of-");
    let mut hits = Vec::new();
    for e in std::fs::read_dir(dir).map_err(|e| format!("read {}: {e}", dir.display()))? {
        let e = e.map_err(|e| format!("read {}: {e}", dir.display()))?;
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with(&prefix) && name.ends_with(".gguf") {
            hits.push(e.path());
        }
    }
    match hits.len() {
        1 => Ok(hits.remove(0)),
        n => Err(format!(
            "{}: {n} files named {prefix}*.gguf, expected one",
            dir.display()
        )
        .into()),
    }
}

fn verify(first: &str, a: &Args) -> Res<()> {
    let fx = open(first)?;
    let spec = spec_of(&fx, first)?;
    let source = a.source.clone().unwrap_or_else(spec.default_source);
    verify_fixture(first, &fx, &spec, &source, a.draft_source.as_deref())
}

/// The fixture `fx` (opened from `first`) held to `source`, and its draft fixture, when it has
/// one, to `draft_source`, else the spec's default draft.
fn verify_fixture(
    first: &str,
    fx: &Split,
    spec: &FixtureSpec,
    source: &str,
    draft_source: Option<&str>,
) -> Res<()> {
    let real = open(source)?;
    let dir = Path::new(first)
        .parent()
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
    let draft_path = spec.draft.map(|ds| dir.join(ds.file));
    let has_draft = match &draft_path {
        Some(p) => p
            .try_exists()
            .map_err(|e| format!("stat {}: {e}", p.display()))?,
        None => false,
    };
    if !has_draft && let Some(src) = draft_source {
        return Err(match &draft_path {
            Some(p) => format!(
                "--draft-source {src} names a draft source, but {} does not exist",
                p.display()
            ),
            None => format!(
                "--draft-source {src} names a draft source, and the {} fixture has no draft file",
                spec.arch
            ),
        }
        .into());
    }
    let draft = match (&draft_path, spec.draft) {
        (Some(p), Some(ds)) if has_draft => {
            let src = draft_source
                .map(str::to_string)
                .or_else(ds.default_source)
                .ok_or_else(|| {
                    format!(
                        "{} exists; give --draft-source or the family's default draft",
                        p.display()
                    )
                })?;
            Some((open(&p.to_string_lossy())?, open(&src)?))
        }
        _ => None,
    };
    println!(
        "fixture: verify {first} ({} shards) against {source}; draft {}",
        fx.shard_count(),
        match (&draft_path, &draft) {
            (Some(p), Some(_)) => p.display().to_string(),
            _ => "none".into(),
        }
    );
    let mut line = |t: &PlannedTensor, s: &Sample| {
        let sigma = t.sigma().map_or("-".into(), |x| format!("{x:.4e}"));
        println!(
            "fixture: verify {} type={} blocks={} rms={:.4e} sigma={sigma}",
            t.name,
            t.ty,
            s.blocks,
            s.rms()
        );
    };
    let pair = draft.as_ref().map(|(d, r)| (d, r));
    let (t, d) = fixture::verify(spec, fx, &real, pair, &mut line)?;
    println!(
        "fixture: verify done tensors={} blocks={} subset={} draft_tensors={}",
        t.tensors,
        t.blocks,
        t.subset,
        d.map_or("none".into(), |d| format!(
            "{} blocks={} subset={}",
            d.tensors, d.blocks, d.subset
        ))
    );
    if let Some(sc) = &t.sidecar {
        println!(
            "fixture: verify sidecar {} tensors={} bytes={} secs={:.2}",
            sc.path.display(),
            sc.tensors,
            sc.bytes,
            sc.secs
        );
    }
    Ok(())
}

/// The row of `--tag` in the table.
fn variant_row(a: &Args) -> Res<fixture::VariantRow> {
    let tag = a.tag.as_deref().ok_or("--tag is required")?;
    let table = a.table.as_deref().unwrap_or(VARIANT_TABLE);
    let text = std::fs::read_to_string(table).map_err(|e| format!("read {table}: {e}"))?;
    Ok(fixture::variant_row(&fixture::variant_parse_table(&text)?, tag)?.clone())
}

fn variant_rules(source: &str, a: &Args) -> Res<()> {
    let row = variant_row(a)?;
    let split = open(source)?;
    let retypes = fixture::variant_expand(&row, &split)?;
    for r in &retypes {
        eprintln!(
            "fixture: variant {} {} {} <- {}",
            row.tag, r.name, r.to, r.from
        );
    }
    eprintln!(
        "fixture: variant {} retypes {} of {} tensors of {source}",
        row.tag,
        retypes.len(),
        split.tensor_count()
    );
    println!(
        "custom-q\t{}",
        fixture::variant_custom_q(&row.tag, &split, &retypes)?
    );
    println!("override-kv\t{}", fixture::variant_override_kv(&row.tag));
    Ok(())
}

fn variant_check(variant: &str, a: &Args) -> Res<()> {
    let row = variant_row(a)?;
    let source = a.source.as_deref().ok_or("variant-check needs --source")?;
    let (src, var) = (open(source)?, open(variant)?);
    let mut line = |l: &str| println!("{l}");
    let s = fixture::variant_check(&row, &src, &var, &mut line)?;
    println!(
        "fixture: variant-check done tag={} tensors={} retyped={} copied_bytes={} shards={}",
        row.tag, s.tensors, s.retyped, s.copied_bytes, s.shards
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_spec_has_one_source_row_and_v41s_draft_is_not_fetchable() {
        for spec in SPECS.iter().map(|f| f()) {
            let rows: Vec<&Source> = SOURCES
                .iter()
                .filter(|s| (s.spec)().arch == spec.arch)
                .collect();
            assert_eq!(rows.len(), 1, "{}: {} source rows", spec.arch, rows.len());
            let row = rows[0];
            assert!(RepoRef::parse(&format!("{}:{}", row.repo, row.quant)).is_ok());
            // A family with a draft fixture says where the draft comes from, and one without
            // says there is none.
            assert_eq!(
                spec.draft.is_none(),
                row.draft == DraftSource::None,
                "{}: the draft row and the spec disagree",
                spec.arch
            );
        }
        assert_eq!(SOURCES.len(), SPECS.len());
        let names: std::collections::HashSet<_> = SOURCES.iter().map(|s| s.family).collect();
        assert_eq!(names.len(), SOURCES.len(), "family names are unique");
        let v41 = SOURCES.iter().find(|s| s.family == "v41").expect("v41 row");
        assert!(matches!(v41.draft, DraftSource::NotFetchable(_)));
        let glm = SOURCES
            .iter()
            .find(|s| s.family == "glm5next")
            .expect("glm row");
        assert_eq!(glm.rev, Some("621d456e93"));
        let q36 = SOURCES
            .iter()
            .find(|s| s.family == "qwen35moe")
            .expect("qwen35moe row");
        assert_eq!(
            (q36.repo, q36.quant, q36.rev),
            (
                "lmstudio-community/Qwen3.6-35B-A3B-GGUF",
                "Q4_K_M",
                Some("68a34855558af61cbef0324d31f411be8a506b08")
            )
        );
    }
}
