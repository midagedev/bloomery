//! fixture — the gate fixture of a model file (`model::fixture`, one spec per family): print its
//! layout, write it, or check a written one against its source; and the variants of a written
//! fixture (`model::fixture::variant`, the table `tools/fixture-variants.tsv`).
//!
//!     fixture plan <real first shard> [--draft <real draft>] [flags]
//!     fixture generate <real first shard> <out dir> [--draft <real draft>] [flags]
//!     fixture fetch <family | repo[@rev][:quant]> <out dir> [--rev R] [flags]
//!     fixture cut <source folder> <out dir> --spec FILE [--shard-bytes B] [--card-budget B]
//!     fixture verify <fixture first shard> [--source <real first shard>]
//!                    [--draft-source <real draft>]
//!     fixture verify <cut dir | its first shard> [--source <source folder>]
//!     fixture variant-rules <fixture first shard> --tag T [--table FILE]
//!     fixture variant-check <variant first shard> --source <fixture first shard> --tag T
//!                           [--table FILE]
//!     fixture variant-budget <variant first shard>
//!     fixture registry
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
//! fetchable, and `fetch` then skips it by name). A family name (`v41`, `qwen3moe`, `qwen38`, `qwen35moe`, `glm5next`, `solar2`, `nemotron`, `nemotron-bolt`, `musespark`, `mistrallarge4`)
//! reads its row of the registry (`model::arch::registry`): repo, quant and revision; a `<repo>:<quant>` takes `main`, a `<repo>@<rev>:<quant>` its `rev`.
//! After writing, it runs `verify`'s check against the sparse sources it still holds, and a
//! fixture that fails it is removed with its sidecar, so a failed fetch leaves nothing.
//! It takes `generate`'s flags but `--draft`, and `--rev R` (a commit hash, tag or branch of the
//! repo; overrides the row's, and is refused beside an `@<rev>`): a source rewritten upstream has another header, so its fixture
//! is the one of the revision it was made from. `verify` takes the source from `--source`, else the spec's default, and checks
//! the spec's draft fixture when it exists, against `--draft-source`, else the spec's default
//! draft, and a whole fixture's sidecar. `variant-rules` prints what the quantizer is run with to
//! make variant T of the fixture: a `custom-q<TAB>` line (its `--custom-q` argument, which names
//! every tensor) and an `override-kv<TAB>` line (the tag's stamp), after the map's expansion over
//! the fixture; it refuses a row that retypes nothing. `variant-check` holds a written variant to
//! its source: the same header but for the tag and the card budget, the same shards and shapes, the
//! map's types, and every other tensor's bytes equal — the budget a planned family's variant holds
//! is the planner's value for the written file, a fixed family's the trunk's. `variant-budget`
//! records that value on a written variant, in place (the planner's cap for the variant's own
//! file, replacing the trunk's the quantizer copied; a fixed family's is kept as it is). The table
//! is `--table`, else `tools/fixture-variants.tsv`
//! under the working directory. `cut` is the fixture of a safetensors checkpoint folder in its stored
//! bits (`cut::cut`): the layers, widths and tensor roles its spec names (`tools/fixture-cuts/<name>.json`),
//! the tokenizer and chat files copied, and `bloomery-cut.tsv` recording each kept tensor's source range
//! and hash; it refuses an existing `<out dir>` and writes beside it, renaming on success. `verify` of a
//! directory (or its first shard) holds such a cut to its record and to its source folder, the one the
//! record names unless `--source` says another (`cut::verify`). `registry` prints the architecture registry's rows (the text of
//! `tools/arch-registry.tsv`). A flag its verb does not take, a flag given twice,
//! `--draft-tensors` without `--draft` and `--draft-source` with no draft fixture are refused. One
//! line per tensor, a summary line, and exit status 1 with the error on any failure.

#[path = "../cut.rs"]
mod cut;

use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use gguf::Split;
use hf::fetch::Client;
use hf::{Entry, RepoRef};
use model::arch::registry::{self, DraftSource};
use model::fixture::{
    self, DEFAULT_SEED, FilePlan, FixtureError, FixtureSpec, Options, PlannedTensor, Sample,
    TensorStat, VARIANT_TABLE,
};
use model::placement::card_budget;
use model_hf::remote::{self, HeaderStat, RemoteError};

const USAGE: &str = "usage: fixture plan <real first shard> [--draft <real draft>] [flags]
       fixture generate <real first shard> <out dir> [--draft <real draft>] [flags]
       fixture fetch <family | repo[@rev][:quant]> <out dir> [--rev R] [flags]
       fixture cut <source folder> <out dir> --spec FILE [--shard-bytes B] [--card-budget B]
       fixture verify <fixture first shard> [--source <real first shard>] [--draft-source <real draft>]
       fixture verify <cut dir | its first shard> [--source <source folder>]
       fixture variant-rules <fixture first shard> --tag T [--table FILE]
       fixture variant-check <variant first shard> --source <fixture first shard> --tag T [--table FILE]
       fixture variant-budget <variant first shard>
       fixture registry
flags of plan and generate: --seed N --card-budget B --shard-bytes B --tensors a,b,... --draft-tensors a,b,...
flags of fetch: the same but --draft, and --rev R";

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
/// The flags `cut` takes.
const CUT_FLAGS: [&str; 3] = ["--spec", "--shard-bytes", "--card-budget"];
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
        "cut" => &CUT_FLAGS,
        "verify" => &VERIFY_FLAGS,
        "variant-rules" => &RULES_FLAGS,
        "variant-check" => &CHECK_FLAGS,
        "variant-budget" => &[],
        "registry" => &[],
        _ => return Err(USAGE.into()),
    };
    let a = Args::parse(cmd, takes, rest)?;
    match (cmd.as_str(), a.paths.as_slice()) {
        ("plan", [source]) => plan(source, &a),
        ("generate", [source, out]) => generate(source, out, &a),
        ("fetch", [repo, out]) => fetch(repo, out, &a),
        ("cut", [source, out]) => cut_folder(source, out, &a),
        ("verify", [first]) => verify(first, &a),
        ("variant-rules", [source]) => variant_rules(source, &a),
        ("variant-check", [variant]) => variant_check(variant, &a),
        ("variant-budget", [variant]) => variant_budget(variant),
        ("registry", []) => {
            print!("{}", registry::print());
            Ok(())
        }
        _ => Err(USAGE.into()),
    }
}

/// The command line after the verb.
struct Args {
    paths: Vec<String>,
    draft: Option<String>,
    source: Option<String>,
    draft_source: Option<String>,
    spec: Option<String>,
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
            spec: None,
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
                "--spec" => a.spec = Some(value()?),
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
    arch.and_then(registry::fixture_spec).ok_or_else(|| {
        let known: Vec<&str> = registry::FIXTURE_SPECS
            .iter()
            .map(|spec| spec().arch)
            .collect();
        format!("{path}: architecture {arch:?} has no fixture spec (the specs: {known:?})").into()
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

/// Peak resident set of this process, in KiB; -1 when the platform refuses
/// the reading.
fn peak_rss_kib() -> i64 {
    os::max_rss_bytes().map_or(-1, |b| i64::try_from(b / 1024).expect("a peak fits an i64"))
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

/// `generate` from the source `repo` (`<owner>/<name>[@<rev>][:<quant>]`) names, its headers read by
/// range. The sources live in `<out>.fetch.<pid>`, removed whether or not the run succeeds.
fn fetch(target: &str, out: &str, a: &Args) -> Res<()> {
    if let Some(row) = registry::cuts().find(|c| c.family == target) {
        return Err(format!(
            "{target} is a cut folder, not a GGUF fixture: `fixture cut <source folder> {out} \
             --spec tools/fixture-cuts/{}.json`",
            row.family
        )
        .into());
    }
    let (r, rev) = match registry::fixtures().find(|s| s.family == target) {
        Some(s) => (
            RepoRef {
                repo: s.repo.to_string(),
                rev: None,
                quant: Some(s.quant.to_string()),
            },
            a.rev.as_deref().or(s.rev).unwrap_or("main").to_owned(),
        ),
        None => {
            let mut r = RepoRef::parse(target)?;
            // The parse drops an `@main`, the default; it still names a revision.
            let at = r
                .rev
                .take()
                .or_else(|| target.contains('@').then(|| "main".into()));
            let rev = match (at, &a.rev) {
                (Some(at), Some(flag)) => {
                    return Err(format!(
                        "{target}: the revision is named twice, @{at} and --rev {flag}"
                    )
                    .into());
                }
                (at, flag) => flag.clone().or(at).unwrap_or_else(|| "main".to_owned()),
            };
            (r, rev)
        }
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
    let done = fetch_into(&r, &rev, out, a, &tmp);
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
    let first = remote::sparse_set(&client, &r.repo, rev, &set.files, &dir, &mut note)?
        .swap_remove(0)
        .to_string_lossy()
        .into_owned();
    let split = open(&first)?;
    println!(
        "fixture: fetch source header sha256={}",
        fixture::header_sha256(&split)
    );
    let spec = spec_of(&split, &first)?;
    let row = registry::fixture_of(spec.arch)
        .ok_or_else(|| format!("{first}: the {} family has no fixture row", spec.arch))?;
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
                let p =
                    remote::sparse_set(&client, &r.repo, rev, one, &dir, &mut note)?.swap_remove(0);
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
            match remote::remove_fixture(Path::new(out), sidecar.as_deref()) {
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
    let path = Path::new(first);
    if path.is_dir() {
        return verify_folder(path, a);
    }
    if first.ends_with(".safetensors") {
        return verify_folder(path.parent().unwrap_or(Path::new(".")), a);
    }
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

/// `cut::cut` of the folder `source` as `--spec` says into `out`, a line per tensor and the summary.
fn cut_folder(source: &str, out: &str, a: &Args) -> Res<()> {
    let spec_path = a.spec.as_deref().ok_or("cut needs --spec FILE")?;
    let text = std::fs::read_to_string(spec_path).map_err(|e| format!("read {spec_path}: {e}"))?;
    let spec = cut::Spec::parse(&text)?;
    let opts = cut::Options {
        shard_bytes: a.shard_bytes.unwrap_or(cut::DEFAULT_SHARD_BYTES),
        extra: a
            .card_budget
            .map(|b| vec![(model::fixture::KEY_CARD_BUDGET.to_string(), b.to_string())])
            .unwrap_or_default(),
    };
    println!(
        "fixture: cut {source} -> {out} spec={} arch={} ({spec_path}) shard_bytes={}",
        spec.name, spec.arch, opts.shard_bytes
    );
    let mut line = |t: &cut::Written| {
        println!(
            "fixture: cut tensor {} <- {} bytes={} file={} sha256={}",
            t.name, t.src_name, t.bytes, t.shard, t.sha256
        );
    };
    let c = cut::cut(Path::new(source), Path::new(out), &spec, &opts, &mut line)?;
    for f in &c.copied {
        println!("fixture: cut copied {f}");
    }
    for f in &c.left_out {
        println!("fixture: cut left out {f}");
    }
    println!(
        "fixture: cut done tensors={} bytes={} read_bytes={} shards={} dropped_tensors={} \
         dropped_bytes={} manifest_sha256={} secs={:.2} peak_rss_kib={} out={}",
        c.tensors,
        c.data_bytes,
        c.read_bytes,
        c.shards,
        c.dropped_tensors,
        c.dropped_bytes,
        c.manifest_sha256,
        c.secs,
        peak_rss_kib(),
        c.out.display()
    );
    Ok(())
}

/// `cut::verify` of the cut folder `dir` against its record and its source (`--source`, else the
/// folder the record names), a line per tensor and the summary.
fn verify_folder(dir: &Path, a: &Args) -> Res<()> {
    if a.draft_source.is_some() {
        return Err("--draft-source names a draft, and a cut folder has none".into());
    }
    let source = a.source.as_deref().map(Path::new);
    println!(
        "fixture: verify cut {} against {}",
        dir.display(),
        source.map_or("the folder its record names".to_string(), |s| s
            .display()
            .to_string())
    );
    let mut line = |t: &cut::Checked| {
        println!(
            "fixture: verify tensor {} bytes={} sha256={}",
            t.name, t.bytes, t.sha256
        );
    };
    let v = cut::verify(dir, source, &mut line)?;
    println!(
        "fixture: verify done tensors={} bytes={} read_bytes={} shards={} copied={} secs={:.2} \
         peak_rss_kib={}",
        v.tensors,
        v.data_bytes,
        v.read_bytes,
        v.shards,
        v.copied,
        v.secs,
        peak_rss_kib()
    );
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
    let spec = spec_of(&src, source)?;
    let mut line = |l: &str| println!("{l}");
    let s = fixture::variant_check(&row, &src, &var, spec.card_budget, &mut line)?;
    println!(
        "fixture: variant-check done tag={} tensors={} retyped={} copied_bytes={} shards={}",
        row.tag, s.tensors, s.retyped, s.copied_bytes, s.shards
    );
    Ok(())
}

/// Record on the written `variant` the card budget its own file plans, in place: the family's
/// planner chooses the cap for the variant's headers (the tag names the file a refusal speaks of),
/// replacing the trunk's the quantizer copied; a family with a fixed budget keeps the trunk's.
fn variant_budget(variant: &str) -> Res<()> {
    let var = open(variant)?;
    let tag = var
        .value(model::fixture::KEY_VARIANT)
        .and_then(gguf::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| {
            format!(
                "{variant} carries no {}: a budget is recorded on a variant",
                model::fixture::KEY_VARIANT
            )
        })?;
    let spec = spec_of(&var, variant)?;
    match spec.card_budget {
        model::fixture::CardBudget::Fixed(b) => {
            println!(
                "fixture: variant-budget {tag}: the {} family records a fixed budget; the trunk's \
                 {b} is kept",
                spec.arch
            );
        }
        model::fixture::CardBudget::Planned(planner) => {
            let planned = fixture::choose_budget_written(&planner, &var)?;
            let was = fixture::variant_record_budget(&tag, &var, planned)?;
            let key = model::fixture::KEY_CARD_BUDGET;
            let own = if fixture::budget_host_only(&planner, &var)? {
                "host-only: no layer holds a card expert, the trunk's budget is kept"
            } else {
                "planned on the variant's own file"
            };
            println!("fixture: variant-budget {tag}: {key} {was} -> {planned} ({own})");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_spec_has_one_source_row_and_v41s_draft_is_not_fetchable() {
        for spec in registry::FIXTURE_SPECS.iter().map(|f| f()) {
            assert_eq!(registry::fixture_spec(spec.arch).unwrap().arch, spec.arch);
            let row = registry::fixture_of(spec.arch)
                .unwrap_or_else(|| panic!("{}: no fixture row", spec.arch));
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
        // One spec to one row: each spec found its row above, the counts agree, and no two specs
        // declare one architecture.
        assert!(registry::fixture_spec("no-such-fixture").is_none());
        assert_eq!(registry::fixtures().count(), registry::FIXTURE_SPECS.len());
        let archs: std::collections::HashSet<_> =
            registry::FIXTURE_SPECS.iter().map(|f| f().arch).collect();
        assert_eq!(
            archs.len(),
            registry::FIXTURE_SPECS.len(),
            "two specs declare one architecture"
        );
        let find = |family: &str| {
            registry::fixtures()
                .find(|s| s.family == family)
                .unwrap_or_else(|| panic!("no {family} row"))
        };
        assert!(matches!(find("v41").draft, DraftSource::NotFetchable(_)));
        assert_eq!(find("glm5next").rev, Some("621d456e93"));
        let q36 = find("qwen35moe");
        assert_eq!(
            (q36.repo, q36.quant, q36.rev),
            (
                "lmstudio-community/Qwen3.6-35B-A3B-GGUF",
                "Q4_K_M",
                Some("68a34855558af61cbef0324d31f411be8a506b08")
            )
        );
    }

    /// A cut row or profile cut is its rules file's family: the file parses, names the cut's
    /// directory and its architecture string, and `fetch` of a cut family says `cut` instead of
    /// reading a hub repo. (The registry's own test holds a rules file to each cut and each cut to
    /// a file.)
    #[test]
    fn a_cut_row_is_its_rules_files_family_and_fetch_refuses_it() {
        let mut cuts = 0;
        let rows = registry::ROWS.iter().filter_map(|row| match row.fixture {
            Some(registry::Fixture::Cut(c)) => Some((c, row.canonical())),
            _ => None,
        });
        let profiles = registry::PROFILE_CUTS.iter().map(|p| (p.cut, p.arch));
        for (c, arch) in rows.chain(profiles) {
            cuts += 1;
            let path = format!(
                "{}/../../tools/fixture-cuts/{}.json",
                env!("CARGO_MANIFEST_DIR"),
                c.family
            );
            let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
            let spec = cut::Spec::parse(&text).unwrap_or_else(|e| panic!("{path}: {e}"));
            assert_eq!(spec.name, c.family, "{path}: the rules name another family");
            assert_eq!(
                spec.arch, arch,
                "{path}: the rules name another architecture"
            );
            let args = ["fetch", c.family, "/nonexistent/out"].map(String::from);
            let refused = run(&args)
                .expect_err("a cut family is not fetched")
                .to_string();
            assert!(
                refused.contains("is a cut folder") && refused.contains("fixture cut"),
                "{refused}"
            );
        }
        assert_eq!(cuts, registry::cuts().count());
        assert!(cuts > 0, "no cut row to hold");
    }

    /// A repo's `@<rev>` beside `--rev`, `@main` included, is refused before any fetch.
    #[test]
    fn fetch_refuses_a_revision_named_twice() {
        for target in ["a/b@v2:Q4_K_M", "a/b@main:Q4_K_M", "a/b@main"] {
            let args = ["fetch", target, "/nonexistent/out", "--rev", "v1"].map(String::from);
            let refused = run(&args).expect_err("two revisions").to_string();
            assert!(refused.contains("the revision is named twice"), "{refused}");
        }
    }

    /// `cut` and `verify` take their folders and flags from the command line: a toy folder cut with
    /// a spec file and a card budget, then verified by its directory (the source its record names),
    /// by its first shard and with `--source`; a missing `--spec`, a flag `cut` does not take and a
    /// draft source for a folder are each refused by name.
    #[test]
    fn the_command_line_cuts_and_verifies_a_folder() {
        let w = cut::tests::world("cli");
        let spec = w.root.join("spec.json");
        std::fs::write(&spec, cut::tests::spec_json(&w.all, 32, 32, 16)).unwrap();
        let (src, out) = (
            w.src.display().to_string(),
            w.root.join("out").display().to_string(),
        );
        let spec = spec.display().to_string();
        let args = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let refused = |v: &[&str]| run(&args(v)).expect_err("a refused command").to_string();

        run(&args(&[
            "cut",
            &src,
            &out,
            "--spec",
            &spec,
            "--card-budget",
            "8589934592",
        ]))
        .unwrap();
        run(&args(&["verify", &out])).unwrap();
        let first = format!("{out}/model-00001-of-00001.safetensors");
        run(&args(&["verify", &first, "--source", &src])).unwrap();
        let meta = cut::read_metadata(Path::new(&first)).unwrap();
        assert_eq!(
            meta.get("bloomery.fixture.card_budget").map(String::as_str),
            Some("8589934592"),
            "--card-budget reaches the first shard's metadata"
        );

        let other = w.root.join("other").display().to_string();
        assert!(refused(&["cut", &src, &other]).contains("cut needs --spec"));
        assert!(
            refused(&["cut", &src, &other, "--spec", &spec, "--seed", "1"])
                .contains("cut does not take --seed")
        );
        assert!(
            refused(&["verify", &out, "--draft-source", "x"]).contains("a cut folder has none")
        );
        std::fs::remove_dir_all(&w.root).unwrap();
    }

    /// Every written fixture of a family with a planned budget holds the budget its header records:
    /// the plan of the file under it is half the experts on every card-eligible layer
    /// ([`fixture::check_budget`], the re-plan `verify` runs, here from the written header alone, so
    /// it needs neither the source nor the bytes). A plan term that moves after a fixture was
    /// written (a table the card holds, a reserve, a rounding) leaves its recorded number a granule
    /// short, and every gate that plans under the header sees a layer one expert off. The fixture
    /// root is `refset::fixture::root`'s (`BLOOMERY_FIXTURE_ROOT`, else `/models/fixtures`); a family with no directory there
    /// is named and skipped, a directory with no first shard or no budget key fails.
    #[test]
    #[ignore = "reads the written fixtures under the fixture root: gate-fixture-budget"]
    fn hw_the_written_fixtures_hold_their_recorded_budget() {
        use model::fixture::{CardBudget, KEY_CARD_BUDGET};

        let root = PathBuf::from(refset::fixture::root().expect("the fixture root"));
        let (mut held, mut bad) = (Vec::new(), Vec::new());
        for spec in registry::FIXTURE_SPECS.iter().map(|f| f()) {
            let row = registry::fixture_of(spec.arch).expect("a fixture row");
            let CardBudget::Planned(planner) = spec.card_budget else {
                println!("{}: a fixed budget, nothing planned to check", row.family);
                continue;
            };
            let dir = root.join(row.family);
            if !dir.is_dir() {
                println!("{}: no fixture directory {}", row.family, dir.display());
                continue;
            }
            let checked = first_shard(&dir, spec.stem)
                .and_then(|path| open(&path.to_string_lossy()))
                .and_then(|split| {
                    let budget = split
                        .value(KEY_CARD_BUDGET)
                        .and_then(gguf::Value::as_u64)
                        .ok_or(format!("the header holds no u64 {KEY_CARD_BUDGET}"))?;
                    let counts = fixture::check_budget(&planner, &split, budget)?;
                    Ok((budget, counts.per_layer))
                });
            match checked {
                Ok((budget, per_layer)) => {
                    println!(
                        "{}: budget {budget}, experts a layer {per_layer:?}",
                        row.family
                    );
                    held.push(row.family);
                }
                Err(e) => bad.push(format!("{}: {e}", row.family)),
            }
        }
        assert!(
            bad.is_empty(),
            "stale recorded budgets:\n{}",
            bad.join("\n")
        );
        assert!(
            !held.is_empty(),
            "no fixture under {} was checked",
            root.display()
        );
    }
}
