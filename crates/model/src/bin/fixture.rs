//! fixture — the gate fixture of a model file (`model::fixture`, one spec per family): print its
//! layout, write it, or check a written one against its source.
//!
//!     fixture plan <real first shard> [--draft <real draft>] [flags]
//!     fixture generate <real first shard> <out dir> [--draft <real draft>] [flags]
//!     fixture verify <fixture first shard> [--source <real first shard>]
//!                    [--draft-source <real draft>]
//!
//! The spec is the family's whose architecture the first file declares (for `verify`, the
//! fixture's, which is its source's); an architecture with no spec is refused by name.
//! Flags of `plan` and `generate`: `--seed N`, `--card-budget B` (bytes, or `nM`/`nG` as
//! `BLOOMERY_CARD_BUDGET` takes them; the spec's budget when absent), `--shard-bytes B`
//! (tensor data a shard holds at most), and `--tensors a,b,…` / `--draft-tensors a,b,…` for a
//! file holding only those tensors (it carries `bloomery.fixture.subset`). `plan` writes
//! nothing. `generate` refuses an existing `<out dir>`, writes into `<out dir>.tmp.<pid>` and
//! renames it on success. `verify` takes the source from `--source`, else the spec's default,
//! and checks the spec's draft fixture when it exists, against `--draft-source`, else the
//! spec's default draft. A flag its verb does not take, a flag given twice, `--draft-tensors`
//! without `--draft` and `--draft-source` with no draft fixture are refused. One line per
//! tensor, a summary line, and exit status 1 with the error on any failure.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use gguf::Split;
use model::arch::deepseek41::fixture as v41;
use model::fixture::{
    self, DEFAULT_SEED, DEFAULT_SHARD_BYTES, FilePlan, FixtureSpec, Options, PlannedTensor, Sample,
    TensorStat,
};
use model::placement::card_budget;

const USAGE: &str = "usage: fixture plan <real first shard> [--draft <real draft>] [flags]
       fixture generate <real first shard> <out dir> [--draft <real draft>] [flags]
       fixture verify <fixture first shard> [--source <real first shard>] [--draft-source <real draft>]
flags of plan and generate: --seed N --card-budget B --shard-bytes B --tensors a,b,... --draft-tensors a,b,...";

/// Every family's spec, found by the architecture it declares.
const SPECS: [fn() -> FixtureSpec; 1] = [v41::spec];

/// The flags `plan` and `generate` take.
const PLAN_FLAGS: [&str; 6] = [
    "--draft",
    "--seed",
    "--card-budget",
    "--shard-bytes",
    "--tensors",
    "--draft-tensors",
];
/// The flags `verify` takes.
const VERIFY_FLAGS: [&str; 2] = ["--source", "--draft-source"];

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
        "verify" => &VERIFY_FLAGS,
        _ => return Err(USAGE.into()),
    };
    let a = Args::parse(cmd, takes, rest)?;
    match (cmd.as_str(), a.paths.as_slice()) {
        ("plan", [source]) => plan(source, &a),
        ("generate", [source, out]) => generate(source, out, &a),
        ("verify", [first]) => verify(first, &a),
        _ => Err(USAGE.into()),
    }
}

/// The command line after the verb.
struct Args {
    paths: Vec<String>,
    draft: Option<String>,
    source: Option<String>,
    draft_source: Option<String>,
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
        if a.draft_tensors.is_some() && a.draft.is_none() {
            return Err(format!("--draft-tensors needs --draft\n{USAGE}").into());
        }
        Ok(a)
    }

    /// The plan's options: the flags given, the spec's and the generator's
    /// defaults for the rest.
    fn options(&self, spec: &FixtureSpec) -> Options {
        Options {
            seed: self.seed.unwrap_or(DEFAULT_SEED),
            card_budget: self.card_budget.unwrap_or(spec.card_budget),
            shard_bytes: self.shard_bytes.unwrap_or(DEFAULT_SHARD_BYTES),
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
        opts.card_budget,
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
        opts.card_budget,
        a.draft.as_deref().unwrap_or("none")
    );
    let mut line = |t: &TensorStat| {
        println!(
            "fixture: tensor {} type={} bytes={} file={} gen_secs={:.3} write_secs={:.3}",
            t.name, t.ty, t.bytes, t.file, t.gen_secs, t.write_secs
        );
    };
    let s = fixture::generate(
        &spec,
        &split,
        draft.as_ref(),
        Path::new(out),
        &opts,
        &mut line,
    )?;
    println!(
        "fixture: generate done tensors={} bytes={} file_bytes={} gen_secs={:.2} write_secs={:.2} sync_secs={:.2} secs={:.2} peak_rss_kib={} out={}",
        s.tensors,
        s.bytes,
        s.file_bytes,
        s.gen_secs,
        s.write_secs,
        s.sync_secs,
        s.secs,
        peak_rss_kib(),
        s.out.display()
    );
    Ok(())
}

fn verify(first: &str, a: &Args) -> Res<()> {
    let fx = open(first)?;
    let spec = spec_of(&fx, first)?;
    let source = a.source.clone().unwrap_or_else(spec.default_source);
    let real = open(&source)?;
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
    if !has_draft && let Some(src) = &a.draft_source {
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
            let src = a
                .draft_source
                .clone()
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
    let (t, d) = fixture::verify(&spec, &fx, &real, pair, &mut line)?;
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
    Ok(())
}
