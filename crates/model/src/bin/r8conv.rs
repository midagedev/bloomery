//! r8conv — the r8 sidecar of a V4.1 file (`model::r8file`): write it, check
//! it against its source, or print where it goes.
//!
//!     r8conv convert <first shard> [<out>] [--tensors <name>,<name>...]
//!     r8conv verify <first shard> [<sidecar>]
//!     r8conv path <first shard>
//!
//! `convert` takes V4.1's own predicate: the gate and up stacks
//! (`names::ffn_gate_exps`, `ffn_up_exps`) of every layer `Hparams` reads as
//! routed, in layer order — the stacks the host tier serves. `--tensors`
//! names the stacks instead, for a file that is not V4.1 (the gate's
//! synthetic source). `<out>` and `<sidecar>` default to
//! `r8file::sidecar_path` of the first shard. One line per tensor, a summary
//! line, and exit status 1 with the error on any failure.

use std::error::Error;
use std::path::PathBuf;
use std::process::ExitCode;

use gguf::{Split, Weights};
use model::arch::deepseek41::{hparams::Hparams, names};
use model::r8file::{self, Progress, Sidecar};

const USAGE: &str = "usage: r8conv convert <first shard> [<out>] [--tensors <name>,<name>...]\n       r8conv verify <first shard> [<sidecar>]\n       r8conv path <first shard>";

type Res<T> = Result<T, Box<dyn Error>>;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("r8conv: error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Res<()> {
    let Some((cmd, rest)) = args.split_first() else {
        return Err(USAGE.into());
    };
    match (cmd.as_str(), rest) {
        ("convert", rest) => convert(rest),
        ("verify", [first]) => verify(first, None),
        ("verify", [first, side]) => verify(first, Some(side)),
        ("path", [first]) => {
            println!(
                "{}",
                r8file::sidecar_path(&std::path::absolute(first)?).display()
            );
            Ok(())
        }
        _ => Err(USAGE.into()),
    }
}

/// The gate and up stacks of every routed layer, in layer order.
fn routed_stacks(split: &Split) -> Res<Vec<String>> {
    let hp = Hparams::read(split)?;
    Ok(hp
        .layers
        .iter()
        .enumerate()
        .filter(|(_, kind)| kind.routed)
        .flat_map(|(l, _)| [names::ffn_gate_exps(l), names::ffn_up_exps(l)])
        .collect())
}

fn report(verb: &str, p: Progress<'_>) {
    match p {
        Progress::RemovedPart(part) => {
            println!(
                "r8conv: removed {}, left by a run that did not finish",
                part.display()
            );
        }
        Progress::Tensor(t) => {
            println!(
                "r8conv: {verb} {} bytes={} secs={:.3}",
                t.name, t.bytes, t.secs
            );
        }
    }
}

/// GB/s of `bytes` over `secs`.
fn rate(bytes: u64, secs: f64) -> f64 {
    bytes as f64 / secs.max(1e-9) / 1e9
}

fn convert(args: &[String]) -> Res<()> {
    let mut paths = Vec::new();
    let mut tensors = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--tensors" => {
                let list = it.next().ok_or("--tensors needs a comma-separated list")?;
                tensors = Some(list.split(',').map(str::to_string).collect::<Vec<_>>());
            }
            flag if flag.starts_with("--") => {
                return Err(format!("unknown flag {flag}\n{USAGE}").into());
            }
            _ => paths.push(a),
        }
    }
    let (first, out) = match paths.as_slice() {
        [first] => (std::path::absolute(first)?, None),
        [first, out] => (std::path::absolute(first)?, Some(PathBuf::from(out))),
        _ => return Err(USAGE.into()),
    };
    let split = Split::open(&first)?;
    let names = match tensors {
        Some(list) => list,
        None => routed_stacks(&split)?,
    };
    let out = out.unwrap_or_else(|| r8file::sidecar_path(&first));
    let bytes: u64 = names
        .iter()
        .filter_map(|n| split.find(n))
        .map(|(_, t)| t.nbytes)
        .sum();
    println!(
        "r8conv: convert {} ({} shards) -> {}: {} tensors, {bytes} bytes",
        first.display(),
        split.shard_count(),
        out.display(),
        names.len()
    );
    let stats = r8file::convert(&split, &names, &out, &mut |p| report("convert", p))?;
    let data: u64 = stats.tensors.iter().map(|t| t.bytes).sum();
    println!(
        "r8conv: convert done tensors={} bytes={data} file_bytes={} secs={:.1} gbps={:.2} out={}",
        stats.tensors.len(),
        stats.file_bytes,
        stats.secs,
        rate(data, stats.secs),
        stats.out.display()
    );
    Ok(())
}

fn verify(first: &str, side: Option<&String>) -> Res<()> {
    let first = std::path::absolute(first)?;
    let split = Split::open(&first)?;
    let side = side.map_or_else(|| r8file::sidecar_path(&first), PathBuf::from);
    let sidecar = Sidecar::open(&side, &split, Weights::Mapped { populate: false })?;
    println!(
        "r8conv: verify {} against {} ({} shards): {} tensors",
        side.display(),
        first.display(),
        split.shard_count(),
        sidecar.names().count()
    );
    let stats = r8file::verify(&split, &sidecar, &mut |p| report("verify", p))?;
    println!(
        "r8conv: verify done tensors={} bytes={} secs={:.1} gbps={:.2} sidecar={}",
        stats.tensors.len(),
        stats.bytes,
        stats.secs,
        rate(stats.bytes, stats.secs),
        side.display()
    );
    Ok(())
}
