//! refset-check: which reference sets are stale, in one command.
//!
//!   refset-check                      every family's sets in place
//!   refset-check <family> <path>...   the sets at <path>, as sets of <family>
//!   refset-check --fixture-line <first shard>
//!                                     the `# fixture` line of a fixture file
//!
//! One line per set: the family, the set, what it states it was dumped from
//! (the draft file and the ik build where the family records them) and
//! `PASS`, or `FAIL` and the reason the family's reader refuses it —
//! `stale reference: <set> was dumped from <file>, the tree runs <file>` for
//! a set of another model file. A `<path>` is a set directory, a greedy tsv,
//! or a KLD base's `.kld` file. Exit 1 when any set fails, 2 on a family name
//! the table does not hold. The file the tree runs is the V4.1 file
//! (`BLOOMERY_V41_MODEL`); the draft set needs `BLOOMERY_DSPARK_MODEL`, which
//! `just refset-check` exports as the dspark recipes do. The tree runs the
//! fixture families' files too (`BLOOMERY_FIXTURE_ROOT`, default
//! `/models/fixtures`); a family whose file cannot be found is listed as
//! `<family>: <the error>`.
//!
//! `--fixture-line` is what the dumpers write as a set's `# fixture` line: it
//! prints that line, and nothing else, to stdout and exits 0; a file that is
//! not a whole fixture is the error on stderr and exit 1. Exit 2 on any
//! argument count but one.

use refset::arch;
use refset::family::Family;
use refset::fixture;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// The line of set `path`; whether it passed.
fn line(f: &Family, path: &Path) -> bool {
    match f.check_set(path) {
        Ok(p) => {
            let draft = p.draft.map_or_else(String::new, |d| format!(" draft {d}"));
            let build = p.build.map_or_else(String::new, |b| format!(" build {b}"));
            println!(
                "{} {}: dumped from {}{draft}{build} — PASS",
                f.name,
                path.display(),
                p.dumped_from
            );
            true
        }
        Err(e) => {
            println!("{} {}: FAIL {e}", f.name, path.display());
            false
        }
    }
}

/// `--fixture-line <first shard>`: the file's `# fixture` line on stdout.
fn fixture_line(args: &[String]) -> ExitCode {
    let [file] = args else {
        eprintln!("refset-check: usage: refset-check --fixture-line <first shard>");
        return ExitCode::from(2);
    };
    match fixture::line(Path::new(file)) {
        Ok(line) => {
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("refset-check: {e}");
            ExitCode::FAILURE
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--fixture-line") {
        return fixture_line(&args[1..]);
    }
    let sets: Vec<(&Family, PathBuf)> = match args.split_first() {
        None => arch::all()
            .flat_map(|f| f.in_place().into_iter().map(move |p| (f, p)))
            .collect(),
        Some((name, paths)) => {
            let Some(f) = arch::named(name) else {
                let names: Vec<&str> = arch::all().map(|f| f.name).collect();
                eprintln!("refset-check: no family {name:?}; the table holds {names:?}");
                return ExitCode::from(2);
            };
            paths.iter().map(|p| (f, PathBuf::from(p))).collect()
        }
    };
    let mut runs: Vec<String> = arch::all()
        .filter(|f| f.runs.is_some())
        .map(|f| f.runs().unwrap_or_else(|e| format!("{}: {e}", f.name)))
        .collect();
    runs.dedup();
    println!("refset-check: the tree runs {}", runs.join(", "));
    let failed = sets.iter().filter(|(f, p)| !line(f, p)).count();
    println!("refset-check: {} set(s), {failed} failed", sets.len());
    if failed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
