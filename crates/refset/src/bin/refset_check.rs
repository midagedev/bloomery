//! refset-check: which reference sets are stale, in one command.
//!
//!   refset-check                      every family's sets in place
//!   refset-check <family> <path>...   the sets at <path>, as sets of <family>
//!
//! One line per set: the family, the set, what it states it was dumped from
//! (the draft file and the ik build where the family records them) and
//! `PASS`, or `FAIL` and the reason the family's reader refuses it —
//! `stale reference: <set> was dumped from <file>, the tree runs <file>` for
//! a set of another model file. A `<path>` is a set directory, a greedy tsv,
//! or a KLD base's `.kld` file. Exit 1 when any set fails, 2 on a family name
//! the table does not hold. The file the tree runs is the V4.1 file
//! (`BLOOMERY_V41_MODEL`); the draft set needs `BLOOMERY_DSPARK_MODEL`, which
//! `just refset-check` exports as the dspark recipes do.

use refset::arch;
use refset::family::Family;
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

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
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
    let mut runs: Vec<String> = arch::all().filter_map(|f| f.runs.map(|r| r())).collect();
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
