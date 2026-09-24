//! Write the PTX payloads of an `.oxart` section to files, one per bundle
//! that carries one: `oxart_ptx <section.bin> <dir>` reads the bytes
//! `objcopy -O binary --only-section=.oxart` wrote, writes `<dir>/mod<N>.ptx`
//! for N = 1, 2, … in section order, and prints `mod<N> bundle=<name>
//! bytes=<len>` for each. The container is cut by oxide-artifacts' own
//! parser (`bloomery_gpu_gates::ptx`); a section it rejects fails the run
//! with the parser's reason. `tools/ptx-scan.sh` reads nothing else.
//!
//! `oxart_ptx --norm <section.bin> <dir>` does the same and also writes
//! `<dir>/norm/<entry>` for every entry: its body through `ptx::normalize`,
//! the text the scan's digest is the md5 of. What it prints is unchanged.
//!
//! `oxart_ptx --norm-dump <entry> <file>` prints the normalized body of a
//! saved single-entry dump: a file that opens with the entry's `.visible
//! .entry <entry>(` line and ends where its body ends. The bytes after that
//! opening line go through `ptx::normalize` as the entry's body.
//!
//! Host-only: no device code, and the package's device dependency sits
//! behind the `gpu` feature, so plain `cargo build --release -p
//! bloomery-gpu-gates --bin oxart_ptx` builds it with no device crate.

use bloomery_gpu_gates::{GateError, exit_with, ptx};
use std::io::Write;
use std::path::{Path, PathBuf};

const USAGE: &str =
    "usage: oxart_ptx [--norm] <section.bin> <out-dir> | oxart_ptx --norm-dump <entry> <file>";

fn main() -> std::process::ExitCode {
    exit_with("oxart_ptx", run())
}

fn run() -> Result<(), GateError> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>()[..] {
        [section, dir] => extract(Path::new(section), Path::new(dir), false),
        ["--norm", section, dir] => extract(Path::new(section), Path::new(dir), true),
        ["--norm-dump", entry, file] => norm_dump(entry, Path::new(file)),
        _ => Err(USAGE.into()),
    }
}

/// One `mod<N>.ptx` per PTX module; with `norm`, one `norm/<entry>` per entry.
fn extract(section: &Path, dir: &Path, norm: bool) -> Result<(), GateError> {
    let bytes = std::fs::read(section).map_err(|e| format!("read {}: {e}", section.display()))?;
    let bundles =
        ptx::section_bundles(&bytes).map_err(|e| format!("{}: {e}", section.display()))?;
    let modules = ptx::modules(&bundles).map_err(|e| format!("{}: {e}", section.display()))?;
    for (n, m) in (1_usize..).zip(&modules) {
        let path = dir.join(format!("mod{n}.ptx"));
        std::fs::write(&path, m.text()).map_err(|e| format!("write {}: {e}", path.display()))?;
        println!("mod{n} bundle={} bytes={}", m.bundle(), m.text().len());
    }
    if norm {
        let out = dir.join("norm");
        std::fs::create_dir_all(&out).map_err(|e| format!("mkdir {}: {e}", out.display()))?;
        for name in ptx::entries(&modules) {
            let body = ptx::body(&modules, &name).ok_or_else(|| format!("no body for {name}"))?;
            let path: PathBuf = out.join(&name);
            std::fs::write(&path, ptx::normalize(&name, body))
                .map_err(|e| format!("write {}: {e}", path.display()))?;
        }
    }
    Ok(())
}

/// The normalized body of a saved single-entry dump, on stdout.
fn norm_dump(entry: &str, file: &Path) -> Result<(), GateError> {
    let text = std::fs::read(file).map_err(|e| format!("read {}: {e}", file.display()))?;
    let head = format!(".visible .entry {entry}(");
    let body = text
        .trim_ascii_start()
        .strip_prefix(head.as_bytes())
        .ok_or_else(|| format!("{} does not open with `{head}`", file.display()))?;
    std::io::stdout()
        .write_all(&ptx::normalize(entry, body))
        .map_err(|e| format!("write stdout: {e}"))?;
    Ok(())
}
