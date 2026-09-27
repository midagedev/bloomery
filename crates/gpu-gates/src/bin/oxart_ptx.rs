//! Write the PTX payloads of an `.oxart` section to files, one per bundle
//! that carries one: `oxart_ptx <section.bin> <dir>` reads the bytes
//! `objcopy -O binary --only-section=.oxart` wrote, writes `<dir>/mod<N>.ptx`
//! for N = 1, 2, … in section order, and prints `mod<N> bundle=<name>
//! bytes=<len>` for each. The container is cut by oxide-artifacts' own
//! parser (`bloomery_gpu_gates::ptx`); a section it rejects fails the run
//! with the parser's reason. `tools/ptx-scan.sh` reads nothing else.
//!
//! `oxart_ptx --norm <section.bin> <dir>` does the same and also writes
//! `<dir>/norm/<entry>` for every entry: its body through `ptx::normalize`
//! against its module's declarations, the text the scan's digest is the md5
//! of, and `<dir>/norm-method`, the name of that rule set
//! (`ptx::DIGEST_METHOD`). What it prints is unchanged.
//!
//! `oxart_ptx --norm-ptx <module.ptx> <dir>` writes the same `norm/<entry>`
//! files and `norm-method` for a saved module (a `mod<N>.ptx` of the above,
//! or an edited copy of one): the digest of a module that is not in a
//! binary.
//!
//! Host-only: no device code, and the package's device dependency sits
//! behind the `gpu` feature, so plain `cargo build --release -p
//! bloomery-gpu-gates --bin oxart_ptx` builds it with no device crate.

use bloomery_gpu_gates::{GateError, exit_with, ptx};
use std::path::{Path, PathBuf};

const USAGE: &str = "usage: oxart_ptx [--norm] <section.bin> <out-dir> | oxart_ptx --norm-ptx <module.ptx> <out-dir>";

fn main() -> std::process::ExitCode {
    exit_with("oxart_ptx", run())
}

fn run() -> Result<(), GateError> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>()[..] {
        [section, dir] => extract(Path::new(section), Path::new(dir), false),
        ["--norm", section, dir] => extract(Path::new(section), Path::new(dir), true),
        ["--norm-ptx", module, dir] => norm_ptx(Path::new(module), Path::new(dir)),
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
        write_norm(&modules, dir)?;
    }
    Ok(())
}

/// `<dir>/norm/<entry>` for every entry of `modules`, and `<dir>/norm-method`.
fn write_norm(modules: &[ptx::Module<'_>], dir: &Path) -> Result<(), GateError> {
    let out = dir.join("norm");
    std::fs::create_dir_all(&out).map_err(|e| format!("mkdir {}: {e}", out.display()))?;
    let decls = ptx::module_decls(modules)?;
    for name in ptx::entries(modules) {
        let path: PathBuf = out.join(&name);
        std::fs::write(&path, ptx::normalized(modules, &decls, &name)?)
            .map_err(|e| format!("write {}: {e}", path.display()))?;
    }
    let method = dir.join("norm-method");
    std::fs::write(&method, format!("{}\n", ptx::DIGEST_METHOD))
        .map_err(|e| format!("write {}: {e}", method.display()))?;
    Ok(())
}

/// The normalized bodies of a saved module file.
fn norm_ptx(file: &Path, dir: &Path) -> Result<(), GateError> {
    let text = std::fs::read(file).map_err(|e| format!("read {}: {e}", file.display()))?;
    let name = file.display().to_string();
    write_norm(&[ptx::Module::saved(&name, &text)], dir)
}
