//! Write the PTX payloads of an `.oxart` section to files, one per bundle
//! that carries one: `oxart_ptx <section.bin> <dir>` reads the bytes
//! `objcopy -O binary --only-section=.oxart` wrote, writes `<dir>/mod<N>.ptx`
//! for N = 1, 2, … in section order, and prints `mod<N> bundle=<name>
//! bytes=<len>` for each. The container is cut by oxide-artifacts' own
//! parser (`bloomery_gpu_gates::ptx`); a section it rejects fails the run
//! with the parser's reason. `tools/ptx-scan.sh` reads nothing else.
//!
//! Host-only: no device code, and the package's device dependency sits
//! behind the `gpu` feature, so plain `cargo build --release -p
//! bloomery-gpu-gates --bin oxart_ptx` builds it with no device crate.

use bloomery_gpu_gates::{GateError, exit_with, ptx};
use std::path::PathBuf;

fn main() -> std::process::ExitCode {
    exit_with("oxart_ptx", run())
}

fn run() -> Result<(), GateError> {
    let mut args = std::env::args_os().skip(1);
    let (Some(section), Some(dir), None) = (args.next(), args.next(), args.next()) else {
        return Err("usage: oxart_ptx <section.bin> <out-dir>".into());
    };
    let (section, dir) = (PathBuf::from(section), PathBuf::from(dir));
    let bytes = std::fs::read(&section).map_err(|e| format!("read {}: {e}", section.display()))?;
    let bundles =
        ptx::section_bundles(&bytes).map_err(|e| format!("{}: {e}", section.display()))?;
    for (n, m) in (1_usize..).zip(ptx::modules(&bundles)) {
        let path = dir.join(format!("mod{n}.ptx"));
        std::fs::write(&path, m.text()).map_err(|e| format!("write {}: {e}", path.display()))?;
        println!("mod{n} bundle={} bytes={}", m.bundle(), m.text().len());
    }
    Ok(())
}
