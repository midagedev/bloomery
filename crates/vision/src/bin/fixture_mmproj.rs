//! fixture_mmproj — write the synthetic `qwen3vl_merger` encoder file (`vision::arch::qwen3vl::fixture`): the
//! real tensor table with seeded values, the one file an image-input seat needs whose real mmproj
//! is not kept.
//!
//!     fixture_mmproj <out> [--seed N]
//!
//! `out` must not exist (the fixture is never written over); the bytes are a function of the seed
//! alone, so the file at a path is one file. One line per tensor and a summary that ends with the
//! file's sha256 — the figure the seat's families pin — and exit status 1 with the error on any
//! failure.

use std::error::Error;
use std::path::Path;
use std::process::ExitCode;

use vision::arch::qwen3vl::fixture;

const USAGE: &str = "usage: fixture_mmproj <out> [--seed N]";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("fixture_mmproj: error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let (mut out, mut seed) = (None, None);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seed" => {
                if seed.is_some() {
                    return Err(format!("--seed is given twice; {USAGE}").into());
                }
                let v = args
                    .next()
                    .ok_or_else(|| format!("--seed needs a value; {USAGE}"))?;
                seed = Some(
                    v.parse::<u64>()
                        .map_err(|_| format!("--seed {v:?} is not a u64"))?,
                );
            }
            flag if flag.starts_with("--") => {
                return Err(format!("unknown flag {flag}; {USAGE}").into());
            }
            _ => {
                if out.replace(arg).is_some() {
                    return Err(USAGE.into());
                }
            }
        }
    }
    let Some(out) = out else {
        return Err(USAGE.into());
    };
    let seed = seed.unwrap_or(fixture::DEFAULT_SEED);
    let written = fixture::write(Path::new(&out), seed, &mut |name, bytes| {
        println!("fixture_mmproj: tensor {name} bytes={bytes}");
    })?;
    println!(
        "fixture_mmproj: wrote {} tensors={} bytes={} file_bytes={} sha256={} gen_secs={:.2} \
         write_secs={:.2} secs={:.2}",
        written.path.display(),
        written.tensors,
        written.bytes,
        written.file_bytes,
        written.sha256,
        written.gen_secs,
        written.write_secs,
        written.secs
    );
    Ok(())
}
