//! The residency-set teardown repro: the pattern the Metal e2e gate panicked
//! the kernel in, in a bounded loop, and nothing else (`residency::repro`). It
//! is the queue-drain fix's FAIL-first, not a gate clause.
//!
//! `--iterations N` (default 1000) runs the loop N times. `--wait` waits for
//! each command buffer before the teardown (the rule the fix keeps), so the
//! loop runs to the end; without it the teardown frees the buffer under an
//! in-flight command buffer, the driver walks a freed resource, and the
//! kernel faults. **Do not run the no-`--wait` arm except on a Mac the user
//! names, with the user's go: it panics the machine.**
//!
//! The body is macOS only; elsewhere it refuses and exits 78.

#[cfg(target_os = "macos")]
fn main() -> std::process::ExitCode {
    use bloomery_metal::device::MetalDevice;
    use bloomery_metal::instance::HOST;
    use bloomery_metal::residency;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let wait = args.iter().any(|a| a == "--wait");
    let iterations: u32 = match args.iter().position(|a| a == "--iterations") {
        Some(i) => match args.get(i + 1).and_then(|v| v.parse().ok()) {
            Some(n) => n,
            None => {
                eprintln!("metal_residency_repro: --iterations takes a whole number");
                return std::process::ExitCode::from(64);
            }
        },
        None => 1000,
    };
    let device = match MetalDevice::open(&HOST) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("metal_residency_repro: the Metal device: {e}");
            return std::process::ExitCode::from(78);
        }
    };
    println!(
        "metal_residency_repro: {iterations} iterations, wait_for_queue={wait} \
         ({})",
        if wait {
            "the rule the fix keeps"
        } else {
            "the buggy teardown; panics the kernel on a build without the stream drain"
        }
    );
    match residency::repro(&device, iterations, wait) {
        Ok(()) => {
            println!("metal_residency_repro: survived {iterations} iterations");
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("metal_residency_repro: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn main() -> std::process::ExitCode {
    eprintln!("metal_residency_repro: refused: the Metal repro runs on macOS only");
    std::process::ExitCode::from(78)
}
