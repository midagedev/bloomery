//! `--ops`: where a step's wall goes, op by op. A tool, not a clause, and a
//! development-loop reading, not a timed number: the traced step puts every
//! op on a command buffer of its own and waits for it, so an op's wall is
//! its kernels' time plus one commit and one wait; the embed, whose kernel
//! reads one row, is about that floor. The plain step's wall is printed
//! beside the sum.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use bloomery_metal::arch::qwen3moe::session::{Op, Session, StepMode};

use super::says;

/// Plain steps before the readings, and traced steps read.
const WARM: usize = 4;
const READS: usize = 7;

pub fn run(s: &mut Session) -> ExitCode {
    match go(s) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            println!("ops: FAIL ({e})");
            ExitCode::FAILURE
        }
    }
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort_unstable();
    v[v.len() / 2]
}

fn go(s: &mut Session) -> Result<(), String> {
    s.reset().map_err(says)?;
    let mut id = 0u32;
    let mut plain = Vec::new();
    for _ in 0..WARM + READS {
        let t = Instant::now();
        id = s.step(id, StepMode::HostFed).map_err(says)?.id;
        plain.push(t.elapsed());
    }
    let plain = median(plain.split_off(WARM));
    let mut walls: Vec<(Op, Vec<Duration>)> = Vec::new();
    for _ in 0..READS {
        let mut at = 0usize;
        let mut last = Instant::now();
        id = s
            .step_traced(id, |op, _| {
                let now = Instant::now();
                if walls.len() == at {
                    walls.push((op, Vec::new()));
                }
                walls[at].1.push(now - last);
                at += 1;
                last = Instant::now();
                Ok(())
            })
            .map_err(says)?
            .id;
    }
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let mut sum = Duration::ZERO;
    for (op, w) in walls {
        let m = median(w);
        sum += m;
        println!("ops: {op:?} {:.3} ms", ms(m));
    }
    println!(
        "ops: traced sum {:.3} ms over the ops; plain step median {:.3} ms ({READS} steps after \
         {WARM})",
        ms(sum),
        ms(plain)
    );
    Ok(())
}
