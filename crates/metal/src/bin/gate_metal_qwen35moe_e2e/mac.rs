//! The qwen35moe e2e gate's clauses (macOS only).
//!
//! The session (`bloomery_metal::arch::qwen3moe::session`) opens the file
//! the fixture family runs (`refset::fixture::first_shard`), the file its
//! reference sets were dumped from, and every clause starts from its reset:
//!
//! - `d_count`: every step of a depth within one segment of keys encodes the
//!   groups and dispatches `count::token_count` gives (one barrier between
//!   groups), the device's allocated size is the same before and after those
//!   steps (the step allocates nothing on the device), and the host's
//!   allocation calls, read as a difference around the step (the first step,
//!   which may warm the driver's paths, aside), are none in every steady
//!   step. It prints the step's wall per token, host-fed and waited.
//! - `e_token_id`: an id past the vocabulary raises the token-id site at
//!   layer 0 and poisons the session; the next step is refused by name; the
//!   reset clears it and a step runs again.
//! - `a_free`: the batch set's tokens host-fed one at a time; every layer's
//!   output over the rows ik kept within [`free::FREE_BAND`] of ik's, the
//!   last position's argmax as the fixture tier takes it, the rows a
//!   near-tie router flip carries left unheld by the CUDA gate's rule; and
//!   the traced run's logits and picks equal to the plain steps'.
//!
//! `--nodes` is a tool, not a clause: the step set's prompt host-fed, then
//! its step token traced, every named node against ik's, the first past its
//! band printed. `--ops` is another: each op's wall in the traced step
//! (`wall.rs`). `--list` prints the clause names; `--only CLAUSE` runs one;
//! `--mutant NAME` runs the clause that must catch a kernel defect against
//! kernels compiled with it (`mutants.rs`); `--mutants-all` runs each in a
//! child gate and is red when one stays green.
//!
//! Exit 1 when a clause failed, 78 when this machine cannot run the backend,
//! 64 on a bad argument, 0 otherwise.

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::PathBuf;
use std::process::{Command, ExitCode};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

use bloomery_device::conformance::{Clause, Report, Verdict, why};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Device, DeviceError};
use bloomery_metal::arch::qwen3moe::count::{LiveKeys, Ticket, token_count};
use bloomery_metal::arch::qwen3moe::session::{Session, SessionError};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::library::Library;

#[path = "flip.rs"]
mod flip;
#[path = "free.rs"]
mod free;
#[path = "mutants.rs"]
mod mutants;
#[path = "nodes.rs"]
mod nodes;
#[path = "sets.rs"]
mod sets;
#[path = "wall.rs"]
mod wall;

const GATE: &str = "gate_metal_qwen35moe_e2e";

/// The cache rows a session of this gate holds: one segment of keys, the
/// depths `d_count` pins.
const CTX: usize = bloomery_metal::kernels::SEGMENT_KEYS;

/// The token `d_count` and `e_token_id` start from: any id the vocabulary
/// holds. Neither clause reads a reference set.
const SEED: u32 = 0;

/// `--name VALUE`'s value.
pub fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let at = args.iter().position(|a| a == name)?;
    args.get(at + 1).map(String::as_str)
}

/// A session error as a clause's reason.
pub fn says(e: SessionError) -> String {
    e.to_string()
}

/// The file the fixture family runs: the one its sets were dumped from.
fn model_path() -> Result<PathBuf, String> {
    refset::fixture::first_shard(refset::arch::qwen35moe::ARCH)
        .map(PathBuf::from)
        .map_err(|e| format!("the fixture file: {e}"))
}

/// Allocation calls the process has made, for `d_count`'s host half: read as
/// a difference around a step, it is the step's own count, whatever the
/// process allocates for its output and errors outside it.
static HOST_ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);

fn counted() {
    HOST_ALLOC_CALLS.fetch_add(1, Relaxed);
}

/// The process's allocator, which counts every allocating call and forwards
/// it to `System` unchanged.
struct Counting;

// SAFETY: every method forwards to `System` with the caller's arguments unchanged.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        counted();
        // SAFETY: the caller meets `alloc`'s contract for `l`, and `System`
        // receives `l` unchanged.
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        counted();
        // SAFETY: the caller meets `alloc_zeroed`'s contract for `l`, and
        // `System` receives `l` unchanged.
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
        counted();
        // SAFETY: `p` came from this allocator with layout `l` — so from
        // `System`, as every allocating method forwards there — and the
        // caller meets `realloc`'s contract for `new_size`; `System` receives
        // all three unchanged.
        unsafe { System.realloc(p, l, new_size) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: `p` came from this allocator with layout `l`, so from
        // `System`, as every allocating method forwards there.
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// `d`: the token's count at every depth of one segment, and no allocation on
/// either side of a steady step.
fn d_count(s: &mut Session, device: &MetalDevice) -> Clause {
    s.reset().map_err(says)?;
    let want = token_count(s.token_layers(), LiveKeys::Within64, Ticket::Folded);
    let before = device.info().map_err(why)?.free_bytes;
    let mut walls: Vec<Duration> = Vec::with_capacity(s.ctx());
    let mut host: Vec<u64> = Vec::with_capacity(s.ctx().saturating_sub(1));
    let mut id = SEED;
    for step in 0..s.ctx() {
        let pos = s.pos();
        let t = Instant::now();
        let calls0 = HOST_ALLOC_CALLS.load(Relaxed);
        let stepped = s.step(id).map_err(says)?;
        walls.push(t.elapsed());
        let calls = HOST_ALLOC_CALLS.load(Relaxed) - calls0;
        let c = stepped.count;
        let (groups, dispatches) = (want.groups as usize, want.dispatches as usize);
        if c.groups != groups || c.dispatches != dispatches || c.barriers + 1 != c.groups {
            return Err(format!(
                "position {pos}: {} groups, {} dispatches, {} barriers encoded; the count is \
                 {groups} groups, {dispatches} dispatches, {} barriers",
                c.groups,
                c.dispatches,
                c.barriers,
                groups - 1
            ));
        }
        // The first step may warm the driver's paths; a later one is steady
        // and must allocate nothing on the host.
        if step > 0 {
            host.push(calls);
            if calls != 0 {
                return Err(format!(
                    "position {pos}: {calls} host allocation calls in the step; a steady \
                     step allocates none"
                ));
            }
        }
        id = stepped.id;
    }
    let after = device.info().map_err(why)?.free_bytes;
    if after != before {
        return Err(format!(
            "the device's free bytes moved from {before} to {after} across {} steps: a step \
             allocated",
            s.ctx()
        ));
    }
    walls.sort_unstable();
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let mid = walls[walls.len() / 2];
    println!(
        "d_count: step wall per token, host-fed and waited, {} steps: median {:.3} ms, min {:.3} \
         ms, max {:.3} ms (development-loop number)",
        walls.len(),
        ms(mid),
        ms(walls[0]),
        ms(walls[walls.len() - 1])
    );
    let steady = host.iter().copied().max().unwrap_or(0);
    println!(
        "d_count: host allocation calls per steady step ({} steps, the first aside): {steady}",
        host.len()
    );
    Ok(Verdict::Noted(format!(
        "{} steps at positions 0..{}: {} groups, {} dispatches, {} barriers each; device \
         allocation unmoved; {steady} host allocation calls in every steady step",
        s.ctx(),
        s.ctx(),
        want.groups,
        want.dispatches,
        want.groups - 1
    )))
}

/// `e`: an id past the vocabulary raises the token-id site at layer 0, the
/// session refuses the next step by name, and the reset clears it.
fn e_token_id(s: &mut Session) -> Clause {
    s.reset().map_err(says)?;
    let bad = u32::try_from(s.vocab()).map_err(|_| "the vocabulary does not fit u32")?;
    let fault = match s.step(bad) {
        Err(SessionError::Fault(f)) => f,
        Err(e) => return Err(format!("id {bad}: another error than a fault: {e}")),
        Ok(st) => {
            return Err(format!(
                "id {bad} (the vocabulary's size) stepped clean, picking {}",
                st.id
            ));
        }
    };
    let token_bit = 1u32 << (FaultSite::TokenId as u32);
    if fault.layer != 0 || fault.sites & token_bit == 0 {
        return Err(format!(
            "id {bad} raised {fault}; the token-id site at layer 0 is not among them"
        ));
    }
    let good = SEED;
    match s.step(good) {
        Err(SessionError::Poisoned(f)) if f == fault => {}
        Err(e) => {
            return Err(format!(
                "the step after the fault: {e}; a refusal by poison is expected"
            ));
        }
        Ok(_) => return Err("the step after the fault ran on a poisoned session".to_string()),
    }
    s.reset().map_err(says)?;
    let st = s
        .step(good)
        .map_err(|e| format!("the step after the reset: {}", says(e)))?;
    if st.id as usize >= s.vocab() {
        return Err(format!("the step after the reset picked {}", st.id));
    }
    Ok(Verdict::Noted(format!(
        "id {bad} raised {fault}; the next step refused by poison; after the reset id {good} \
         picked {}",
        st.id
    )))
}

/// The library the session's kernels come from: the device's, or the
/// source with a mutant's edits in.
fn library_of(
    device: &MetalDevice,
    mutant: Option<&mutants::Mutant>,
) -> Result<Option<Library>, String> {
    match mutant {
        None => Ok(None),
        Some(m) => {
            let text = mutants::source_of(m)?;
            Library::from_source_text(device, &text)
                .map(Some)
                .map_err(why)
        }
    }
}

pub fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(name) = ["--only", "--mutant"]
        .into_iter()
        .find(|name| args.last().is_some_and(|last| last == name))
    {
        println!("{GATE}: {name} takes a value");
        return ExitCode::from(64);
    }
    const NAMES: [&str; 3] = ["d_count", "e_token_id", "a_free"];
    if args.iter().any(|a| a == "--list") {
        for name in NAMES {
            println!("{name}");
        }
        return ExitCode::SUCCESS;
    }
    if args.iter().any(|a| a == "--mutants-all") {
        return mutants_all();
    }
    let device = match MetalDevice::open() {
        Ok(device) => device,
        Err(e @ DeviceError::Unsupported { .. }) => {
            println!("clause open: REFUSED ({e})");
            return ExitCode::from(78);
        }
        Err(e) => {
            println!("clause open: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    let mutant = match flag(&args, "--mutant") {
        Some(name) => match mutants::find(name) {
            Some(m) => Some(m),
            None => {
                println!("{GATE}: no mutant named {name}");
                return ExitCode::from(64);
            }
        },
        None => None,
    };
    let mutated = match library_of(&device, mutant) {
        Ok(lib) => lib,
        Err(e) => {
            println!("{GATE}: the mutant does not build: {e}");
            return ExitCode::from(64);
        }
    };
    let library = mutated.as_ref().unwrap_or_else(|| device.library());
    if let Some(m) = mutant {
        println!("mutant: {} ({})", m.name, m.why);
    }
    let only = mutant.map(|m| m.clause).or(flag(&args, "--only"));
    if let Some(name) = only
        && !NAMES.contains(&name)
    {
        println!("{GATE}: no clause named {name}");
        return ExitCode::from(64);
    }
    let path = match model_path() {
        Ok(p) => p,
        Err(e) => {
            println!("clause open: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    let t = Instant::now();
    let mut session = match Session::open(&device, library, &path, CTX) {
        Ok(s) => s,
        Err(e) => {
            println!("clause open: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "session: {} ({} layers, ctx {CTX}) open in {:.1} s; token {:?}",
        path.display(),
        session.layer_count(),
        t.elapsed().as_secs_f64(),
        session.token_layers()
    );
    if args.iter().any(|a| a == "--nodes") {
        return nodes::run(&mut session);
    }
    if args.iter().any(|a| a == "--ops") {
        return wall::run(&mut session);
    }
    let mut report = Report::new();
    for name in NAMES {
        if only.is_some_and(|o| o != name) {
            continue;
        }
        let result = match name {
            "d_count" => d_count(&mut session, &device),
            "e_token_id" => e_token_id(&mut session),
            _ => free::a_free(&mut session),
        };
        report.clause(name, result);
    }
    let scope = match (mutant, only) {
        (Some(m), _) => format!(" (mutant {})", m.name),
        (None, Some(name)) => format!(" (only {name})"),
        (None, None) => String::new(),
    };
    report.finish(&format!("{GATE}{scope}"))
}

/// Every mutant of the table, each in a child run of this gate
/// (`--mutant NAME`): its clause must be red and the child's exit code not 0.
fn mutants_all() -> ExitCode {
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(e) => {
            println!("{GATE} --mutants-all: FAIL (the gate's own path: {e})");
            return ExitCode::FAILURE;
        }
    };
    let mut green = Vec::new();
    for mutant in mutants::MUTANTS {
        let out = match Command::new(&exe).args(["--mutant", mutant.name]).output() {
            Ok(out) => out,
            Err(e) => {
                println!("mutant {}: the child gate did not start: {e}", mutant.name);
                return ExitCode::FAILURE;
            }
        };
        let text = String::from_utf8_lossy(&out.stdout);
        let red =
            !out.status.success() && text.contains(&format!("clause {}: FAIL", mutant.clause));
        println!(
            "mutant {}: {} on {}",
            mutant.name,
            if red { "red" } else { "NOT RED" },
            mutant.clause
        );
        if !red {
            green.push(mutant.name);
        }
    }
    if green.is_empty() {
        println!(
            "{GATE} --mutants-all: PASS ({} mutants, each red on its clause)",
            mutants::MUTANTS.len()
        );
        return ExitCode::SUCCESS;
    }
    println!("{GATE} --mutants-all: FAIL (not red: {})", green.join(", "));
    ExitCode::FAILURE
}
