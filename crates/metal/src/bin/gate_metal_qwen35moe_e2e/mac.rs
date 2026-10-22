//! The qwen35moe e2e gate's clauses (macOS only).
//!
//! The session (`bloomery_metal::arch::qwen3moe::session`) opens the file
//! the fixture family runs (`refset::fixture::first_shard`), the file its
//! reference sets were dumped from, and every clause starts from its reset:
//!
//! - `d_count`: host-fed, then in chain steps, every step of a depth within
//!   one segment of keys encodes the groups and dispatches
//!   `count::token_count` gives (one barrier between groups) and waits once;
//!   the device's allocated size is the same before and after those steps
//!   (the step allocates nothing on the device), and the host's allocation
//!   calls, read as a difference around the step (each run's first step,
//!   which may warm the driver's paths, aside), are none in every steady
//!   step. The chain's last call, at the cache's last row, commits nothing
//!   ahead. It prints the step's wall per token in both modes.
//! - `e_token_id`: an id past the vocabulary raises the token-id site at
//!   layer 0 and poisons the session; the next step is refused by name; the
//!   reset clears it and a step runs again. The same in chain steps, where
//!   the token ahead embeds the faulted token's pick: the fault is the
//!   faulted token's, its position does not move, and after the reset chain
//!   steps run clean past the dropped token's position.
//! - `b_chain`, `c_resume`: the run-ahead clauses (`chain.rs`).
//! - `a_free`: the batch set's tokens host-fed one at a time; every layer's
//!   output over the rows ik kept within [`free::FREE_BAND`] of ik's, the
//!   last position's argmax as the fixture tier takes it, the rows a
//!   near-tie router flip carries left unheld by the CUDA gate's rule; and
//!   the traced run's logits and picks equal to the plain steps'.
//!
//! `--nodes` is a tool, not a clause: the step set's prompt host-fed, then
//! its step token traced, every named node against ik's, the first past its
//! band printed. `--ops` is another: each op's wall in the traced step,
//! the host-fed and chain step medians beside it (`wall.rs`). `--list` prints the clause names; `--only CLAUSE` runs one;
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
use bloomery_device::fault::{Fault, FaultSite};
use bloomery_device::{Device, DeviceError};
use bloomery_metal::arch::qwen3moe::count::{LiveKeys, Ticket, token_count};
use bloomery_metal::arch::qwen3moe::session::{Session, SessionError, StepMode};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::library::Library;

#[path = "chain.rs"]
mod chain;
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

/// The chain steps `e_token_id` runs after its reset: past the position of
/// the token ahead the faulted step dropped.
const CHAIN_AFTER_RESET: usize = 3;

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

/// One `d_count` run in `mode` from its reset to the cache's last row: every
/// call's count and its one wait, and no host allocation in a steady call.
/// The walls of the calls, in order.
fn d_run(s: &mut Session, mode: StepMode) -> Result<Vec<Duration>, String> {
    s.reset().map_err(says)?;
    let want = token_count(s.token_layers(), LiveKeys::Within64, Ticket::Folded);
    let (groups, dispatches) = (want.groups as usize, want.dispatches as usize);
    let mut walls: Vec<Duration> = Vec::with_capacity(s.ctx());
    let mut id = SEED;
    for call in 0..s.ctx() {
        let pos = s.pos();
        let waits = s.stats().waits;
        let t = Instant::now();
        let calls0 = HOST_ALLOC_CALLS.load(Relaxed);
        let stepped = s.step(id, mode).map_err(says)?;
        let calls = HOST_ALLOC_CALLS.load(Relaxed) - calls0;
        walls.push(t.elapsed());
        let waits = s.stats().waits - waits;
        let c = stepped.count;
        if c.groups != groups || c.dispatches != dispatches || c.barriers + 1 != c.groups {
            return Err(format!(
                "{mode:?} position {pos}: {} groups, {} dispatches, {} barriers encoded; the \
                 count is {groups} groups, {dispatches} dispatches, {} barriers",
                c.groups,
                c.dispatches,
                c.barriers,
                groups - 1
            ));
        }
        if waits != 1 {
            return Err(format!(
                "{mode:?} position {pos}: the step waited {waits} times; a step waits once"
            ));
        }
        // The first step may warm the driver's paths; a later one is steady
        // and must allocate nothing on the host.
        if call > 0 && calls != 0 {
            return Err(format!(
                "{mode:?} position {pos}: {calls} host allocation calls in the step; a steady \
                 step allocates none"
            ));
        }
        id = stepped.id;
    }
    Ok(walls)
}

/// The median and the 90th percentile of `walls`, in ms.
fn spread(mut walls: Vec<Duration>) -> (f64, f64) {
    walls.sort_unstable();
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let at = |q: usize| ms(walls[(walls.len() - 1) * q / 100]);
    (at(50), at(90))
}

/// `d`: the token's count at every depth of one segment and one wait a step,
/// host-fed and in chain steps, and no allocation on either side of a steady
/// step.
fn d_count(s: &mut Session, device: &MetalDevice) -> Clause {
    let want = token_count(s.token_layers(), LiveKeys::Within64, Ticket::Folded);
    let before = device.info().map_err(why)?.free_bytes;
    let host = d_run(s, StepMode::HostFed)?;
    let stats0 = s.stats();
    let chain = d_run(s, StepMode::Chain)?;
    let stats = chain::moved(s, stats0);
    let after = device.info().map_err(why)?.free_bytes;
    if after != before {
        return Err(format!(
            "the device's free bytes moved from {before} to {after} across {} steps: a step \
             allocated",
            2 * s.ctx()
        ));
    }
    let ctx = s.ctx() as u64;
    if stats.last_row != 1 || stats.ahead != ctx - 1 || stats.kept != ctx - 1 {
        return Err(format!(
            "{ctx} chain calls: {} tokens ahead, {} kept, {} at the last row; {} ahead, {} kept \
             and 1 at the last row are expected",
            stats.ahead,
            stats.kept,
            stats.last_row,
            ctx - 1,
            ctx - 1
        ));
    }
    // Steady: the first call, which commits two tokens, and the chain's last,
    // which commits none ahead, aside.
    let (host_mid, host_p90) = spread(host[1..].to_vec());
    let (chain_mid, chain_p90) = spread(chain[1..chain.len() - 1].to_vec());
    println!(
        "d_count: step wall per token, host-fed, {} steady steps: median {host_mid:.3} ms, p90 \
         {host_p90:.3} ms (development-loop number)",
        host.len() - 1
    );
    println!(
        "d_count: step wall per token, chain, {} steady calls: median {chain_mid:.3} ms, p90 \
         {chain_p90:.3} ms (development-loop number)",
        chain.len() - 2
    );
    Ok(Verdict::Noted(format!(
        "{ctx} steps at positions 0..{ctx}, host-fed and in chain steps: {} groups, {} \
         dispatches, {} barriers and one wait each; device allocation unmoved; no host \
         allocation call in a steady step; call {ctx} took the last-row rule",
        want.groups,
        want.dispatches,
        want.groups - 1
    )))
}

/// The fault an id past the vocabulary raises in `mode`, held to the
/// token-id site at layer 0, and the refusal by poison of the step after it.
fn e_fault(s: &mut Session, mode: StepMode) -> Result<Fault, String> {
    s.reset().map_err(says)?;
    let bad = u32::try_from(s.vocab()).map_err(|_| "the vocabulary does not fit u32")?;
    let fault = match s.step(bad, mode) {
        Err(SessionError::Fault(f)) => f,
        Err(e) => {
            return Err(format!(
                "{mode:?} id {bad}: another error than a fault: {e}"
            ));
        }
        Ok(st) => {
            return Err(format!(
                "{mode:?} id {bad} (the vocabulary's size) stepped clean, picking {}",
                st.id
            ));
        }
    };
    let token_bit = 1u32 << (FaultSite::TokenId as u32);
    if fault.layer != 0 || fault.sites & token_bit == 0 {
        return Err(format!(
            "{mode:?} id {bad} raised {fault}; the token-id site at layer 0 is not among them"
        ));
    }
    if s.pos() != 0 {
        return Err(format!(
            "{mode:?}: the faulted step moved the position to {}",
            s.pos()
        ));
    }
    match s.step(SEED, mode) {
        Err(SessionError::Poisoned(f)) if f == fault => Ok(fault),
        Err(e) => Err(format!(
            "{mode:?}: the step after the fault: {e}; a refusal by poison is expected"
        )),
        Ok(_) => Err(format!(
            "{mode:?}: the step after the fault ran on a poisoned session"
        )),
    }
}

/// `e`: an id past the vocabulary raises the token-id site at layer 0, the
/// session refuses the next step by name, and the reset clears it; host-fed,
/// then in chain steps, where the fault is the same and the reset clears
/// what the dropped token ahead raised.
fn e_token_id(s: &mut Session) -> Clause {
    let fault = e_fault(s, StepMode::HostFed)?;
    s.reset().map_err(says)?;
    let st = s
        .step(SEED, StepMode::HostFed)
        .map_err(|e| format!("the step after the reset: {}", says(e)))?;
    if st.id as usize >= s.vocab() {
        return Err(format!("the step after the reset picked {}", st.id));
    }
    let dropped = s.stats().discarded;
    let chained = e_fault(s, StepMode::Chain)?;
    if chained != fault {
        return Err(format!(
            "in chain steps the id raised {chained}; host-fed it raised {fault}"
        ));
    }
    if s.stats().discarded != dropped + 1 {
        return Err("the faulted chain step did not drop its token ahead".to_string());
    }
    s.reset().map_err(says)?;
    let mut id = SEED;
    for _ in 0..CHAIN_AFTER_RESET {
        id = s
            .step(id, StepMode::Chain)
            .map_err(|e| format!("a chain step after the reset: {}", says(e)))?
            .id;
        if id as usize >= s.vocab() {
            return Err(format!("a chain step after the reset picked {id}"));
        }
    }
    Ok(Verdict::Noted(format!(
        "id {} raised {fault}, host-fed and in chain steps; the next step refused by poison; \
         after each reset steps ran clean, {CHAIN_AFTER_RESET} of them in chain steps",
        s.vocab()
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
    const NAMES: [&str; 5] = ["d_count", "e_token_id", "b_chain", "c_resume", "a_free"];
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
            "b_chain" => chain::b_chain(&mut session),
            "c_resume" => chain::c_resume(&mut session),
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
