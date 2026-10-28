//! The e2e gate's clauses (macOS only).
//!
//! The session (`bloomery_metal::arch::qwen3moe::session`) opens the row's
//! MLX split (`<family>-mlxsplit` under the fixture root), and every clause
//! starts from its reset. The gate runs over a row (`--row`, `row.rs`): the
//! qwen35moe split by default, or the qwen3moe split. A clause reads the
//! file's numbers from the session's geometry and layers; a row names the
//! clauses whose contract its file's architecture lacks (a delta layer's
//! recurrent state), and the gate prints each as that clause's SKIP line
//! with its reason. The kernel mutants (`--mutant`, `--mutants-all`) run on
//! the default row.
//!
//! The clauses that compared this session to ik's GGUF sets are gone, and so
//! is the `--nodes` walk of those sets.
//! PIN(2026-10-28): removed, user decision, Metal verifies MLX only; the GGUF path runs unverified
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
//! - `p_profile`: the profile step (`Session::step_profile`) against the
//!   step it instruments, within one segment of keys and past it: the picks
//!   and last logits equal bit for bit over the same ids, one sample a
//!   group covering every group the count counts, and the groups' weight
//!   bytes equal to the file's own tensors walked under the plan's names.
//! - `m_mlx`: the row's MLX split (`<family>-mlxsplit` under the fixture root,
//!   every typed site MLX_A4, its routers MLX_A8) opens through the session —
//!   the reader admitting its sites through Metal's site lists — and
//!   `generate` over `Plain` emits the session's own host-fed picks (`mlx.rs`).
//! - `m_logits`: the split's logit rows at each of the committed mlx-lm dump's
//!   ids, teacher-forced one id a step, against the dump of the row's folder; the
//!   verdict names each step's distance and the worst (`mlx.rs`).
//! - `m_mlx_folder`: the row's MLX checkpoint folder (`<family>-mlx` under the
//!   fixture root), opened as it is with no `prepare`, tells what the split made
//!   of it (`<family>-mlxsplit`) tells: the plan's lines, the first host-fed
//!   ids and the first step's logits row, bit for bit.
//!
//! - `q_short`, `q_state`, `q_chunk`, `q_free`, `q_fault`, `q_layers`,
//!   `q_mark`, `q_mark_cut`, `q_marked`: the batched prompt pass (`pass.rs`): a prompt of at
//!   most a step's window is the step path; one pass against the stepped ids
//!   leaves layer 0's state and ring within a derived band; the pass size moves
//!   no bit; the greedy tokens teacher-forced through both; an id past the vocabulary in a pass raises the
//!   token-id site and poisons; a pass is one command buffer a layer and leaves
//!   the profiled pass's bits; a pass's copies at rows inside it move none of its
//!   bits and are its state there, and a prompt cut back to a mark inside its
//!   pass continues the same every time, within the teacher-forced band; a
//!   prompt call with a mark inside it is a pass that leaves the unmarked
//!   call's bits, on a file with a recurrent store or without one.
//! - `q_tensor`: `q_free` with the pass's GEMMs forced onto the tensor-op
//!   kernels, on any device that builds them (`pass.rs`).
//!
//! - `r_resident`: the session's kernel buffers are held in one residency set
//!   (`resident.rs`): the set holds every byte the open and the first pass
//!   allocated, and a submit after an idle gap renews its request once.
//!
//! `--dump-layers DIR` is a tool: the first host-fed step's residual after
//! each op, read back through the traced step's probe (`layers.rs`).
//! `--ops` is a tool: each op's wall in the traced step,
//! the host-fed and chain step medians beside it (`wall.rs`). `--list` prints the clause names; `--only CLAUSE` runs one;
//! `--mutant NAME` runs the clause that must catch a kernel defect against
//! kernels compiled with it (`mutants.rs`); `--mutants-all` runs each in a
//! child gate and is red when one stays green. A kept mutant, one no staying
//! clause is expected to catch, runs over every clause: `--mutants-all` prints
//! what turned red and `m_logits`' distance, and is red when one turns a clause
//! red (the kept list is stale).
//!
//! Exit 1 when a clause failed, 78 when this machine cannot run the backend,
//! 64 on a bad argument, 0 otherwise.

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::PathBuf;
use std::process::{Command, ExitCode};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

use bloomery_device::DeviceError;
use bloomery_device::conformance::{Clause, Report, Verdict, why};
use bloomery_device::fault::{Fault, FaultSite};
use bloomery_metal::arch::qwen3moe::count::{Layer, LiveKeys, Ticket, token_count};
use bloomery_metal::arch::qwen3moe::session::{Session, SessionError, StepMode};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::instance::Instance;
use bloomery_metal::library::{Library, sources_of};

#[path = "chain.rs"]
mod chain;
#[path = "flip.rs"]
mod flip;
#[path = "keep.rs"]
mod keep;
#[path = "layers.rs"]
mod layers;
#[path = "mlx.rs"]
mod mlx;
#[path = "../shared/mlx_split.rs"]
mod mlx_split;
#[path = "mutants.rs"]
mod mutants;
#[path = "pass.rs"]
mod pass;
#[path = "profile.rs"]
mod profile;
#[path = "resident.rs"]
mod resident;
#[path = "row.rs"]
mod row;
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

/// The row's MLX split, the file the session opens.
fn model_path() -> Result<PathBuf, String> {
    mlx_split::path(row::current().arch)
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
    let before = device.allocated_bytes().map_err(why)?;
    let host = d_run(s, StepMode::HostFed)?;
    let stats0 = s.stats();
    let chain = d_run(s, StepMode::Chain)?;
    let stats = chain::moved(s, stats0);
    let after = device.allocated_bytes().map_err(why)?;
    if after != before {
        return Err(format!(
            "the device's allocated bytes moved from {before} to {after} across {} steps: a \
             step allocated",
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
/// source with a mutant's edits in. A tensor-op mutant leaves it the
/// device's: its edits are in the program `q_tensor` forces.
fn library_of(
    device: &MetalDevice,
    mutant: Option<(&mutants::Mutant, mutants::Program)>,
) -> Result<Option<Library>, String> {
    match mutant {
        Some((m, mutants::Program::Main)) => {
            let text = mutants::source_of(m, mutants::Program::Main)?;
            Library::from_source_text(device, &text)
                .map(Some)
                .map_err(why)
        }
        _ => Ok(None),
    }
}

/// The tensor-op program `q_tensor` forces: the shipped one of the file's
/// instance, or a tensor-op mutant's.
fn tensor_text_of(
    instance: &'static Instance,
    mutant: Option<(&mutants::Mutant, mutants::Program)>,
) -> Result<String, String> {
    match mutant {
        Some((m, mutants::Program::Tensor)) => mutants::source_of(m, mutants::Program::Tensor),
        _ => sources_of(instance)
            .map(|(_, tensor)| tensor.to_string())
            .map_err(why),
    }
}

pub fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(name) = ["--only", "--mutant", "--row", "--dump-layers"]
        .into_iter()
        .find(|name| args.last().is_some_and(|last| last == name))
    {
        println!("{GATE}: {name} takes a value");
        return ExitCode::from(64);
    }
    const NAMES: [&str; 19] = [
        "d_count",
        "e_token_id",
        "b_chain",
        "c_resume",
        "p_profile",
        "m_mlx",
        "m_mlx_folder",
        "m_logits",
        "q_short",
        "q_state",
        "q_chunk",
        "q_free",
        "q_tensor",
        "q_fault",
        "q_layers",
        "q_mark",
        "q_mark_cut",
        "q_marked",
        "r_resident",
    ];
    if args.iter().any(|a| a == "--list") {
        for name in NAMES {
            println!("{name}");
        }
        return ExitCode::SUCCESS;
    }
    let row = match flag(&args, "--row") {
        None => &row::QWEN35MOE,
        Some(name) => match row::find(name) {
            Some(r) => r,
            None => {
                let rows: Vec<&str> = row::ROWS.iter().map(|r| r.name).collect();
                println!("{GATE}: no row named {name}; the rows are {rows:?}");
                return ExitCode::from(64);
            }
        },
    };
    row::select(row);
    let default_row = std::ptr::eq(row, &row::QWEN35MOE);
    if args.iter().any(|a| a == "--mutants-all") {
        if !default_row {
            println!(
                "{GATE}: --mutants-all: the kernel mutants edit the {} row's program",
                row::QWEN35MOE.name
            );
            return ExitCode::from(64);
        }
        return mutants_all();
    }
    let mutant = match flag(&args, "--mutant") {
        Some(name) => match mutants::find(name) {
            Some(m) if default_row => Some(m),
            Some(_) => {
                println!(
                    "{GATE}: --mutant {name}: the kernel mutants edit the {} row's program",
                    row::QWEN35MOE.name
                );
                return ExitCode::from(64);
            }
            None => {
                println!("{GATE}: no mutant named {name}");
                return ExitCode::from(64);
            }
        },
        None => None,
    };
    // A kept mutant has no clause that must catch it: every clause runs, or the one `--only` names.
    let only = match mutant {
        Some((m, _)) if !mutants::kept(m.name) => Some(m.clause),
        _ => flag(&args, "--only"),
    };
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
    if args.iter().any(|a| a == "--nodes" || a == "--set") {
        println!(
            "{GATE}: --nodes compared the session to ik's GGUF sets and was removed: Metal verifies MLX only; the GGUF path runs unverified"
        );
        return ExitCode::from(64);
    }
    let ctx = CTX;
    // The library is the instance the file's shape selects.
    let instance = match Session::instance_of(&path, ctx) {
        Ok(instance) => instance,
        Err(e) => {
            println!("clause open: FAIL ({})", says(e));
            return ExitCode::FAILURE;
        }
    };
    let device = match MetalDevice::open(instance) {
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
    let mutated = match library_of(&device, mutant) {
        Ok(lib) => lib,
        Err(e) => {
            println!("{GATE}: the mutant does not build: {e}");
            return ExitCode::from(64);
        }
    };
    let library = mutated.as_ref().unwrap_or_else(|| device.library());
    let tensor_text = match tensor_text_of(instance, mutant) {
        Ok(text) => text,
        Err(e) => {
            println!("{GATE}: the mutant does not build: {e}");
            return ExitCode::from(64);
        }
    };
    if let Some((m, _)) = mutant {
        println!("mutant: {} ({})", m.name, m.why);
    }
    let t = Instant::now();
    let mut session = match Session::open(&device, library, &path, ctx) {
        Ok(s) => s,
        Err(e) => {
            println!("clause open: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "session: {} ({} layers, ctx {ctx}) open in {:.1} s; row {}, instance {}; token {:?}",
        path.display(),
        session.layer_count(),
        t.elapsed().as_secs_f64(),
        row.name,
        instance.name,
        session.token_layers()
    );
    let delta_layers = session
        .token_layers()
        .iter()
        .filter(|l| matches!(l, Layer::Delta { .. }))
        .count();
    if (delta_layers > 0) != row.recurrent {
        println!(
            "clause open: FAIL (row {} says recurrent = {}; the file has {delta_layers} delta \
             layers)",
            row.name, row.recurrent
        );
        return ExitCode::FAILURE;
    }
    if let Some(dir) = flag(&args, "--dump-layers") {
        if mutant.is_some() {
            println!("{GATE}: --dump-layers reads the unmutated session");
            return ExitCode::from(64);
        }
        return layers::run(&mut session, std::path::Path::new(dir));
    }
    if args.iter().any(|a| a == "--ops") {
        return wall::run(&mut session);
    }
    let mut report = Report::new();
    for name in NAMES {
        if only.is_some_and(|o| o != name) {
            continue;
        }
        // The crash breadcrumb names the clause in flight: the file's last
        // line after a kernel panic is the clause that was running.
        bloomery_metal::stream::breadcrumb_clause(name);
        let result = match row.skip_of(name) {
            Some(why) => Ok(Verdict::Skip(format!("row {}: {why}", row.name))),
            None => match name {
                "d_count" => d_count(&mut session, &device),
                "e_token_id" => e_token_id(&mut session),
                "b_chain" => chain::b_chain(&mut session),
                "c_resume" => chain::c_resume(&mut session),
                "p_profile" => profile::p_profile(&device, library, &path),
                "m_mlx" => mlx::m_mlx(&device, library),
                "m_mlx_folder" => mlx::m_mlx_folder(&device, library),
                "m_logits" => mlx::m_logits(&device, library),
                "q_short" => pass::q_short(&device, library, &path),
                "q_state" => pass::q_state(&device, library, &path),
                "q_chunk" => pass::q_chunk(&device, library, &path),
                "q_free" => pass::q_free(&device, library, &path),
                "q_tensor" => pass::q_tensor(&device, library, &tensor_text, &path),
                "q_fault" => pass::q_fault(&device, library, &path),
                "q_layers" => pass::q_layers(&device, library, &path),
                "q_mark" => pass::q_mark(&device, library, &path),
                "q_mark_cut" => pass::q_mark_cut(&device, library, &path),
                "q_marked" => pass::q_marked(&device, library, &path),
                "r_resident" => resident::r_resident(&device, library, &path),
                other => Err(format!("no clause named {other}")),
            },
        };
        report.clause(name, result);
    }
    let scope = match (mutant, only) {
        (Some((m, _)), _) => format!(" (mutant {})", m.name),
        (None, Some(name)) => format!(" (only {name})"),
        (None, None) => String::new(),
    };
    report.finish(&format!("{GATE}{scope}"))
}

/// What a child gate run printed of its clauses.
struct Observed {
    /// Each clause's name and verdict word (`PASS`, `FAIL`, `SKIP`), in order.
    verdicts: Vec<(String, String)>,
    /// The `open` line's text: the session did not open.
    open: Option<String>,
    /// What `m_logits` said, its verdict word and note.
    logits: Option<String>,
}

impl Observed {
    /// The clause lines of a child's stdout.
    fn of(text: &str) -> Self {
        let mut seen = Observed {
            verdicts: Vec::new(),
            open: None,
            logits: None,
        };
        for line in text.lines() {
            let Some((name, said)) = line
                .strip_prefix("clause ")
                .and_then(|rest| rest.split_once(": "))
            else {
                continue;
            };
            if name == "open" {
                seen.open = Some(said.to_string());
                continue;
            }
            if name == "m_logits" {
                seen.logits = Some(said.to_string());
            }
            let word = said.split(' ').next().unwrap_or(said);
            seen.verdicts.push((name.to_string(), word.to_string()));
        }
        seen
    }

    /// The clauses that failed.
    fn red(&self) -> Vec<&str> {
        self.verdicts
            .iter()
            .filter(|(_, word)| word == "FAIL")
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// `; m_logits <its line>`, or nothing when the child did not run it.
    fn logits_note(&self) -> String {
        match &self.logits {
            Some(said) => format!("; m_logits {said}"),
            None => String::new(),
        }
    }
}

/// `--mutant NAME` in a child run of this gate.
fn run_child(exe: &std::path::Path, name: &str) -> Result<std::process::Output, String> {
    Command::new(exe)
        .args(["--mutant", name])
        .output()
        .map_err(|e| format!("mutant {name}: the child gate did not start: {e}"))
}

/// Every mutant of the table, each in a child run of this gate
/// (`--mutant NAME`): its clause must be red and the child's exit code not 0.
/// Each kept mutant runs over every clause and the line says what turned red
/// and `m_logits`' distance; a kept mutant that turns a clause red makes the
/// run fail, since the list is then stale, and so does one whose child
/// observed nothing.
fn mutants_all() -> ExitCode {
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(e) => {
            println!("{GATE} --mutants-all: FAIL (the gate's own path: {e})");
            return ExitCode::FAILURE;
        }
    };
    let mut green = Vec::new();
    let mut count = 0usize;
    for (mutant, _) in mutants::all() {
        count += 1;
        let out = match run_child(&exe, mutant.name) {
            Ok(out) => out,
            Err(e) => {
                println!("{e}");
                return ExitCode::FAILURE;
            }
        };
        let text = String::from_utf8_lossy(&out.stdout);
        let red =
            !out.status.success() && text.contains(&format!("clause {}: FAIL", mutant.clause));
        println!(
            "mutant {}: {} on {}{}",
            mutant.name,
            if red { "red" } else { "NOT RED" },
            mutant.clause,
            Observed::of(&text).logits_note()
        );
        if !red {
            green.push(mutant.name);
        }
    }
    let mut stale = Vec::new();
    let mut unseen = Vec::new();
    for mutant in mutants::KEPT {
        let out = match run_child(&exe, mutant.name) {
            Ok(out) => out,
            Err(e) => {
                println!("{e}");
                return ExitCode::FAILURE;
            }
        };
        let seen = Observed::of(&String::from_utf8_lossy(&out.stdout));
        let red = seen.red();
        if seen.open.is_some()
            || seen.verdicts.is_empty()
            || (red.is_empty() && !out.status.success())
        {
            let found = match &seen.open {
                Some(said) => format!("the open line {said:?}"),
                None => "no clause line".to_string(),
            };
            println!(
                "mutant {}: no observation, the child gate ended with {} and {found}",
                mutant.name, out.status
            );
            unseen.push(mutant.name);
        } else if red.is_empty() {
            println!(
                "mutant {}: kept, red on none of {} clauses{}",
                mutant.name,
                seen.verdicts.len(),
                seen.logits_note()
            );
        } else {
            println!(
                "mutant {}: now red on {} (nearest clause {}), the KEPT list is stale{}",
                mutant.name,
                red.join(", "),
                mutant.clause,
                seen.logits_note()
            );
            stale.push(mutant.name);
        }
    }
    if green.is_empty() && stale.is_empty() && unseen.is_empty() {
        println!(
            "{GATE} --mutants-all: PASS ({count} mutants, each red on its clause; {} kept, \
             no clause turns red on any)",
            mutants::KEPT.len()
        );
        return ExitCode::SUCCESS;
    }
    let mut why = Vec::new();
    if !green.is_empty() {
        why.push(format!("not red: {}", green.join(", ")));
    }
    if !stale.is_empty() {
        why.push(format!(
            "the KEPT list is stale, now red: {}",
            stale.join(", ")
        ));
    }
    if !unseen.is_empty() {
        why.push(format!("no observation: {}", unseen.join(", ")));
    }
    println!("{GATE} --mutants-all: FAIL ({})", why.join("; "));
    ExitCode::FAILURE
}
