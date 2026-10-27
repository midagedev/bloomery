//! The e2e gate's clauses (macOS only).
//!
//! The session (`bloomery_metal::arch::qwen3moe::session`) opens the file
//! the fixture family runs (`refset::fixture::first_shard`), the file its
//! reference sets were dumped from, and every clause starts from its reset.
//! The gate runs over a row (`--row`, `row.rs`): the qwen35moe fixture by
//! default, or the qwen3moe fixture. A clause reads the file's numbers from
//! the session's geometry and layers; a row names the clauses whose contract
//! its file's architecture lacks (a delta layer's recurrent state), and the
//! gate prints each as that clause's SKIP line with its reason. The kernel
//! mutants (`--mutant`, `--mutants-all`) and the `--nodes` walk run on the
//! default row.
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
//! - `h_prompt`: the prompt loop's known-id run-ahead (`ahead.rs`): every
//!   id but the last of each checkpoint span steps the next id ahead with
//!   no head on a row whose pick nobody reads; the prompt's pick, logits,
//!   recurrent state, KV rows and greedy picks match the host-fed loop's
//!   bit for bit with a mark crossed and without, `StepStats` shows the ids
//!   ahead all kept, and a checkpoint with a token ahead drops it and keeps
//!   the point clean.
//! - `a_free`: the batch set's tokens host-fed one at a time; every layer's
//!   output over the rows ik kept within [`free::FREE_BAND`] of ik's, the
//!   last position's argmax as the fixture tier takes it, the rows a
//!   near-tie router flip carries left unheld by the CUDA gate's rule; and
//!   the traced run's logits and picks equal to the plain steps'.
//! - `s_serve`: the server's engine over a session of its own: greedy picks,
//!   a sampled step's row and the keep rule held to this session's host-fed
//!   steps (`serve.rs`).
//! - `k_keep`: checkpoints of the delta state (`keep.rs`). A prompt leaves
//!   points at `seqstate::marks`; a cut back to one matches a fresh prompt,
//!   including while a token runs ahead; a cut where no point stands is
//!   refused; the engine keeps a prefill's end.
//! - `t_target`: the session as the generation loop's target
//!   (`runtime::Target`): `generate` over `Plain` and a count stop emits the
//!   plain steps' own greedy picks with every chained token kept; a
//!   `Want::Logits` step's row equals `Node::Logits` after the same step and
//!   its argmax is the row's own; the target's prompt leaves the stepped
//!   run's position and last logits; the refusals (an empty prompt, a prompt
//!   past the cache, a cut between 0 and the position) by name with the
//!   session stepping on, the cut rule's `kept` table, and no host
//!   allocation call in a steady step of either read-back.
//! - `g_generate`: the generation core (`arch::qwen3moe::generate`) the CLI
//!   runs: greedy over the batch set's first ids emits the session's own
//!   picks, the count stop ends a run at its budget, an end-id set holding
//!   the pick at a step ends the run at that step, and the sink sees every
//!   id in order.
//! - `f_deep`: the d1k set's prompt host-fed one id at a time through a
//!   session of the set's depth, past one segment of keys, then its step
//!   token traced; every step's count by its live keys (one wait a step, one
//!   barrier between groups), the device's allocation unmoved and no host
//!   allocation call in a steady step, every layer's step output within
//!   [`free::FREE_BAND`] of ik's and the head's pick by its tie rule.
//! - `p_profile`: the profile step (`Session::step_profile`) against the
//!   step it instruments, within one segment of keys and past it: the picks
//!   and last logits equal bit for bit over the same ids, one sample a
//!   group covering every group the count counts, and the groups' weight
//!   bytes equal to the file's own tensors walked under the plan's names.
//! - `m_mlx`: the MLX-format fixture split (`qwen35moe-mlxsplit` under the
//!   fixture root, every typed site MLX_A4, its routers MLX_A8) opens
//!   through the session — the reader admitting its sites through Metal's
//!   site lists — and `generate` over `Plain` emits the session's own
//!   host-fed picks (`t_target`'s greedy pattern on that session).
//!
//! - `q_short`, `q_state`, `q_chunk`, `q_ref`, `q_free`, `q_fault`, `q_layers`,
//!   `q_mark`, `q_mark_cut`, `q_marked`: the batched prompt pass (`pass.rs`): a prompt of at
//!   most a step's window is the step path; one pass against the stepped ids
//!   leaves layer 0's state and ring within a derived band; the pass size moves
//!   no bit; the step at the prompt's depth against ik's `l_out` and logits, the
//!   pass run's distance over the stepped run's within a ratio; the greedy tokens
//!   teacher-forced through both; an id past the vocabulary in a pass raises the
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
//! `--nodes` is a tool, not a clause: its set's prompt host-fed, then its
//! step token traced, every named node against ik's, the first past its
//! band printed (`--set d1k` walks the d1k set at its depth; the step set is
//! the default). `--ops` is another: each op's wall in the traced step,
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

use bloomery_device::DeviceError;
use bloomery_device::conformance::{Clause, Report, Verdict, why};
use bloomery_device::fault::{Fault, FaultSite};
use bloomery_metal::arch::qwen3moe::count::{Layer, LiveKeys, Ticket, token_count};
use bloomery_metal::arch::qwen3moe::session::{Session, SessionError, StepMode};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::instance::Instance;
use bloomery_metal::library::{Library, sources_of};

#[path = "ahead.rs"]
mod ahead;
#[path = "chain.rs"]
mod chain;
#[path = "deep.rs"]
mod deep;
#[path = "flip.rs"]
mod flip;
#[path = "free.rs"]
mod free;
#[path = "generate.rs"]
mod generate;
#[path = "keep.rs"]
mod keep;
#[path = "mlx.rs"]
mod mlx;
#[path = "../shared/mlx_split.rs"]
mod mlx_split;
#[path = "mutants.rs"]
mod mutants;
#[path = "nodes.rs"]
mod nodes;
#[path = "pass.rs"]
mod pass;
#[path = "profile.rs"]
mod profile;
#[path = "resident.rs"]
mod resident;
#[path = "row.rs"]
mod row;
#[path = "serve.rs"]
mod serve;
#[path = "sets.rs"]
mod sets;
#[path = "target.rs"]
mod target;
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

/// The file the row's fixture family runs: the one its sets were dumped from.
fn model_path() -> Result<PathBuf, String> {
    refset::fixture::first_shard(row::current().arch)
        .map(PathBuf::from)
        .map_err(|e| format!("the fixture file: {e}"))
}

/// The cache rows the `--nodes` walk needs: the step set's depth within one
/// segment, the d1k set's prompt plus its step token.
fn nodes_ctx(set: sets::Set) -> Result<usize, String> {
    match set {
        sets::Set::Step => Ok(CTX),
        sets::Set::D1k => {
            let man = sets::open(set)?;
            let (pos, step, _) = man
                .step()
                .map_err(|e| format!("the d1k set's tokens: {e}"))?;
            Ok(pos as usize + step.len())
        }
        sets::Set::Batch => Err("--nodes walks a step set".to_string()),
    }
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
    if let Some(name) = ["--only", "--mutant", "--row"]
        .into_iter()
        .find(|name| args.last().is_some_and(|last| last == name))
    {
        println!("{GATE}: {name} takes a value");
        return ExitCode::from(64);
    }
    const NAMES: [&str; 25] = [
        "d_count",
        "e_token_id",
        "b_chain",
        "c_resume",
        "h_prompt",
        "a_free",
        "t_target",
        "s_serve",
        "k_keep",
        "g_generate",
        "f_deep",
        "p_profile",
        "m_mlx",
        "q_short",
        "q_state",
        "q_chunk",
        "q_ref",
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
    let only = mutant.map(|(m, _)| m.clause).or(flag(&args, "--only"));
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
    let nodes = args.iter().any(|a| a == "--nodes");
    if nodes && !default_row {
        println!(
            "{GATE}: --nodes walks the {} row's nodes",
            row::QWEN35MOE.name
        );
        return ExitCode::from(64);
    }
    let nodes_set = match flag(&args, "--set") {
        None => sets::Set::Step,
        Some(name) => match name {
            "step4" => sets::Set::Step,
            "d1k" => sets::Set::D1k,
            other => {
                println!("{GATE}: --set takes step4 or d1k, not {other}");
                return ExitCode::from(64);
            }
        },
    };
    if !nodes && flag(&args, "--set").is_some() {
        println!("{GATE}: --set names the --nodes walk's set");
        return ExitCode::from(64);
    }
    let ctx = match if nodes { nodes_ctx(nodes_set) } else { Ok(CTX) } {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("clause open: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
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
    if nodes {
        return nodes::run(&mut session, nodes_set);
    }
    if args.iter().any(|a| a == "--ops") {
        return wall::run(&mut session);
    }
    let mut report = Report::new();
    for name in NAMES {
        if only.is_some_and(|o| o != name) {
            continue;
        }
        let result = match row.skip_of(name) {
            Some(why) => Ok(Verdict::Skip(format!("row {}: {why}", row.name))),
            None => match name {
                "d_count" => d_count(&mut session, &device),
                "e_token_id" => e_token_id(&mut session),
                "b_chain" => chain::b_chain(&mut session),
                "c_resume" => chain::c_resume(&mut session),
                "h_prompt" => ahead::h_prompt(&device, library, &path),
                "t_target" => target::t_target(&mut session),
                "s_serve" => serve::s_serve(&mut session, &device, library, &path),
                "k_keep" => keep::k_keep(&device, library, &path),
                "g_generate" => generate::g_generate(&mut session),
                "f_deep" => deep::f_deep(&device, library, &path),
                "p_profile" => profile::p_profile(&device, library, &path),
                "m_mlx" => mlx::m_mlx(&device, library),
                "q_short" => pass::q_short(&device, library, &path),
                "q_state" => pass::q_state(&device, library, &path),
                "q_chunk" => pass::q_chunk(&device, library, &path),
                "q_ref" => pass::q_ref(&device, library, &path),
                "q_free" => pass::q_free(&device, library, &path),
                "q_tensor" => pass::q_tensor(&device, library, &tensor_text, &path),
                "q_fault" => pass::q_fault(&device, library, &path),
                "q_layers" => pass::q_layers(&device, library, &path),
                "q_mark" => pass::q_mark(&device, library, &path),
                "q_mark_cut" => pass::q_mark_cut(&device, library, &path),
                "q_marked" => pass::q_marked(&device, library, &path),
                "r_resident" => resident::r_resident(&device, library, &path),
                _ => free::a_free(&mut session),
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
    let mut count = 0usize;
    for (mutant, _) in mutants::all() {
        count += 1;
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
        println!("{GATE} --mutants-all: PASS ({count} mutants, each red on its clause)");
        return ExitCode::SUCCESS;
    }
    println!("{GATE} --mutants-all: FAIL (not red: {})", green.join(", "));
    ExitCode::FAILURE
}
