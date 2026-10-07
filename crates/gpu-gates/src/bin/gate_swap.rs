//! The residency machine's gate (`bloomery_gpu::host::swap`): synthetic
//! stacks and a synthetic source, no model file, as `gate_hybrid`'s refusal
//! arm builds its own boundary.
//!
//! The model: layers 2..6 of 32 experts, top-4, with 12, 12, 10 and 0 card
//! slots (the last layer has none and takes no part); each expert has three
//! parts of 1024, 1024 and 1536 bytes, random words from its (layer, id,
//! part). One captured graph per arm launches a probe a (layer, part): each
//! routed id reads its place in the card's copy of the map, as the handoff
//! does, and a card place's slot's words through the stack, as the `_sel`
//! launches do, into one sum. The host serves every id the host map sends
//! it with the source's sum under another rule, and the tier every id the
//! host map puts on the tier under a third, so the pass's value depends on
//! which experts the card holds, as the engine's bits do. A pass's value is
//! the FNV-1a of every routed id's (id, side, parts); a card sum that is not
//! its expert's is `stale`, an id two sides serve `double`, one no side
//! serves `miss`.
//!
//! The trace: 120 passes of the synthetic router (a hot set of 8 drifting
//! every 20 passes, never the seed's first 4), every fifth pass two rows of
//! which the first is kept. The rule: every 2 kept rows, cap 6, margin 1,
//! min count 2, decay 0.9, one spare, live delay 3, 4 pinned seed experts a
//! layer.
//!
//! Each pass runs in the engine's order: the ids refreshed by a
//! synchronizing copy, the boundary, the graph, then the engine stream
//! drained by polling with a bound (a stream waiting on a copy nothing
//! stages is a named failure, not a hang) and read back.
//!
//! Clauses (the residency design's bit contract c1–c7, and the machine's own):
//! - c1: the same history twice with the copies prompt, and once with the
//!   copy stream held (c3's run), gives the same pass values and the same
//!   flips, and the run makes flips on every card layer.
//! - c2: the adaptive run equals, pass for pass, a static replay that puts
//!   each pass's card sets in place with synchronous copies into slots by
//!   ascending id (other slot numbers than the machine's) and a fresh map.
//! - c3: with the staging window closed for the whole run, a copy is staged
//!   only once its flip is due (its live boundary reached), so every landing
//!   is late; no card sum is stale, no id served twice or not at all.
//! - c3 engine: with the machine's copy stream held by a host word from the
//!   boundary that issues a flip until after the launch of the pass it lands
//!   at, the engine stream is still waiting when the host looks, and after
//!   the release the pass reads no stale sum (its mutant: no engine stream
//!   wait on the flip's event).
//! - c3 boundary event: with the engine stream held through a pass that reads
//!   a victim, the next boundary lands the flip that frees the victim's slot
//!   and issues a flip into it; after the release the held pass reads no
//!   stale sum (its mutant: the copy does not wait for the boundary event).
//!   Its arm runs at live delay 2, so a landing boundary is a planning one.
//! - c4: (its mutant: the card's map word written when the flip is made) is
//!   red in c3's run.
//! - c5: every pass of every run has no double and no miss (its mutant: the
//!   host map one pass late).
//! - c6: an independent rule fed every row of the trace and ended at the kept
//!   rows makes, boundary for boundary, the flips the machine made (its
//!   mutant: the machine ends each pass at every row it noted).
//! - clock: a history of passes that keep 2 rows each plans every `every`
//!   kept rows — flips at odd boundaries too, which a pass clock at `every` 2
//!   never issues — and the machine's flips are an independent rule's over
//!   the same history (its mutant: the rule's clock counting passes).
//! - c7: a reset with flips in flight cancels them and brings every layer's
//!   card set back to its seed (`residency reset` diff=0, the live sets and
//!   the slot ledger's), releases the host pages of the seed experts it
//!   copies back and of the cancelled flips' victims that stay on the card,
//!   and the trace after it gives the fresh run's values.
//! - tier: a map with tier entries runs with flips on every card layer, and
//!   no flip admits or evicts a tier expert, though one of them is an expert
//!   the stage-only run admits; every tier entry is unchanged after the run
//!   (its mutant: the tier's experts in the rule's card set, so a victim).
//! - staging failure: a source that fails a landing flip's expert under held
//!   copies is a named error at the boundary it lands at, with no stale sum
//!   in any pass before it (its mutant: the failure not read again after the
//!   landing waits).
//! - tally: a slot noted twice and a note outside the tally's shape are
//!   refused by name, and so is a kept row with a slot missing, after which
//!   the machine takes the full row (its mutant: a note that does not check
//!   the slot's bit).
//! - ahead: the trace driven with every boundary after the first made ahead
//!   of its pass (`SwapMachine::boundary_ahead`: after the pass before it is
//!   launched and ended, before that pass's drain and readback, the engine's
//!   step order) and taken by its launch (`SwapMachine::take_ahead`) gives
//!   the prompt run's values and flips; then a boundary made ahead refuses a
//!   boundary, an end of pass and a prompt call by name until a launch takes
//!   it, the take returns it once and a second take is refused by name, and
//!   a reset drops one made ahead (its mutant: a take that leaves the
//!   boundary waiting, so the second take passes).
//! - keep: a count the tier's own `keep_rows` gives with no pass open,
//!   after a reset, is refused by name — the write side of the boundary's
//!   kept-count refusal — and nothing is stored, so the next boundary
//!   still runs (its mutant: the refusal dropped, the count stored).
//! - broken: a failure after a boundary's first change is a named error, and
//!   every later boundary, end of pass and reset is refused naming it (its
//!   mutant: the machine not marked broken).
//! - panic: a source that panics on a flip's expert is a named error at or
//!   before the boundary the flip lands at, within the deadline (its mutant:
//!   no catch on the staging thread).
//! - refusal: a flip whose victim the source cannot bring to the host is
//!   refused by name at the boundary it would land at; a load whose spare
//!   slot gives up such an expert is refused at construction.
//! - pinned: no flip evicts a pinned seed expert and each keeps its slot for
//!   the whole run, while flips evict other seed experts.
//! - fault code: the driver error the copy stream's query returns at a drop
//!   is a fault leak that carries its code, and the `residency leak` record
//!   prints it (its mutant: the code dropped, the word `fault` alone).
//! - stall: a victim that takes longer to prepare than the machine's
//!   deadline is a named error at the boundary it would land at, returned
//!   before the preparation ends; the machine dropped then leaves its
//!   staging thread to finish, never that thread's to free, and reports the
//!   leak by name.
//! - placement: a machine built on a thread pinned to one cpu runs its
//!   staging thread off that cpu.
//! - leaks: once every machine is dropped, the stall arm's is the one leak
//!   reported.
//! - queue: two arms, each a run to its last flip and a reset under a closed
//!   window (as the engine's step port leaves it between passes) that
//!   finish within [`QUEUE_BOUND`], every layer back at its seed. The light
//!   arm's source adds [`LIGHT`] command a part (a job the engine's size), and
//!   its reset copies more seed experts than flips are kept in flight: the
//!   in-flight bound refuses that by name unless the reset's copies stage as
//!   they are issued. The heavy arm, run only once the light one passes,
//!   adds [`INFLATE`] copy stream commands a part (one job then carries more
//!   than the stream queues): a job reaches the staging thread before its
//!   commands reach the copy stream, and the reset's copies go out a ring's
//!   worth at a time, so the host never blocks enqueuing behind copies
//!   nothing stages. Its mutants: the job sent after its commands; the
//!   reset's copies all enqueued with the flush turned on only after the
//!   last (the light arm's refusal, then with the bound removed the heavy
//!   arm's block). A watchdog thread names a clause past the bound, then
//!   ends the process.
//! - dropq: a machine left with a copy queued behind a staging word (the
//!   first flip boundary issued under a closed window, its jobs not due),
//!   inside a wrapper whose first field synchronizes the context on drop, as
//!   the V4.1 body's ring shadows do: the wrapper's drop stops the machine
//!   first ([`bloomery_gpu::host::HostTier::stop_swap`]'s order), so the
//!   whole drop ends within the machine's deadline with no leak. Its mutant:
//!   the wrapper without that drop, today's field order, which waits in the
//!   synchronize for ever — the queue arm's watchdog names it and ends the
//!   process.
//! - dropq (tier): the same machine inside a real [`HostTier`] — a boundary,
//!   a seed map, a started machine, the passes driven through the tier's own
//!   `keep_rows` and `swap_boundary` — dropped as the engine drops the tier:
//!   its `Drop` stops the machine before any field frees, so the drop ends
//!   within the deadline with no leak. Its mutant: production `stop_swap`'s
//!   body emptied, which leaves the drop freeing the boundary's buffers
//!   while the copy still waits — the clause's watchdog names it.
//! - dropq (free): a plain [`DeviceBuffer`] free with the same copy queued,
//!   timed on its own thread: whether a free blocks behind a copy nothing
//!   stages, which the stop-before-any-free order rests on. A measurement,
//!   not a verdict — the line names the outcome either way.
//!
//! Calls (the prompt call mode, `SwapMachine::begin_call` .. `end_call`):
//! an arm runs 40 passes of the trace, then a call of 8 steps, each step a
//! pick of every layer from its ids' counts (floor 1) and a probe after the
//! engine stream waits for each layer's landed event, each layer's reader
//! after the readback; every expert is host-resident from the load (the
//! churn pool is in the host set).
//! - s1 scripted: a kept call's step values equal a static replay of the
//!   card sets its picks left, no pick is left landing at its end, and the
//!   passes after it run clean; a call not kept returns every layer to its
//!   start set, and the passes after it run clean (its mutant: the host map
//!   admits each expert into another pair's slot).
//! - s2 landing wait: with the copy stream held over the first step's picks
//!   (two wanted a layer), the engine stream is still waiting when the arm
//!   looks after the probe's launch, and no step reads a stale sum (its
//!   mutant: the landed event recorded before the pick's copies).
//! - s3 host map at the pick: no step of the kept call serves an id twice or
//!   not at all (its mutant: the host map moves at the layer's reader, one
//!   layer late).
//! - s4 victim refusal: a pick whose victim the source cannot serve from
//!   resident pages is refused by name, the machine unbroken and the host
//!   map unmoved (its mutant: the victims not checked).
//! - s5 floor: a call whose floor is raised between two picks admits no
//!   expert whose count is under the new floor, and a floor set with no call
//!   open is refused by name (its mutant: the pick reads the floor the call
//!   began with, not the one set since).
//!
//! Evicted (a host set populated and not locked, whose pages the page cache
//! lets go): a victim not host-resident when the machine decides is read
//! back in and the flip goes on — at a boundary whose landing victims the
//! staging thread prepared and lost again, at a call's pick and at the end
//! of a call not kept — each arm running to its end with the values, card
//! sets and flips of its twin without the fault (its mutant: no prepare in
//! the machine's one decision, which refuses each by name).

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_swap: built without the `gpu` feature; see `just gate-gpu-swap`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_swap", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use std::ops::Range;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::sync::{Arc, Mutex, mpsc};
    use std::time::{Duration, Instant};

    use bloomery_gpu::host::PassKind;
    use bloomery_gpu::host::slots::{HOST, Slot, SlotMap};
    use bloomery_gpu::host::swap::{
        CallCfg, CallPick, CallReport, Leak, LeakReason, MachineCfg, PassReport, Piece, SlotState,
        SwapMachine, SwapSource, Transform, set_leak_sink,
    };
    use bloomery_gpu::hybrid::{Boundary, BoundaryShape, HostExperts, HostTier};
    use bloomery_gpu::{DeviceTensor, Gpu, GpuError, Graph, HostFlags};
    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError, LaunchConfig1D, sys};
    use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
    use cuda_host::cuda_module;
    use model::Tensor2;
    use runtime::swaprule::{Flip, KeptRows, Shape, SwapParams, SwapRule};

    const LAYERS: Range<usize> = 2..6;
    const L: usize = LAYERS.end - LAYERS.start;
    const E: usize = 32;
    const K: usize = 4;
    const MAX_ROWS: usize = 2;
    const NIDS: usize = MAX_ROWS * K;
    const N_L: [usize; L] = [12, 12, 10, 0];
    const PART_BYTES: [usize; 3] = [1024, 1024, 1536];
    const PARTS: usize = PART_BYTES.len();
    /// Bytes one expert takes over its parts.
    const EXPERT_BYTES: u64 = 1024 + 1024 + 1536;
    const PINNED: usize = 4;
    const PASSES: usize = 120;
    /// The machine's bound on every host wait in the arms that must pass.
    const DEADLINE: Duration = Duration::from_secs(10);
    /// The stall arm's bound, and how long its slow victim takes to prepare.
    const STALL_DEADLINE: Duration = Duration::from_millis(300);
    const SLOW: Duration = Duration::from_secs(2);
    /// How soon a dropped machine's copy stream must drain; a copy it left
    /// waiting on the slow victim's ticket never does.
    const RELEASED: Duration = Duration::from_millis(500);
    /// How long after its drop the stall arm's staging thread may take to
    /// end: the slow preparation it was left inside, and its exit.
    const JOINED: Duration = Duration::from_secs(4);
    /// The gate's own bound on a pass's engine stream.
    const ENGINE_DRAIN: Duration = Duration::from_secs(30);
    /// How long a held arm lets the unheld side run before it looks: every
    /// copy and launch here takes microseconds.
    const HOLD_SETTLE: Duration = Duration::from_millis(200);
    const DELAY: u64 = 3;
    /// Copy stream commands the queue arm's source adds to each part it
    /// copies: one job then carries thousands and a reset's seed copies tens
    /// of thousands, past what a stream queues before an enqueue blocks.
    const INFLATE: usize = 2048;
    /// The light queue arm's commands a part: a job then carries the engine's
    /// size of work (its staging wait, three copies, a command a part, the
    /// drained write), far under what a stream queues, so the in-flight bound
    /// in the reset's copies is what refuses them, not the driver.
    const LIGHT: usize = 1;
    /// The queue arm's bound on its run and reset, which take a few seconds
    /// when every copy stages as it is issued.
    const QUEUE_BOUND: Duration = Duration::from_secs(20);
    /// How long the queue arm's watchdog waits after naming a blocked clause
    /// before it ends the process: a stack watch (`BLOOMERY_GATE_STACKS`
    /// under this) dumps the blocked threads in between.
    const QUEUE_GRACE: Duration = Duration::from_secs(60);
    /// The boundary-event arm's live delay: with `every` 2 a flip lands at a
    /// planning boundary.
    const DELAY_EVEN: u64 = 2;
    /// The tier arm's boundary width: nothing serves on it, it only has to
    /// allocate and free as the engine's does.
    const TIER_HIDDEN: usize = 256;
    /// How long the free arm looks at a plain free with a copy queued behind
    /// an unstaged job before it releases the machine behind it: past this
    /// the free is blocked, and only the machine's drop ends the block.
    const FREE_LOOK: Duration = Duration::from_secs(2);
    /// The host's rule: its sum is the card's under this mask, so a pass's
    /// value says which side served each id.
    const HOST_MASK: u32 = 0x1234_5678;
    /// The tier's rule, as the host's.
    const TIER_MASK: u32 = 0x0f0f_a5a5;
    /// A probe output for an id the card's map sends to the host.
    const OUT_HOST: u32 = u32::MAX;
    /// A probe output for a place past the stack.
    const OUT_PAST: u32 = u32::MAX - 1;
    /// A probe output for an id past the experts.
    const OUT_BAD_ID: u32 = u32::MAX - 2;

    #[cuda_module]
    mod probe_kernels {
        use super::*;

        /// One routed id a thread, `n_ids` of them: id `j` of layer `li` of
        /// `n_layers` in `ids` (row `j / k`, slot `j % k`), its place `map[row_off
        /// + id]`, and for a card place the sum over the slot's `words` words
        /// of `word · (i + 1)`, wrapping, halved (so it never meets the three
        /// marks) — at `out[out_at + j]`. A host place writes [`OUT_HOST`], a
        /// place at or past `cap` [`OUT_PAST`], an id past `n_expert`
        /// [`OUT_BAD_ID`].
        #[allow(
            clippy::too_many_arguments,
            reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
        )]
        #[kernel]
        #[launch_bounds(32)]
        #[launch_contract(
            domain = 1,
            block = (32, 1, 1),
            requires = (
                map.len() >= row_off + n_expert,
                stack.len() >= cap * words,
                out.len() >= out_at + n_ids
            )
        )]
        pub fn swap_probe(
            map: &[u32],
            row_off: u32,
            n_expert: u32,
            ids: &[u32],
            li: u32,
            n_layers: u32,
            k: u32,
            n_ids: u32,
            stack: &[u32],
            cap: u32,
            words: u32,
            out_at: u32,
            mut out: DisjointSlice<u32>,
        ) {
            let j = thread::index_1d().get();
            if j >= n_ids as usize {
                return;
            }
            let k = k as usize;
            let at = ((j / k) * n_layers as usize + li as usize) * k + j % k;
            let v = if at >= ids.len() {
                OUT_BAD_ID
            } else {
                // SAFETY: at < ids.len(), checked above.
                let id = unsafe { *ids.get_unchecked(at) };
                if id >= n_expert {
                    OUT_BAD_ID
                } else {
                    // SAFETY: id < n_expert, so row_off + id < map.len() by
                    // the launch contract.
                    let place = unsafe { *map.get_unchecked(row_off as usize + id as usize) };
                    if place == HOST {
                        OUT_HOST
                    } else if place >= cap {
                        OUT_PAST
                    } else {
                        let base = place as usize * words as usize;
                        let mut sum = 0u32;
                        let mut i = 0usize;
                        while i < words as usize {
                            // SAFETY: place < cap, so base + i < cap · words
                            // <= stack.len() by the launch contract.
                            let w = unsafe { *stack.get_unchecked(base + i) };
                            sum = sum.wrapping_add(w.wrapping_mul(i as u32 + 1));
                            i += 1;
                        }
                        sum >> 1
                    }
                }
            };
            // SAFETY: out_at + j < out_at + n_ids <= out.len() by the launch
            // contract, and thread j is the word's only writer.
            unsafe { *out.get_unchecked_mut(out_at as usize + j) = v };
        }
    }

    /// Word `i` of part `p` of layer `l`'s expert `e`.
    fn word(l: usize, e: u32, p: usize, i: usize) -> u32 {
        let mut x = (l as u64) << 48 ^ u64::from(e) << 32 ^ (p as u64) << 24 ^ i as u64;
        x = x.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        x ^= x >> 29;
        x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        (x >> 32) as u32
    }

    /// The probe's sum of part `p` of layer `l`'s expert `e`, from the source.
    fn expect(l: usize, e: u32, p: usize) -> u32 {
        let mut sum = 0u32;
        for i in 0..PART_BYTES[p] / 4 {
            sum = sum.wrapping_add(word(l, e, p, i).wrapping_mul(i as u32 + 1));
        }
        sum >> 1
    }

    fn li(l: usize) -> usize {
        l - LAYERS.start
    }

    /// What an arm's source does wrong: `stuck` experts never become
    /// host-resident, `evicted` ones lose their pages after every prepare
    /// but the machine's own thread's — not resident from the load,
    /// `all_resident` or not, and a staging thread's prepare of one leaves
    /// it not resident (the page cache let it go again before the machine
    /// looked) — `slow` ones take `SLOW` to prepare, `fail_source` and
    /// `panic_source` fail or panic when their bytes are read, every stack
    /// destination of layer `fail_dest` is an error, and each part a copy
    /// moves carries `inflate` more copy stream commands. `dropped_on` gets
    /// the name of the thread that drops the source.
    #[derive(Clone, Debug, Default)]
    struct Faults {
        dropped_on: Arc<Mutex<Option<String>>>,
        stuck: Vec<(usize, u32)>,
        evicted: Vec<(usize, u32)>,
        slow: Vec<(usize, u32)>,
        fail_source: Option<(usize, u32)>,
        panic_source: Option<(usize, u32)>,
        fail_dest: Option<usize>,
        inflate: usize,
        /// Every expert host-resident from the load (a churn pool the load's
        /// host set holds), bar the stuck ones.
        all_resident: bool,
    }

    /// The synthetic source: every expert's bytes on the host; the stacks'
    /// addresses; residency per expert, the card's experts not resident at
    /// load (their pages dropped), preparing makes an expert resident unless
    /// it is stuck.
    struct Synth {
        bytes: Vec<Vec<u8>>,
        bases: Vec<[sys::CUdeviceptr; PARTS]>,
        caps: [usize; L],
        resident: Vec<AtomicBool>,
        faults: Faults,
        prepared: AtomicU32,
    }

    impl Synth {
        fn new(stacks: &Stacks, caps: [usize; L], faults: Faults) -> Synth {
            let mut bytes = Vec::with_capacity(L * E * PARTS);
            for l in LAYERS {
                for e in 0..E as u32 {
                    for (p, &n) in PART_BYTES.iter().enumerate() {
                        bytes.push(
                            (0..n / 4)
                                .flat_map(|i| word(l, e, p, i).to_le_bytes())
                                .collect(),
                        );
                    }
                }
            }
            let resident = (0..L * E)
                .map(|x| {
                    let (l, e) = (LAYERS.start + x / E, (x % E) as u32);
                    let all = faults.all_resident && !faults.stuck.contains(&(l, e));
                    let evicted = faults.evicted.contains(&(l, e));
                    AtomicBool::new(!evicted && (all || x % E >= caps[x / E]))
                })
                .collect();
            Synth {
                bytes,
                bases: stacks.bases(),
                caps,
                resident,
                faults,
                prepared: AtomicU32::new(0),
            }
        }
    }

    impl Drop for Synth {
        fn drop(&mut self) {
            let name = std::thread::current()
                .name()
                .unwrap_or("unnamed")
                .to_string();
            if let Ok(mut on) = self.faults.dropped_on.lock() {
                *on = Some(name);
            }
        }
    }

    /// Every leak a dropped machine of this process reported, in order.
    static LEAKS: Mutex<Vec<Leak>> = Mutex::new(Vec::new());

    /// The gate's leak sink: the `residency leak` record, and the leak kept
    /// for the clauses.
    fn note_leak(l: &Leak) {
        record::residency_leak(l).print();
        if let Ok(mut v) = LEAKS.lock() {
            v.push(*l);
        }
    }

    fn leaks() -> Result<Vec<Leak>, GateError> {
        Ok(LEAKS
            .lock()
            .map_err(|_| "gate_swap: the leak list")?
            .clone())
    }

    /// The tids of this process's threads named as the swap machine names
    /// its staging thread.
    fn staging_tids() -> Result<Vec<String>, GateError> {
        let mut out = Vec::new();
        for t in std::fs::read_dir("/proc/self/task")? {
            let t = t?;
            let comm = std::fs::read_to_string(t.path().join("comm"))?;
            if comm.trim_end() == "swap-staging" {
                out.push(t.file_name().to_string_lossy().into_owned());
            }
        }
        Ok(out)
    }

    /// Cpus in a kernel cpu list (`0-63`, `0,32`, `0-3,8`).
    fn list_cpus(list: &str) -> Result<usize, GateError> {
        let mut n = 0;
        for part in list.trim().split(',') {
            n += match part.split_once('-') {
                Some((a, b)) => b.parse::<usize>()? - a.parse::<usize>()? + 1,
                None => {
                    part.parse::<usize>()?;
                    1
                }
            };
        }
        Ok(n)
    }

    /// The calling thread's mask set to `cpus`.
    fn set_mask(cpus: &[usize]) -> Result<(), GateError> {
        // SAFETY: `set` is a zeroed cpu_set_t only written through `CPU_SET`
        // with indices read from a mask of the same size, and
        // `sched_setaffinity` reads it with the matching size.
        let ok = unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            for &c in cpus {
                libc::CPU_SET(c, &mut set);
            }
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0
        };
        if ok {
            Ok(())
        } else {
            Err(format!("gate_swap: sched_setaffinity to {cpus:?}").into())
        }
    }

    /// placement: a machine built on a thread pinned to one cpu (as a
    /// binary's step thread is) runs its staging thread off that cpu: the
    /// staging copies never take turns with the step. With no pool built the
    /// staging thread floats, its mask wider than the one cpu.
    fn placement(gpu: &Gpu, pm: &probe_kernels::LoadedModule) -> Result<bool, GateError> {
        let wide = threads::helper::mask()?;
        let Some(&c) = wide.first().filter(|_| wide.len() > 1) else {
            println!("placement: the gate thread's mask is {wide:?}, too narrow to tell FAIL");
            return Ok(false);
        };
        let before = staging_tids()?;
        set_mask(&[c])?;
        let built = plain(gpu, pm, Faults::default(), DELAY);
        let restored = set_mask(&wide);
        let r = built?;
        restored?;
        let tid = staging_tids()?
            .into_iter()
            .find(|t| !before.contains(t))
            .ok_or("gate_swap: the machine's staging thread")?;
        let status = std::fs::read_to_string(format!("/proc/self/task/{tid}/status"))?;
        let list = status
            .lines()
            .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))
            .ok_or("gate_swap: Cpus_allowed_list")?
            .trim()
            .to_owned();
        let cpus = list_cpus(&list)?;
        drop(r);
        let ok = !(cpus == 1 && list == c.to_string());
        println!(
            "placement: a machine built on a thread pinned to cpu {c}: its staging thread {tid} \
             Cpus_allowed_list {list} ({cpus} cpus) {}",
            verdict(ok)
        );
        Ok(ok)
    }

    /// Threads of this process named as the swap machine names its staging
    /// thread.
    fn staging_threads() -> Result<usize, GateError> {
        let mut n = 0;
        for t in std::fs::read_dir("/proc/self/task")? {
            let comm = std::fs::read_to_string(t?.path().join("comm"))?;
            n += usize::from(comm.trim_end() == "swap-staging");
        }
        Ok(n)
    }

    impl SwapSource for Synth {
        fn part_bytes(&self, _layer: usize) -> &[usize] {
            &PART_BYTES
        }

        fn source(&self, layer: usize, id: u32, part: usize) -> Result<Piece<'_>, GpuError> {
            if self.faults.fail_source == Some((layer, id)) {
                return Err(GpuError::Shape {
                    what: "Synth::source",
                    detail: format!("layer {layer} expert {id}: the source fails it"),
                });
            }
            if self.faults.panic_source == Some((layer, id)) {
                panic!("Synth::source panics on layer {layer} expert {id} (gate_swap's panic arm)");
            }
            let at = (li(layer) * E + id as usize) * PARTS + part;
            Ok(Piece {
                bytes: &self.bytes[at],
                transform: Transform::Identity,
            })
        }

        fn dest(&self, layer: usize, part: usize, slot: u32) -> Result<sys::CUdeviceptr, GpuError> {
            if slot as usize >= self.caps[li(layer)].max(1) || self.faults.fail_dest == Some(layer)
            {
                return Err(GpuError::Shape {
                    what: "Synth::dest",
                    detail: format!("layer {layer} slot {slot}"),
                });
            }
            Ok(self.bases[li(layer)][part] + (slot as usize * PART_BYTES[part]) as u64)
        }

        /// `inflate` one-word memsets on the stream, into the stack of the
        /// layer with no card slot (one slot no probe reads).
        fn convert(
            &self,
            _layer: usize,
            _part: usize,
            _dst: sys::CUdeviceptr,
            stream: &CudaStream,
        ) -> Result<(), GpuError> {
            let scratch = self.bases[L - 1][0];
            for _ in 0..self.faults.inflate {
                // SAFETY: `scratch` is the first word of a live stack buffer
                // of the arm's card, which outlives the machine; the memset
                // writes that one word on the stream.
                let rc = unsafe { sys::cuMemsetD32Async(scratch, 0, 1, stream.cu_stream()) };
                if rc != sys::cudaError_enum_CUDA_SUCCESS {
                    return Err(GpuError::Shape {
                        what: "Synth::convert",
                        detail: format!("cuMemsetD32Async rc {rc}"),
                    });
                }
            }
            Ok(())
        }

        fn prepare_victim(&self, layer: usize, id: u32) -> Result<(), GpuError> {
            self.prepared.fetch_add(1, Ordering::Relaxed);
            if self.faults.slow.contains(&(layer, id)) {
                std::thread::sleep(SLOW);
            }
            let staging = std::thread::current().name() == Some("swap-staging");
            let evicted = staging && self.faults.evicted.contains(&(layer, id));
            if !self.faults.stuck.contains(&(layer, id)) && !evicted {
                self.resident[li(layer) * E + id as usize].store(true, Ordering::Release);
            }
            Ok(())
        }

        fn host_resident(&self, layer: usize, id: u32) -> Result<bool, GpuError> {
            Ok(self.resident[li(layer) * E + id as usize].load(Ordering::Acquire))
        }

        fn release_host(&self, layer: usize, id: u32) -> Result<u64, GpuError> {
            self.resident[li(layer) * E + id as usize].store(false, Ordering::Release);
            Ok(EXPERT_BYTES)
        }
    }

    /// One arm's card: a stack a (layer, part) of `max(cap, 1)` slots.
    struct Stacks {
        bufs: Vec<Vec<DeviceBuffer<u32>>>,
    }

    impl Stacks {
        fn new(gpu: &Gpu) -> Result<Stacks, GateError> {
            let s = gpu.stream();
            let mut bufs = Vec::with_capacity(L);
            for &n in &N_L {
                let mut parts = Vec::with_capacity(PARTS);
                for &b in &PART_BYTES {
                    parts.push(DeviceBuffer::<u32>::zeroed(s, n.max(1) * b / 4)?);
                }
                bufs.push(parts);
            }
            Ok(Stacks { bufs })
        }

        fn bases(&self) -> Vec<[sys::CUdeviceptr; PARTS]> {
            self.bufs
                .iter()
                .map(|p| {
                    [
                        p[0].cu_deviceptr(),
                        p[1].cu_deviceptr(),
                        p[2].cu_deviceptr(),
                    ]
                })
                .collect()
        }

        /// Expert `e` of layer `l` into slot `slot`, synchronously.
        fn put(&mut self, gpu: &Gpu, l: usize, e: u32, slot: usize) -> Result<(), GateError> {
            for (p, &b) in PART_BYTES.iter().enumerate() {
                let words: Vec<u32> = (0..b / 4).map(|i| word(l, e, p, i)).collect();
                let at = slot * b / 4;
                let buf = &mut self.bufs[li(l)][p];
                // SAFETY: slot < the stack's slots (callers place below the
                // layer's capacity), so [at, at + b/4) is inside `buf`; the
                // copy is synchronous and `words` outlives it.
                let rc = unsafe {
                    sys::cuMemcpyHtoD_v2(
                        buf.cu_deviceptr() + (at * 4) as u64,
                        words.as_ptr().cast(),
                        b,
                    )
                };
                if rc != sys::cudaError_enum_CUDA_SUCCESS {
                    return Err(format!("gate_swap: cuMemcpyHtoD_v2 rc {rc}").into());
                }
            }
            gpu.stream().synchronize()?;
            Ok(())
        }
    }

    /// The seed: experts `0..n_l` of every layer in slots `0..n_l`, and each
    /// layer's `tier[i]` experts in tier slots `0..` in that order.
    fn seed_map(tier: &[Vec<u32>]) -> Result<SlotMap, GateError> {
        let mut rows = vec![HOST; L * E];
        for (i, &n) in N_L.iter().enumerate() {
            for e in 0..n {
                rows[i * E + e] = e as u32;
            }
            for (s, &e) in tier
                .get(i)
                .map_or(&[][..], Vec::as_slice)
                .iter()
                .enumerate()
            {
                rows[i * E + e as usize] = Slot::Tier {
                    tier: 0,
                    slot: s as u32,
                }
                .entry()?;
            }
        }
        Ok(SlotMap::from_rows(LAYERS, E, rows)?)
    }

    /// The synthetic router: per pass its rows (each `L` × `K` ids) and its
    /// kept count.
    struct Trace {
        passes: Vec<(Vec<[[u32; K]; L]>, usize)>,
    }

    fn trace() -> Trace {
        let mut x: u32 = 1;
        let mut below = |n: u32| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (x >> 8) % n
        };
        let mut passes = Vec::with_capacity(PASSES);
        for p in 0..PASSES {
            let rows = if p % 5 == 4 { 2 } else { 1 };
            let mut out = Vec::with_capacity(rows);
            for _ in 0..rows {
                let mut r = [[0u32; K]; L];
                for (i, ids) in r.iter_mut().enumerate() {
                    let base = (PINNED + (p / 20) * 3 + i * 2) as u32;
                    let mut n = 0;
                    while n < K {
                        let e = if below(4) < 3 {
                            PINNED as u32 + (base - PINNED as u32 + below(8)) % (E - PINNED) as u32
                        } else {
                            PINNED as u32 + below((E - PINNED) as u32)
                        };
                        if !ids[..n].contains(&e) {
                            ids[n] = e;
                            n += 1;
                        }
                    }
                }
                out.push(r);
            }
            passes.push((out, 1));
        }
        Trace { passes }
    }

    fn params(delay: u64) -> SwapParams {
        SwapParams {
            every: 2,
            cap: 6,
            margin: 1.0,
            min_count: 2.0,
            decay: 0.9,
            spares: 1,
            delay,
        }
    }

    /// The trace's first `2 × passes` rows re-chunked into `passes` passes
    /// of 2 rows each, both kept: the drafted shape (a pass of several kept
    /// rows) at this trace's own routing.
    fn two_row_trace(trace: &Trace, passes: usize) -> Trace {
        let flat: Vec<[[u32; K]; L]> = trace
            .passes
            .iter()
            .flat_map(|(rows, _)| rows.iter().copied())
            .take(2 * passes)
            .collect();
        Trace {
            passes: flat.chunks(2).map(|c| (c.to_vec(), 2)).collect(),
        }
    }

    /// A card with the probe's graph over its own buffers. The graph is
    /// declared first so it drops before the buffers it reads.
    struct Card {
        graph: Option<Graph>,
        stacks: Stacks,
        view: DeviceBuffer<u32>,
        ids: DeviceBuffer<u32>,
        out: DeviceBuffer<u32>,
    }

    impl Card {
        fn new(gpu: &Gpu, map: &SlotMap) -> Result<Card, GateError> {
            let s = gpu.stream();
            let mut stacks = Stacks::new(gpu)?;
            for l in LAYERS {
                for e in 0..E as u32 {
                    if let Some(Slot::Card(slot)) = map.slot(l, e) {
                        stacks.put(gpu, l, e, slot as usize)?;
                    }
                }
            }
            Ok(Card {
                graph: None,
                stacks,
                view: DeviceBuffer::from_host(s, &map.stage_view())?,
                ids: DeviceBuffer::<u32>::zeroed(s, MAX_ROWS * L * K)?,
                out: DeviceBuffer::<u32>::zeroed(s, L * PARTS * NIDS)?,
            })
        }

        /// Capture the probe of every (layer, part) once.
        fn capture(
            &mut self,
            gpu: &Gpu,
            pm: &probe_kernels::LoadedModule,
        ) -> Result<(), GateError> {
            let Card {
                stacks,
                view,
                ids,
                out,
                ..
            } = self;
            let graph = gpu.capture(|stream| {
                for (i, l) in LAYERS.enumerate() {
                    for (p, &b) in PART_BYTES.iter().enumerate() {
                        let prep = pm
                            .prepare_swap_probe(LaunchConfig1D::new(1, 32, 0))
                            .map_err(|e| GpuError::Shape {
                                what: "gate_swap probe",
                                detail: e.to_string(),
                            })?;
                        pm.swap_probe(
                            stream,
                            &prep,
                            view,
                            (i * E) as u32,
                            E as u32,
                            ids,
                            i as u32,
                            L as u32,
                            K as u32,
                            NIDS as u32,
                            &stacks.bufs[i][p],
                            N_L[i] as u32,
                            (b / 4) as u32,
                            ((i * PARTS + p) * NIDS) as u32,
                            out,
                        )
                        .map_err(|e| GpuError::Shape {
                            what: "gate_swap probe",
                            detail: format!("layer {l} part {p}: {e}"),
                        })?;
                    }
                }
                Ok(())
            })?;
            self.graph = Some(graph);
            Ok(())
        }

        /// The pass's ids into the buffer the graph reads, on the engine
        /// stream.
        fn refresh(&mut self, gpu: &Gpu, rows: &[[[u32; K]; L]]) -> Result<(), GateError> {
            let mut flat = vec![0u32; MAX_ROWS * L * K];
            for (r, row) in rows.iter().enumerate() {
                for (i, ids) in row.iter().enumerate() {
                    flat[(r * L + i) * K..][..K].copy_from_slice(ids);
                }
            }
            self.ids.copy_from_host(gpu.stream(), &flat)?;
            Ok(())
        }

        fn launch(&self, gpu: &Gpu) -> Result<(), GateError> {
            self.graph
                .as_ref()
                .ok_or("gate_swap: no captured probe")?
                .launch(gpu.stream())?;
            Ok(())
        }

        fn read(&self, gpu: &Gpu) -> Result<Vec<u32>, GateError> {
            Ok(self.out.to_host_vec(gpu.stream())?)
        }
    }

    /// A pass's value and its three error counts.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct Value {
        fnv: u64,
        stale: u32,
        double: u32,
        miss: u32,
    }

    /// The pass's value from the probe's output `out` and the host map.
    fn value(rows: &[[[u32; K]; L]], out: &[u32], slots: &SlotMap) -> Value {
        let mut v = Value {
            fnv: 0xcbf2_9ce4_8422_2325,
            ..Value::default()
        };
        let mut mix = |x: u32| {
            for b in x.to_le_bytes() {
                v.fnv ^= u64::from(b);
                v.fnv = v.fnv.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        let (mut stale, mut double, mut miss) = (0, 0, 0);
        for (r, row) in rows.iter().enumerate() {
            for (i, ids) in row.iter().enumerate() {
                let l = LAYERS.start + i;
                for (k, &id) in ids.iter().enumerate() {
                    let j = r * K + k;
                    let card = (0..PARTS).map(|p| out[(i * PARTS + p) * NIDS + j]);
                    let on_card = out[i * PARTS * NIDS + j] != OUT_HOST;
                    let entry = slots.slot(l, id);
                    let on_host = entry == Some(Slot::Host);
                    let on_tier = matches!(entry, Some(Slot::Tier { .. }));
                    mix(id);
                    match u32::from(on_card) + u32::from(on_host) + u32::from(on_tier) {
                        0 => miss += 1,
                        1 => {}
                        _ => double += 1,
                    }
                    if on_card {
                        mix(1);
                        for (p, got) in card.enumerate() {
                            if got != expect(l, id, p) {
                                stale += 1;
                            }
                            mix(got);
                        }
                    }
                    if on_host {
                        mix(2);
                        for p in 0..PARTS {
                            mix(expect(l, id, p) ^ HOST_MASK);
                        }
                    }
                    if on_tier {
                        mix(3);
                        for p in 0..PARTS {
                            mix(expect(l, id, p) ^ TIER_MASK);
                        }
                    }
                }
            }
        }
        v.stale = stale;
        v.double = double;
        v.miss = miss;
        v
    }

    /// How an adaptive run's copies are timed.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Copies {
        /// The window open: a copy runs as soon as it is issued.
        Prompt,
        /// The window closed until a flip's live boundary is enqueued.
        Held,
    }

    /// A stream an arm holds by a host flag ([`HostFlags::enqueue_wait`]).
    #[derive(Clone, Copy)]
    enum Hold<'a> {
        None,
        /// The machine's copy stream, from before boundary `from` until
        /// after the launch of pass `to`; the arm looks whether the engine
        /// stream is still waiting before it releases.
        Copies {
            flags: &'a HostFlags,
            from: usize,
            to: usize,
        },
        /// The engine stream, from before pass `at`'s launch until after
        /// boundary `at + 1`.
        Engine {
            flags: &'a HostFlags,
            at: usize,
        },
    }

    impl Hold<'_> {
        /// Raise the hold's flag: every path out of a held arm does.
        fn release(self) -> Result<(), GateError> {
            match self {
                Hold::None => Ok(()),
                Hold::Copies { flags, .. } | Hold::Engine { flags, .. } => Ok(flags.raise(0)?),
            }
        }
    }

    /// What an adaptive run saw: per pass its value, its report and its
    /// card sets; the flips made, with the boundary; and the machine, the
    /// map and the card, for the arms that go on.
    struct Run {
        delay: u64,
        values: Vec<Value>,
        reports: Vec<PassReport>,
        sets: Vec<Vec<Vec<u32>>>,
        flips: Vec<(u64, Flip)>,
        slot_moves: usize,
        pinned_kept: bool,
        /// A held arm: whether the stream held on the other side was still
        /// waiting when the arm looked.
        held_waited: Option<bool>,
        /// An engine hold: whether the machine's copy stream was still
        /// waiting when the arm looked, before the release.
        copy_waited: Option<bool>,
        err: Option<String>,
        machine: Option<SwapMachine>,
        slots: SlotMap,
        card: Card,
    }

    fn card_sets(slots: &SlotMap) -> Vec<Vec<u32>> {
        LAYERS
            .map(|l| {
                (0..E as u32)
                    .filter(|&e| matches!(slots.slot(l, e), Some(Slot::Card(_))))
                    .collect()
            })
            .collect()
    }

    fn cfg(delay: u64, deadline: Duration) -> MachineCfg {
        MachineCfg {
            params: params(delay),
            pinned: N_L
                .iter()
                .map(|&n| if n > 0 { PINNED } else { 0 })
                .collect(),
            top_k: K,
            max_rows: MAX_ROWS,
            unrouted: Vec::new(),
            deadline,
        }
    }

    /// Wait for the engine stream to drain, polling, at most `ENGINE_DRAIN`:
    /// a stream that waits on a copy nothing stages is a named failure here,
    /// not a hang in a synchronizing call.
    fn drain(gpu: &Gpu) -> Result<(), String> {
        let t0 = Instant::now();
        loop {
            match gpu.stream().query() {
                Ok(true) => return Ok(()),
                Ok(false) if t0.elapsed() > ENGINE_DRAIN => {
                    return Err(format!(
                        "the engine stream did not drain in {:?}: it waits on a copy that was \
                         never staged",
                        t0.elapsed()
                    ));
                }
                Ok(false) => std::thread::sleep(Duration::from_micros(50)),
                Err(e) => return Err(format!("cuStreamQuery: {e}")),
            }
        }
    }

    /// Note every row of a pass into the tally.
    fn note_rows(
        tally: &mut bloomery_gpu::host::swap::Tally,
        rows: &[[[u32; K]; L]],
    ) -> Result<(), GateError> {
        for (r, row) in rows.iter().enumerate() {
            for (i, ids) in row.iter().enumerate() {
                for (k, &id) in ids.iter().enumerate() {
                    tally.note(LAYERS.start + i, r, k, id)?;
                }
            }
        }
        Ok(())
    }

    /// Drive `passes` of the trace through a machine, in the engine's order:
    /// per pass the ids refreshed (a synchronizing copy), the boundary, the
    /// graph, a bounded drain and the readback, the value, then the ids
    /// noted and the pass's end with its kept rows. Held, the staging window
    /// stays closed for the whole run, so every copy is staged only once its
    /// flip is due. `hold` holds a stream for one stretch ([`Hold`]); an
    /// engine hold takes the next boundary before the held pass is read, and
    /// refreshes the next pass's ids only after the release.
    fn drive(
        gpu: &Gpu,
        run: &mut Run,
        trace: &Trace,
        passes: Range<usize>,
        copies: Copies,
        hold: Hold<'_>,
    ) -> Result<(), GateError> {
        let r = drive_held(gpu, run, trace, passes, copies, hold);
        if r.is_err() {
            hold.release()?;
        }
        r
    }

    /// [`drive`]'s passes; `drive` raises the hold's flag when this fails.
    fn drive_held(
        gpu: &Gpu,
        run: &mut Run,
        trace: &Trace,
        passes: Range<usize>,
        copies: Copies,
        hold: Hold<'_>,
    ) -> Result<(), GateError> {
        let stream = gpu.stream();
        let delay = run.delay;
        let m = run.machine.as_mut().ok_or("gate_swap: no machine")?;
        let window = m.window();
        let mut tally = m.tally();
        if copies == Copies::Held {
            window.store(0, Ordering::Release);
        }
        let mut pinned_slots: Vec<Vec<(u32, Option<Slot>)>> = Vec::with_capacity(L);
        for l in LAYERS {
            let seed = m.seed(l)?;
            let pins = m.pinned(l)?;
            pinned_slots.push(
                seed[..pins]
                    .iter()
                    .map(|&e| (e, run.slots.slot(l, e)))
                    .collect(),
            );
        }
        let mut pre: Option<PassReport> = None;
        let end = passes.end;
        for p in passes {
            let (rows, kept) = &trace.passes[p];
            let report = match pre.take() {
                Some(r) => {
                    run.card.refresh(gpu, rows)?;
                    r
                }
                None => {
                    run.card.refresh(gpu, rows)?;
                    if let Hold::Copies { flags, from, .. } = hold
                        && p == from
                    {
                        flags.clear(0)?;
                        flags.enqueue_wait(m.copy_stream(), 0)?;
                    }
                    match m.boundary(stream, &mut run.slots) {
                        Ok(r) => r,
                        Err(e) => {
                            run.err = Some(e.to_string());
                            window.store(1, Ordering::Release);
                            hold.release()?;
                            return Ok(());
                        }
                    }
                }
            };
            for f in m.rule().in_flight() {
                if f.live_at == report.boundary + delay {
                    run.flips.push((report.boundary, *f));
                }
            }
            run.slot_moves += report.landed;
            if let Hold::Engine { flags, at } = hold
                && p == at
            {
                flags.clear(0)?;
                flags.enqueue_wait(stream, 0)?;
                run.card.launch(gpu)?;
                note_rows(&mut tally, rows)?;
                m.end_pass(&mut tally, KeptRows::prefix(*kept))?;
                let held_map = run.slots.clone();
                if p + 1 < end {
                    match m.boundary(stream, &mut run.slots) {
                        Ok(r) => pre = Some(r),
                        Err(e) => {
                            run.err = Some(format!("the boundary after held pass {p}: {e}"));
                            window.store(1, Ordering::Release);
                            hold.release()?;
                            return Ok(());
                        }
                    }
                }
                std::thread::sleep(HOLD_SETTLE);
                run.held_waited = Some(stream.query() == Ok(false));
                run.copy_waited = Some(m.copy_stream().query() == Ok(false));
                hold.release()?;
                if let Err(e) = drain(gpu) {
                    run.err = Some(format!("pass {p}: {e}"));
                    window.store(1, Ordering::Release);
                    return Ok(());
                }
                let out = run.card.read(gpu)?;
                run.values.push(value(rows, &out, &held_map));
                run.reports.push(report);
                run.sets.push(card_sets(&held_map));
                continue;
            }
            run.card.launch(gpu)?;
            if let Hold::Copies { to, .. } = hold
                && p == to
            {
                std::thread::sleep(HOLD_SETTLE);
                run.held_waited = Some(stream.query() == Ok(false));
                hold.release()?;
            }
            if let Err(e) = drain(gpu) {
                run.err = Some(format!("pass {p}: {e}"));
                window.store(1, Ordering::Release);
                hold.release()?;
                return Ok(());
            }
            let out = run.card.read(gpu)?;
            run.values.push(value(rows, &out, &run.slots));
            run.reports.push(report);
            run.sets.push(card_sets(&run.slots));
            note_rows(&mut tally, rows)?;
            m.end_pass(&mut tally, KeptRows::prefix(*kept))?;
        }
        window.store(1, Ordering::Release);
        run.pinned_kept = LAYERS.zip(&pinned_slots).all(|(l, pins)| {
            pins.iter()
                .all(|&(e, slot)| slot.is_some() && run.slots.slot(l, e) == slot)
        });
        Ok(())
    }

    /// Drive `passes` of the trace with every boundary after the first made
    /// ahead of its pass: per pass the ids refreshed, the boundary taken
    /// (the first made), the graph, the ids noted and the pass's end, the
    /// next pass's boundary made ahead ([`SwapMachine::boundary_ahead`]),
    /// then the bounded drain and the readback, the value read against the
    /// map the pass ran on.
    fn drive_ahead(
        gpu: &Gpu,
        run: &mut Run,
        trace: &Trace,
        passes: Range<usize>,
    ) -> Result<(), GateError> {
        let stream = gpu.stream();
        let delay = run.delay;
        let m = run.machine.as_mut().ok_or("gate_swap: no machine")?;
        let mut tally = m.tally();
        let mut pre: Option<PassReport> = None;
        let end = passes.end;
        for p in passes {
            let (rows, kept) = &trace.passes[p];
            run.card.refresh(gpu, rows)?;
            let report = match pre.take() {
                Some(r) => {
                    m.take_ahead()?;
                    r
                }
                None => m.boundary(stream, &mut run.slots)?,
            };
            for f in m.rule().in_flight() {
                if f.live_at == report.boundary + delay {
                    run.flips.push((report.boundary, *f));
                }
            }
            run.slot_moves += report.landed;
            run.card.launch(gpu)?;
            note_rows(&mut tally, rows)?;
            m.end_pass(&mut tally, KeptRows::prefix(*kept))?;
            let ran_on = run.slots.clone();
            if p + 1 < end {
                pre = Some(m.boundary_ahead(stream, &mut run.slots)?);
            }
            drain(gpu).map_err(|e| format!("pass {p}: {e}"))?;
            let out = run.card.read(gpu)?;
            run.values.push(value(rows, &out, &ran_on));
            run.reports.push(report);
            run.sets.push(card_sets(&ran_on));
        }
        Ok(())
    }

    /// Whether `r` is the refusal of a boundary made ahead that no launch
    /// has taken.
    fn refused_ahead<T>(r: Result<T, GpuError>) -> bool {
        r.err()
            .is_some_and(|e| e.to_string().contains("made ahead of its pass"))
    }

    /// ahead: the trace with its boundaries made ahead against run `a`, then
    /// the made-ahead state's refusals on that machine.
    fn ahead(
        gpu: &Gpu,
        pm: &probe_kernels::LoadedModule,
        trace: &Trace,
        a: &Run,
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let mut r = plain(gpu, pm, Faults::default(), DELAY)?;
        drive_ahead(gpu, &mut r, trace, 0..PASSES)?;
        let same = r.err.is_none()
            && r.values.len() == PASSES
            && fnvs(&r.values) == fnvs(&a.values)
            && r.flips == a.flips
            && clean(&r.values);
        let made = r.reports.iter().filter(|p| p.ahead).count();
        let n = r.values.len();
        let Run { machine, slots, .. } = &mut r;
        let m = machine.as_mut().ok_or("gate_swap: no machine")?;
        let b = m.boundary_ahead(stream, slots)?.boundary;
        let boundary = refused_ahead(m.boundary(stream, slots));
        let mut t = m.tally();
        let end = refused_ahead(m.end_pass(&mut t, KeptRows::prefix(0)));
        let call = refused_ahead(m.begin_call(stream, CallCfg { floor: 1 }));
        let took = m.take_ahead().ok() == Some(b);
        let again = m.take_ahead().err().map(|e| e.to_string());
        let twice = again.as_deref().is_some_and(|e| e.contains("none waits"));
        m.end_pass(&mut t, KeptRows::prefix(0))?;
        m.boundary_ahead(stream, slots)?;
        m.reset(stream, slots)?;
        let dropped = m.ahead().is_none() && m.take_ahead().is_err();
        let unbroken = m.broken().is_none();
        let ok = same
            && made == PASSES - 1
            && boundary
            && end
            && call
            && took
            && twice
            && dropped
            && unbroken;
        println!(
            "ahead: {} passes with {made} boundaries made ahead, values and flips equal the \
             prompt run {same}; boundary {b} made ahead refuses a boundary {boundary}, an end of \
             pass {end}, a prompt call {call}; taken once {took}, a second take {again:?} \
             refused {twice}; a reset drops one made ahead {dropped}; unbroken {unbroken} {}",
            n,
            verdict(ok)
        );
        Ok(ok)
    }

    /// A fresh arm over `slots`: its card, its captured probe and a machine
    /// of `cfg` over a source with `faults`.
    fn start(
        gpu: &Gpu,
        pm: &probe_kernels::LoadedModule,
        mut slots: SlotMap,
        faults: Faults,
        cfg: MachineCfg,
    ) -> Result<Run, GateError> {
        let mut card = Card::new(gpu, &slots)?;
        card.capture(gpu, pm)?;
        let source = Arc::new(Synth::new(&card.stacks, N_L, faults));
        let delay = cfg.params.delay;
        let machine = SwapMachine::new(
            gpu.context(),
            gpu.stream(),
            &mut slots,
            &card.view,
            source,
            cfg,
        );
        let (machine, err) = match machine {
            Ok(m) => (Some(m), None),
            Err(e) => (None, Some(e.to_string())),
        };
        Ok(Run {
            delay,
            values: Vec::new(),
            reports: Vec::new(),
            sets: Vec::new(),
            flips: Vec::new(),
            slot_moves: 0,
            pinned_kept: false,
            held_waited: None,
            copy_waited: None,
            err,
            machine,
            slots,
            card,
        })
    }

    /// A fresh stage-only arm at live delay `delay` with the pass deadline.
    fn plain(
        gpu: &Gpu,
        pm: &probe_kernels::LoadedModule,
        faults: Faults,
        delay: u64,
    ) -> Result<Run, GateError> {
        start(gpu, pm, seed_map(&[])?, faults, cfg(delay, DEADLINE))
    }

    fn clean(values: &[Value]) -> bool {
        values
            .iter()
            .all(|v| v.stale == 0 && v.double == 0 && v.miss == 0)
    }

    fn errs(values: &[Value]) -> (u32, u32, u32) {
        values.iter().fold((0, 0, 0), |(s, d, m), v| {
            (s + v.stale, d + v.double, m + v.miss)
        })
    }

    fn fnvs(values: &[Value]) -> Vec<u64> {
        values.iter().map(|v| v.fnv).collect()
    }

    fn show(e: &Option<String>) -> String {
        e.as_ref()
            .map_or_else(|| "none".to_string(), ToString::to_string)
    }

    /// Per card layer, the flips `run` made there.
    fn flips_per_layer(run: &Run) -> Vec<usize> {
        LAYERS
            .map(|l| {
                run.flips
                    .iter()
                    .filter(|(_, f)| f.layer + LAYERS.start == l)
                    .count()
            })
            .collect()
    }

    /// Flips on every layer with card slots and on no other.
    fn flips_everywhere(run: &Run) -> bool {
        flips_per_layer(run)
            .iter()
            .zip(N_L)
            .all(|(&n, cap)| (cap == 0) == (n == 0))
    }

    /// Whether any row of pass `pass` routes expert `id` at the rule's layer
    /// `layer`: the probe then reads its slot.
    fn routes(trace: &Trace, pass: usize, layer: usize, id: u32) -> bool {
        trace
            .passes
            .get(pass)
            .is_some_and(|(rows, _)| rows.iter().any(|row| row[layer].contains(&id)))
    }

    /// The first flip `run` made: its boundary, its layer in the map's
    /// numbering and the flip.
    fn first_flip(run: &Run) -> Result<(u64, usize, Flip), GateError> {
        let &(b, f) = run.flips.first().ok_or("gate_swap: the run made no flip")?;
        Ok((b, LAYERS.start + f.layer, f))
    }

    /// The static replay: per pass, the adaptive run's card sets placed by
    /// synchronous copies, each layer's set in slots by ascending id, and a
    /// fresh map from those rows.
    fn static_replay(
        gpu: &Gpu,
        pm: &probe_kernels::LoadedModule,
        trace: &Trace,
        sets: &[Vec<Vec<u32>>],
    ) -> Result<Vec<Value>, GateError> {
        let mut slots = seed_map(&[])?;
        let mut card = Card::new(gpu, &slots)?;
        card.capture(gpu, pm)?;
        let mut placed: Vec<Vec<u32>> = vec![Vec::new(); L];
        let mut values = Vec::with_capacity(sets.len());
        for (p, set) in sets.iter().enumerate() {
            let mut rows = vec![HOST; L * E];
            for (i, ids) in set.iter().enumerate() {
                for (s, &e) in ids.iter().enumerate() {
                    rows[i * E + e as usize] = s as u32;
                    if placed[i].get(s) != Some(&e) {
                        card.stacks.put(gpu, LAYERS.start + i, e, s)?;
                    }
                }
                placed[i].clone_from(ids);
            }
            slots = SlotMap::from_rows(LAYERS, E, rows)?;
            card.view
                .copy_from_host(gpu.stream(), &slots.stage_view())?;
            let (pass_rows, _) = &trace.passes[p];
            card.refresh(gpu, pass_rows)?;
            card.launch(gpu)?;
            let out = card.read(gpu)?;
            values.push(value(pass_rows, &out, &slots));
        }
        Ok(values)
    }

    /// The flips an independent rule makes over the trace at live delay
    /// `delay`, boundary by boundary: every row observed, each pass ended at
    /// its kept rows, each boundary after the first planned — the machine's
    /// seed (slots `0..n_l` less one spare), pinned counts and parameters.
    fn rule_replay(trace: &Trace, delay: u64) -> Result<Vec<(u64, Flip)>, GateError> {
        let seeds: Vec<Vec<u32>> = N_L.iter().map(|&n| (0..n as u32).collect()).collect();
        let refs: Vec<&[u32]> = seeds.iter().map(Vec::as_slice).collect();
        let capacity: Vec<usize> = N_L.iter().map(|&n| n.saturating_sub(1)).collect();
        let pinned = cfg(delay, DEADLINE).pinned;
        let shape = Shape {
            experts: E,
            top_k: K,
            max_rows: MAX_ROWS,
        };
        let mut r = SwapRule::new_pinned(params(delay), shape, &refs, &capacity, &pinned)?;
        let mut flips = Vec::new();
        for (p, (rows, kept)) in trace.passes.iter().enumerate() {
            let b = p as u64;
            if b > 0 {
                flips.extend(r.plan(b)?.iter().map(|&f| (b, f)));
            }
            for (row, ids) in rows.iter().enumerate() {
                for (i, layer_ids) in ids.iter().enumerate() {
                    r.observe(i, row, layer_ids)?;
                }
            }
            r.end_pass(KeptRows::prefix(*kept))?;
        }
        Ok(flips)
    }

    /// Per card layer, two experts off the seed for the tier: the one the
    /// trace's kept rows route most (the rule would admit it) and the one
    /// they route least; lower id first on a tie.
    fn tier_sets(trace: &Trace) -> Vec<Vec<u32>> {
        let mut counts = vec![[0u32; E]; L];
        for (rows, kept) in &trace.passes {
            for row in &rows[..*kept] {
                for (i, ids) in row.iter().enumerate() {
                    for &id in ids {
                        counts[i][id as usize] += 1;
                    }
                }
            }
        }
        (0..L)
            .map(|i| {
                if N_L[i] == 0 {
                    return Vec::new();
                }
                let off = N_L[i] as u32..E as u32;
                let hot = off
                    .clone()
                    .max_by_key(|&e| (counts[i][e as usize], std::cmp::Reverse(e)))
                    .expect("experts off the seed");
                let cold = off
                    .filter(|&e| e != hot)
                    .min_by_key(|&e| (counts[i][e as usize], e))
                    .expect("two experts off the seed");
                vec![hot, cold]
            })
            .collect()
    }

    // ------------------------------------------------------------ clauses

    fn c1(a: &Run, b: &Run, h: &Run) -> bool {
        let c1 = a.err.is_none()
            && b.err.is_none()
            && h.err.is_none()
            && a.values.len() == PASSES
            && fnvs(&a.values) == fnvs(&b.values)
            && fnvs(&a.values) == fnvs(&h.values)
            && a.flips == b.flips
            && a.flips == h.flips
            && flips_everywhere(a);
        println!(
            "c1 same history: {} passes, {} flips (per layer {:?}), landed {}; prompt twice \
             equal {} flips equal {}; held equal {} flips equal {}; errors {} / {} / {} {}",
            a.values.len(),
            a.flips.len(),
            flips_per_layer(a),
            a.slot_moves,
            fnvs(&a.values) == fnvs(&b.values),
            a.flips == b.flips,
            fnvs(&a.values) == fnvs(&h.values),
            a.flips == h.flips,
            show(&a.err),
            show(&b.err),
            show(&h.err),
            verdict(c1)
        );
        c1
    }

    fn c2(
        gpu: &Gpu,
        pm: &probe_kernels::LoadedModule,
        trace: &Trace,
        a: &Run,
    ) -> Result<bool, GateError> {
        let reference = static_replay(gpu, pm, trace, &a.sets)?;
        let first_diff = fnvs(&a.values)
            .iter()
            .zip(fnvs(&reference))
            .position(|(x, y)| *x != y);
        let c2 = reference.len() == a.values.len() && first_diff.is_none() && clean(&reference);
        println!(
            "c2 static replay (slots by ascending id, synchronous copies): {} passes, first \
             differing pass {first_diff:?}, reference errors (stale, double, miss) {:?} {}",
            reference.len(),
            errs(&reference),
            verdict(c2)
        );
        Ok(c2)
    }

    fn c3(a: &Run, h: &Run) -> bool {
        let late: usize = h.reports.iter().map(|r| r.late).sum();
        let c3 = h.err.is_none()
            && late > 0
            && late == h.slot_moves
            && clean(&h.values)
            && fnvs(&h.values) == fnvs(&a.values);
        println!(
            "c3 held copies (window closed all run): late landings {late} of {}, errors (stale, \
             double, miss) {:?}, values = the prompt run's {}, prompt run's late landings {} {}",
            h.slot_moves,
            errs(&h.values),
            fnvs(&h.values) == fnvs(&a.values),
            a.reports.iter().map(|r| r.late).sum::<usize>(),
            verdict(c3)
        );
        c3
    }

    /// c3 engine: the copy stream held by a host word from the boundary of
    /// the first flip whose admitted expert the pass it lands at routes, until
    /// after that pass's launch.
    fn c3_engine(
        gpu: &Gpu,
        pm: &probe_kernels::LoadedModule,
        trace: &Trace,
        a: &Run,
    ) -> Result<bool, GateError> {
        let Some(&(b0, f)) = a
            .flips
            .iter()
            .find(|(b, f)| routes(trace, (b + DELAY) as usize, f.layer, f.admit))
        else {
            println!("c3 engine waits: no flip whose admitted expert its landing pass routes FAIL");
            return Ok(false);
        };
        let l = LAYERS.start + f.layer;
        let flags = HostFlags::new(gpu.context(), 1)?;
        let (from, to) = (b0 as usize, (b0 + DELAY) as usize);
        let mut e = plain(gpu, pm, Faults::default(), DELAY)?;
        drive(
            gpu,
            &mut e,
            trace,
            0..PASSES,
            Copies::Prompt,
            Hold::Copies {
                flags: &flags,
                from,
                to,
            },
        )?;
        let pass = e.values.get(to).copied().unwrap_or_default();
        let ok = e.err.is_none()
            && e.held_waited == Some(true)
            && clean(&e.values)
            && fnvs(&e.values) == fnvs(&a.values);
        println!(
            "c3 engine waits: copy stream held from boundary {from} (layer {l} expert {} into the \
             card) to after pass {to}'s launch: engine stream still waiting {:?}; pass {to} \
             (stale, double, miss) ({}, {}, {}), run errors {:?}, values = the prompt run's {}; \
             error {} {}",
            f.admit,
            e.held_waited,
            pass.stale,
            pass.double,
            pass.miss,
            errs(&e.values),
            fnvs(&e.values) == fnvs(&a.values),
            show(&e.err),
            verdict(ok)
        );
        Ok(ok)
    }

    /// c3 boundary event: at live delay 2, the engine stream held through
    /// the pass before a boundary that lands a flip and issues one into the
    /// slot its victim frees, a pass that routes that victim.
    fn c3_event(
        gpu: &Gpu,
        pm: &probe_kernels::LoadedModule,
        trace: &Trace,
    ) -> Result<bool, GateError> {
        let mut q = plain(gpu, pm, Faults::default(), DELAY_EVEN)?;
        drive(gpu, &mut q, trace, 0..PASSES, Copies::Prompt, Hold::None)?;
        let reuse = q.flips.iter().find_map(|&(b, f)| {
            let land = b + DELAY_EVEN;
            (routes(trace, (land - 1) as usize, f.layer, f.evict)
                && q.flips
                    .iter()
                    .any(|&(b2, g)| b2 == land && g.layer == f.layer))
            .then_some((land, f))
        });
        let Some((b, f)) = reuse else {
            println!(
                "c3 boundary event: no boundary at delay {DELAY_EVEN} lands a flip whose victim the \
                 pass before it routes and issues one in the same layer ({} flips) FAIL",
                q.flips.len()
            );
            return Ok(false);
        };
        let at = (b - 1) as usize;
        let flags = HostFlags::new(gpu.context(), 1)?;
        let mut v = plain(gpu, pm, Faults::default(), DELAY_EVEN)?;
        drive(
            gpu,
            &mut v,
            trace,
            0..PASSES,
            Copies::Prompt,
            Hold::Engine { flags: &flags, at },
        )?;
        let pass = v.values.get(at).copied().unwrap_or_default();
        let ok = q.err.is_none()
            && v.err.is_none()
            && v.held_waited == Some(true)
            && v.copy_waited == Some(true)
            && clean(&v.values)
            && fnvs(&v.values) == fnvs(&q.values);
        println!(
            "c3 boundary event: engine stream held through pass {at}, which reads layer {} victim \
             {}; boundary {b} frees its slot and issues into it; the copy stream still waiting \
             {:?}; held pass (stale, double, miss) ({}, {}, {}), run errors {:?}, values = the \
             unheld run's {}; errors {} / {} {}",
            LAYERS.start + f.layer,
            f.evict,
            v.copy_waited,
            pass.stale,
            pass.double,
            pass.miss,
            errs(&v.values),
            fnvs(&v.values) == fnvs(&q.values),
            show(&q.err),
            show(&v.err),
            verdict(ok)
        );
        Ok(ok)
    }

    fn c5(runs: &[&Run]) -> bool {
        let c5 = runs.iter().all(|r| clean(&r.values));
        let per: Vec<(u32, u32, u32)> = runs.iter().map(|r| errs(&r.values)).collect();
        println!(
            "c5 host map and card map agree every pass: errors (stale, double, miss) prompt, \
             prompt, held {per:?} {}",
            verdict(c5)
        );
        c5
    }

    fn c6(trace: &Trace, a: &Run) -> Result<bool, GateError> {
        let want = rule_replay(trace, DELAY)?;
        let first_diff = (0..want.len().max(a.flips.len()))
            .find(|&i| want.get(i) != a.flips.get(i))
            .map(|i| (want.get(i).copied(), a.flips.get(i).copied()));
        let c6 = !want.is_empty() && first_diff.is_none();
        println!(
            "c6 kept rows only: an independent rule over every row, ended at the kept rows, makes \
             {} flips, the machine {}; first difference (rule, machine) {first_diff:?} {}",
            want.len(),
            a.flips.len(),
            verdict(c6)
        );
        Ok(c6)
    }

    /// The kept-row clock over a two-row-pass history: with `every` 2 and
    /// every pass keeping 2 rows the machine plans at every boundary, so
    /// some flip is issued at an odd one — a pass clock at `every` 2 issues
    /// flips at even boundaries only — and its flips are an independent
    /// rule's over the same history, its values clean.
    fn clock(
        gpu: &Gpu,
        pm: &probe_kernels::LoadedModule,
        trace: &Trace,
    ) -> Result<bool, GateError> {
        let two = two_row_trace(trace, 40);
        let mut r = plain(gpu, pm, Faults::default(), DELAY)?;
        drive(
            gpu,
            &mut r,
            &two,
            0..two.passes.len(),
            Copies::Prompt,
            Hold::None,
        )?;
        let want = rule_replay(&two, DELAY)?;
        let first_diff = (0..want.len().max(r.flips.len()))
            .find(|&i| want.get(i) != r.flips.get(i))
            .map(|i| (want.get(i).copied(), r.flips.get(i).copied()));
        let odd = r.flips.iter().filter(|(b, _)| b % 2 == 1).count();
        let ok = r.err.is_none()
            && !want.is_empty()
            && first_diff.is_none()
            && odd > 0
            && clean(&r.values);
        println!(
            "clock kept rows: {} two-row passes, {} flips (an independent rule's {}), at odd \
             boundaries {odd} (a pass clock at every 2 issues none), errors (stale, double, miss) \
             {:?}, first difference (rule, machine) {first_diff:?} {}",
            two.passes.len(),
            r.flips.len(),
            want.len(),
            errs(&r.values),
            verdict(ok)
        );
        Ok(ok)
    }

    fn pinned(a: &Run) -> bool {
        let pinned_victim = a
            .flips
            .iter()
            .filter(|(_, f)| (f.evict as usize) < PINNED)
            .count();
        let seed_victims = a
            .flips
            .iter()
            .filter(|(_, f)| (PINNED..N_L[f.layer]).contains(&(f.evict as usize)))
            .count();
        let pinned = pinned_victim == 0 && a.pinned_kept && seed_victims > 0;
        println!(
            "pinned: flips evicting a pinned seed expert {pinned_victim}, every pinned expert in its \
             load slot throughout {}, flips evicting an unpinned seed expert {seed_victims} {}",
            a.pinned_kept,
            verdict(pinned)
        );
        pinned
    }

    /// priority: the machine's copy stream runs behind the engine's. At one
    /// priority the card places a launched unpack grid's pending blocks before
    /// the engine's next launch, and the engine waits the unpack out.
    fn priority(gpu: &Gpu, a: &Run) -> Result<bool, GateError> {
        let m = a
            .machine
            .as_ref()
            .ok_or("priority: the arm has no machine")?;
        let engine = gpu.stream().priority()?;
        let copy = m.copy_stream().priority()?;
        let ok = copy > engine;
        println!(
            "priority: copy stream {copy}, engine stream {engine} (a larger value is a lower \
             priority) {}",
            verdict(ok)
        );
        Ok(ok)
    }

    /// c7: a fresh arm driven to the last boundary that made flips, so they
    /// are in flight, then reset, then the trace again from the seed.
    fn c7(
        gpu: &Gpu,
        pm: &probe_kernels::LoadedModule,
        trace: &Trace,
        a: &Run,
    ) -> Result<bool, GateError> {
        let &(b_last, _) = a.flips.last().ok_or("gate_swap: the run made no flip")?;
        let mut r = plain(gpu, pm, Faults::default(), DELAY)?;
        drive(
            gpu,
            &mut r,
            trace,
            0..b_last as usize + 1,
            Copies::Prompt,
            Hold::None,
        )?;
        let driven = r.err.is_none();
        let m = r.machine.as_mut().ok_or("gate_swap: no machine")?;
        let in_flight: Vec<Flip> = m.rule().in_flight().to_vec();
        let mut stay = 0u64;
        for f in &in_flight {
            if m.seed(LAYERS.start + f.layer)?.contains(&f.evict) {
                stay += 1;
            }
        }
        let reset = m.reset(gpu.stream(), &mut r.slots);
        let rep = match reset {
            Ok(rep) => rep,
            Err(e) => {
                println!("c7 reset after boundary {b_last}: {e} FAIL");
                return Ok(false);
            }
        };
        record::residency_reset(&rep).print();
        let mut sets_seed = true;
        let mut ledger_seed = true;
        for (i, l) in LAYERS.enumerate() {
            let mut seed = m.seed(l)?;
            let seed_len = seed.len();
            seed.sort_unstable();
            sets_seed &= card_sets(&r.slots)[i] == seed
                && m.rule()
                    .live(i)
                    .is_ok_and(|live| live.collect::<Vec<_>>() == seed);
            let row = m.ledger().row(l).unwrap_or(&[]);
            let live = row
                .iter()
                .filter(|s| matches!(s, SlotState::Live(_)))
                .count();
            let spare = row.iter().filter(|s| **s == SlotState::Spare).count();
            ledger_seed &= live == seed_len && spare == row.len() - live;
        }
        let want_dropped = (rep.copies as u64 + stay) * EXPERT_BYTES;
        r.values.clear();
        r.flips.clear();
        drive(gpu, &mut r, trace, 0..40, Copies::Prompt, Hold::None)?;
        let again = r.err.is_none() && fnvs(&r.values) == fnvs(&a.values[..40]);
        let c7 = driven
            && rep.diff == 0
            && rep.cancelled == in_flight.len()
            && rep.cancelled > 0
            && rep.copies > 0
            && rep.dropped_bytes == want_dropped
            && sets_seed
            && ledger_seed
            && again;
        println!(
            "c7 reset after boundary {b_last}: diff {} cancelled {} (in flight {}) copies {} \
             dropped_bytes {} (want {want_dropped}: copies and {stay} cancelled victims that stay); \
             card sets = the seed {sets_seed}, slot ledger = the seed's {ledger_seed}; 40 passes \
             after it = the fresh run's {again} {}",
            rep.diff,
            rep.cancelled,
            in_flight.len(),
            rep.copies,
            rep.dropped_bytes,
            verdict(c7)
        );
        Ok(c7)
    }

    /// `run` on this thread under a watchdog: past [`QUEUE_BOUND`] the
    /// watchdog prints `fail` and, after [`QUEUE_GRACE`], ends the process —
    /// a host blocked inside the driver has nothing that returns it.
    fn watched<T>(fail: String, run: impl FnOnce() -> T) -> T {
        let (done, watched) = mpsc::channel::<()>();
        let watchdog = std::thread::spawn(move || {
            if watched.recv_timeout(QUEUE_BOUND).is_err() {
                println!("{fail}");
                std::thread::sleep(QUEUE_GRACE);
                std::process::abort();
            }
        });
        let ran = run();
        let _ = done.send(());
        let _ = watchdog.join();
        ran
    }

    /// queue: an arm whose source adds `inflate` commands a part, driven to
    /// the last boundary that made flips, its window closed, then reset under
    /// a watchdog; `heavy` also asks for the tens of thousands of commands
    /// that overflow the stream's queue, `light` for more reset copies than
    /// flips kept in flight (the bound a reset without its flush would meet).
    fn queue(
        gpu: &Gpu,
        pm: &probe_kernels::LoadedModule,
        trace: &Trace,
        a: &Run,
        inflate: usize,
    ) -> Result<bool, GateError> {
        let arm = if inflate == LIGHT { "light" } else { "heavy" };
        let &(b_last, _) = a.flips.last().ok_or("gate_swap: the run made no flip")?;
        let run = || -> Result<_, GateError> {
            let faults = Faults {
                inflate,
                ..Faults::default()
            };
            let mut r = plain(gpu, pm, faults, DELAY)?;
            drive(
                gpu,
                &mut r,
                trace,
                0..b_last as usize + 1,
                Copies::Prompt,
                Hold::None,
            )?;
            let driven = r.err.is_none();
            let m = r.machine.as_mut().ok_or("gate_swap: no machine")?;
            let window = m.window();
            window.store(0, Ordering::Release);
            let t0 = Instant::now();
            let reset = m.reset(gpu.stream(), &mut r.slots);
            let took = t0.elapsed();
            window.store(1, Ordering::Release);
            Ok((driven, reset, took))
        };
        let (driven, reset, took) = watched(
            format!(
                "queue ({arm}): the clause did not finish in {QUEUE_BOUND:?}: the host blocked \
                 enqueuing copies behind copies nothing stages FAIL"
            ),
            run,
        )?;
        let rep = match reset {
            Ok(rep) => rep,
            Err(e) => {
                println!("queue ({arm}) reset after boundary {b_last}: {e} FAIL");
                return Ok(false);
            }
        };
        let commands = rep.copies * PARTS * inflate;
        let enough = if inflate == LIGHT {
            rep.copies > LAYERS.len()
        } else {
            commands >= 10_000
        };
        let ok = driven && rep.diff == 0 && enough;
        println!(
            "queue ({arm}): a reset under a closed window, {} seed copies carrying {commands} copy \
             stream commands of their sources, returned in {:.1} ms, diff {}: {}",
            rep.copies,
            took.as_secs_f64() * 1e3,
            rep.diff,
            verdict(ok)
        );
        Ok(ok)
    }

    /// tier: a two-card map, each card layer's hottest and coldest expert off
    /// the seed on the tier.
    fn tier(
        gpu: &Gpu,
        pm: &probe_kernels::LoadedModule,
        trace: &Trace,
        a: &Run,
    ) -> Result<bool, GateError> {
        let tiers = tier_sets(trace);
        let mut t = start(
            gpu,
            pm,
            seed_map(&tiers)?,
            Faults::default(),
            cfg(DELAY, DEADLINE),
        )?;
        let built = t.err.is_none();
        drive(gpu, &mut t, trace, 0..PASSES, Copies::Prompt, Hold::None)?;
        let touches = t
            .flips
            .iter()
            .filter(|(_, f)| tiers[f.layer].contains(&f.admit) || tiers[f.layer].contains(&f.evict))
            .count();
        let kept = LAYERS.enumerate().all(|(i, l)| {
            tiers[i].iter().enumerate().all(|(s, &e)| {
                t.slots.slot(l, e)
                    == Some(Slot::Tier {
                        tier: 0,
                        slot: s as u32,
                    })
            })
        });
        let bait = a
            .flips
            .iter()
            .any(|(_, f)| tiers[f.layer].first() == Some(&f.admit));
        let ok = built
            && t.err.is_none()
            && t.values.len() == PASSES
            && flips_everywhere(&t)
            && touches == 0
            && kept
            && bait
            && clean(&t.values);
        println!(
            "tier: tier experts {tiers:?}; {} flips (per layer {:?}), flips admitting or evicting a \
             tier expert {touches}, every tier entry unchanged {kept}, a tier expert the stage-only \
             run admits {bait}, errors (stale, double, miss) {:?}; error {} {}",
            t.flips.len(),
            flips_per_layer(&t),
            errs(&t.values),
            show(&t.err),
            verdict(ok)
        );
        Ok(ok)
    }

    /// staging failure: held copies, the first flip's expert fails to read.
    fn staging_failure(
        gpu: &Gpu,
        pm: &probe_kernels::LoadedModule,
        trace: &Trace,
        a: &Run,
    ) -> Result<bool, GateError> {
        let (b0, l, f) = first_flip(a)?;
        let faults = Faults {
            fail_source: Some((l, f.admit)),
            ..Faults::default()
        };
        let mut g = plain(gpu, pm, faults, DELAY)?;
        drive(gpu, &mut g, trace, 0..PASSES, Copies::Held, Hold::None)?;
        let land = b0 + DELAY;
        let said = show(&g.err);
        let named = said.contains("staging failed for job")
            && said.contains(&format!("layer {l} expert {}", f.admit))
            && said.contains(&format!("found at boundary {land}"));
        let ok = named && g.values.len() == land as usize && clean(&g.values);
        println!(
            "staging failure: layer {l} expert {}'s source fails, its flip made at {b0} lands at \
             {land} under held copies: after {} passes \"{said}\" named {named}, errors (stale, \
             double, miss) {:?} {}",
            f.admit,
            g.values.len(),
            errs(&g.values),
            verdict(ok)
        );
        Ok(ok)
    }

    /// tally: the refusals of a note and of a kept row with a slot missing.
    fn tally_clause(
        gpu: &Gpu,
        pm: &probe_kernels::LoadedModule,
        trace: &Trace,
    ) -> Result<bool, GateError> {
        let mut r = plain(gpu, pm, Faults::default(), DELAY)?;
        let m = r.machine.as_mut().ok_or("gate_swap: no machine")?;
        m.boundary(gpu.stream(), &mut r.slots)?;
        let (rows, _) = &trace.passes[0];
        let mut t = m.tally();
        t.note(LAYERS.start, 0, 0, rows[0][0][0])?;
        let twice = t.note(LAYERS.start, 0, 0, rows[0][0][1]);
        let twice_named = twice
            .as_ref()
            .is_err_and(|e| e.to_string().contains("noted twice"));
        let shapes = [
            (LAYERS.end, 0, 0),
            (LAYERS.start - 1, 0, 0),
            (LAYERS.start, MAX_ROWS, 0),
            (LAYERS.start, 0, K),
        ];
        let outside = shapes
            .iter()
            .filter(|&&(l, row, k)| t.note(l, row, k, 0).is_err())
            .count();
        let mut t = m.tally();
        let skip = (LAYERS.start + 1, 2usize);
        for (i, ids) in rows[0].iter().enumerate() {
            for (k, &id) in ids.iter().enumerate() {
                if (LAYERS.start + i, k) != skip {
                    t.note(LAYERS.start + i, 0, k, id)?;
                }
            }
        }
        let missing = m.end_pass(&mut t, KeptRows::prefix(1));
        let missing_named = missing.as_ref().is_err_and(|e| {
            e.to_string()
                .contains(&format!("row 0 at layer {}: slots [2]", skip.0))
        });
        let unbroken = m.broken().is_none();
        t.note(skip.0, 0, skip.1, rows[0][1][2])?;
        let taken = m.end_pass(&mut t, KeptRows::prefix(1)).is_ok();
        let ok = twice_named && outside == shapes.len() && missing_named && unbroken && taken;
        println!(
            "tally: a slot noted twice {:?}; notes outside the shape refused {outside} of {}; a kept \
             row missing a slot {:?}; not broken {unbroken}; the full row then taken {taken} {}",
            twice.err().map(|e| e.to_string()),
            shapes.len(),
            missing.err().map(|e| e.to_string()),
            verdict(ok)
        );
        Ok(ok)
    }

    /// keep: the tier's own `keep_rows` refuses a count with no pass open —
    /// the write side of the boundary's kept-count refusal.
    fn keep_clause(gpu: &Gpu) -> Result<bool, GateError> {
        let stacks = Stacks::new(gpu)?;
        let source = Arc::new(Synth::new(&stacks, N_L, Faults::default()));
        let slots = seed_map(&[])?;
        let view = Arc::new(DeviceTensor::upload(
            gpu.stream(),
            &slots.stage_view(),
            L,
            E,
        )?);
        let boundary = Boundary::with_rows(
            gpu.context(),
            gpu.stream(),
            BoundaryShape {
                hidden: TIER_HIDDEN,
                n_used: K,
            },
            MAX_ROWS,
        )?;
        let mut tier = HostTier::new(boundary, slots, NoExperts, L)?;
        tier.start_swap(
            gpu.context(),
            gpu.stream(),
            view,
            source,
            cfg(DELAY, DEADLINE),
        )?;
        tier.swap_boundary(gpu.stream())?
            .ok_or("gate_swap: the tier's boundary")?;
        let kept_open = tier.keep_rows(KeptRows::prefix(1), PassKind::Step).is_ok();
        tier.swap_reset(gpu.stream())?
            .ok_or("gate_swap: the tier's reset")?;
        let no_pass = tier.swap().is_some_and(|m| !m.pass_open());
        let refused = tier
            .keep_rows(KeptRows::prefix(1), PassKind::Step)
            .err()
            .map(|e| e.to_string());
        let named = refused
            .as_deref()
            .is_some_and(|e| e.contains("no boundary opened the pass"));
        let after = tier.swap_boundary(gpu.stream()).is_ok();
        let ok = kept_open && no_pass && named && after;
        println!(
            "keep: a step kept on the pass the boundary opened {kept_open}; after the reset no \
             pass open {no_pass}, that keep {refused:?} named {named}, nothing stored (the next \
             boundary {after}) {}",
            verdict(ok)
        );
        Ok(ok)
    }

    /// broken: every stack destination of the first flip's layer fails, so
    /// its boundary fails after its first change.
    fn broken(
        gpu: &Gpu,
        pm: &probe_kernels::LoadedModule,
        trace: &Trace,
        a: &Run,
    ) -> Result<bool, GateError> {
        let (b1, l, _) = first_flip(a)?;
        let faults = Faults {
            fail_dest: Some(l),
            ..Faults::default()
        };
        let mut r = plain(gpu, pm, faults, DELAY)?;
        drive(gpu, &mut r, trace, 0..PASSES, Copies::Prompt, Hold::None)?;
        let first = show(&r.err);
        let m = r.machine.as_mut().ok_or("gate_swap: no machine")?;
        let again = m
            .boundary(gpu.stream(), &mut r.slots)
            .err()
            .map(|e| e.to_string());
        let mut t = m.tally();
        let end = m
            .end_pass(&mut t, KeptRows::prefix(0))
            .err()
            .map(|e| e.to_string());
        let reset = m
            .reset(gpu.stream(), &mut r.slots)
            .err()
            .map(|e| e.to_string());
        let names = |s: &Option<String>| {
            s.as_ref()
                .is_some_and(|s| s.contains("broken") && s.contains(&format!("boundary {b1}")))
        };
        let ok = first.contains("Synth::dest")
            && r.values.len() == b1 as usize
            && m.broken().is_some()
            && names(&again)
            && names(&end)
            && names(&reset);
        println!(
            "broken: layer {l}'s stacks fail at the flip made at boundary {b1}: \"{first}\"; then \
             boundary {again:?}, end of pass {end:?}, reset {reset:?} {}",
            verdict(ok)
        );
        Ok(ok)
    }

    /// panic: the source panics on the first flip's expert.
    fn panic_clause(
        gpu: &Gpu,
        pm: &probe_kernels::LoadedModule,
        trace: &Trace,
        a: &Run,
    ) -> Result<bool, GateError> {
        let (b0, l, f) = first_flip(a)?;
        let faults = Faults {
            panic_source: Some((l, f.admit)),
            ..Faults::default()
        };
        println!(
            "panic: the staging thread's panic message below is this clause's (layer {l} expert \
             {})",
            f.admit
        );
        let mut r = plain(gpu, pm, faults, DELAY)?;
        let t0 = Instant::now();
        drive(gpu, &mut r, trace, 0..PASSES, Copies::Prompt, Hold::None)?;
        let took = t0.elapsed();
        let said = show(&r.err);
        let ok = said.contains("panicked")
            && said.contains(&format!("layer {l} expert {}", f.admit))
            && r.values.len() <= (b0 + DELAY) as usize
            && took < DEADLINE;
        println!(
            "panic: flip made at {b0}, lands at {}: after {} passes in {took:?} \"{said}\" {}",
            b0 + DELAY,
            r.values.len(),
            verdict(ok)
        );
        Ok(ok)
    }

    /// refusal: a victim the source cannot bring to the host.
    fn refusal(
        gpu: &Gpu,
        pm: &probe_kernels::LoadedModule,
        trace: &Trace,
    ) -> Result<bool, GateError> {
        let stuck: Vec<(usize, u32)> = (PINNED as u32..(N_L[0] - 1) as u32)
            .map(|e| (2, e))
            .collect();
        let faults = Faults {
            stuck,
            ..Faults::default()
        };
        let mut r = plain(gpu, pm, faults, DELAY)?;
        let built = r.err.is_none();
        drive(gpu, &mut r, trace, 0..PASSES, Copies::Prompt, Hold::None)?;
        let said = show(&r.err);
        let named = said.contains("is not host-resident") && said.contains("layer 2 expert");
        let tail = Faults {
            stuck: vec![(3usize, (N_L[1] - 1) as u32)],
            ..Faults::default()
        };
        let at_load = plain(gpu, pm, tail, DELAY)?;
        let load_said = show(&at_load.err);
        let load_named = load_said.contains("is not host-resident")
            && load_said.contains("layer 3 expert 11")
            && at_load.machine.is_none();
        let refusal = built && named && load_named;
        println!(
            "refusal: after {} passes \"{said}\" {named}; a spare slot's expert at load \"{load_said}\" \
             {load_named} {}",
            r.values.len(),
            verdict(refusal)
        );
        Ok(refusal)
    }

    /// fault code: a driver error the copy stream's query returns at a drop
    /// is a fault leak that carries the error's `CUresult`, and the
    /// `residency leak` record prints it — the driver's own text asks the
    /// context at print time, which a faulted context may not answer, so the
    /// code it already gave is the fact to keep. The drop's arm is
    /// `Leak::fault`, driven here with the code a sticky context fault
    /// carries; no card has to fault for the clause.
    fn fault_code() -> Result<bool, GateError> {
        const CODE: sys::cudaError_enum = sys::cudaError_enum_CUDA_ERROR_ILLEGAL_ADDRESS;
        let (ring, words) = (65536u64, 512u64);
        let leak = Leak::fault(
            &GpuError::Driver {
                op: None,
                source: DriverError(CODE),
            },
            ring,
            words,
        );
        let line = record::residency_leak(&leak).line();
        let ok = leak
            == Leak {
                reason: LeakReason::Fault,
                code: Some(CODE),
                ring_bytes: ring,
                words_bytes: words,
            }
            && line == format!("residency leak reason=fault code={CODE} ring={ring} words={words}");
        println!(
            "fault code: a query returning {CODE}: the leak's reason {:?} code {:?}, the record \
             \"{line}\" {}",
            leak.reason.word(),
            leak.code,
            verdict(ok)
        );
        Ok(ok)
    }

    /// stall: a victim that takes longer to prepare than the machine's
    /// deadline is a named error at the boundary it would land at, not a
    /// wait without end; and the machine, dropped with that preparation in
    /// flight, leaves no copy waiting on the card, and its staging thread,
    /// left to finish the preparation, is never the last owner of the shared
    /// state (it would free the ring's pinned pages and the source there,
    /// inside the driver beside the next clause). The clause ends only once
    /// that thread has. A copy nobody releases holds every later free on the
    /// card, so on that fail the gate ends here, by name.
    fn stall(
        gpu: &Gpu,
        pm: &probe_kernels::LoadedModule,
        trace: &Trace,
    ) -> Result<bool, GateError> {
        let faults = Faults {
            slow: vec![(2usize, 7u32)],
            ..Faults::default()
        };
        let dropped_on = Arc::clone(&faults.dropped_on);
        let threads = staging_threads()?;
        let leaks_before = leaks()?.len();
        let mut st = start(gpu, pm, seed_map(&[])?, faults, cfg(DELAY, STALL_DEADLINE))?;
        let t0 = Instant::now();
        drive(gpu, &mut st, trace, 0..PASSES, Copies::Prompt, Hold::None)?;
        let took = t0.elapsed();
        let stall_said = show(&st.err);
        let m = st.machine.take().ok_or("gate_swap: no machine")?;
        let ev = gpu.context().new_event(None)?;
        ev.record(m.copy_stream())?;
        let t1 = Instant::now();
        drop(m);
        let dropped = t1.elapsed();
        let t2 = Instant::now();
        let mut drained = ev.query()?;
        while !drained && t2.elapsed() < RELEASED {
            std::thread::sleep(Duration::from_micros(200));
            drained = ev.query()?;
        }
        let drain_took = t2.elapsed();
        let mut left = staging_threads()?;
        while left > threads && t1.elapsed() < JOINED {
            std::thread::sleep(Duration::from_millis(5));
            left = staging_threads()?;
        }
        let joined = left <= threads;
        let on = dropped_on
            .lock()
            .map_err(|_| "gate_swap: the drop cell")?
            .clone();
        let leaked: Vec<LeakReason> = leaks()?[leaks_before..].iter().map(|l| l.reason).collect();
        let stall = stall_said.contains("was not prepared for the host")
            && stall_said.contains("layer 2: the victim 7")
            && took < SLOW
            && drained
            && dropped < SLOW
            && joined
            && on.as_deref() != Some("swap-staging")
            && leaked == [LeakReason::Join];
        println!(
            "stall: a victim {SLOW:?} to prepare under a {STALL_DEADLINE:?} deadline: after {} passes \
             in {took:?} \"{stall_said}\"; dropped in {dropped:?}, the copy stream {}; its staging \
             thread {} {:?} after the drop, the source dropped on {on:?} (never swap-staging), \
             leaks reported {leaked:?} (want [Join]) {}",
            st.values.len(),
            if drained {
                format!("drained {drain_took:?} after it")
            } else {
                format!("still waiting {RELEASED:?} after it")
            },
            if joined { "ended" } else { "still running" },
            t1.elapsed(),
            verdict(stall)
        );
        if !drained {
            println!(
                "gate_swap: FAIL — a dropped machine left a copy waiting on the card; every later \
                 free on the card would wait on it, so the gate ends here"
            );
            std::process::exit(1);
        }
        Ok(stall)
    }

    /// Synchronizes the context when it drops, as the V4.1 body's ring
    /// shadows do — through the capture-safe owner, which a drop path calls
    /// because it must not panic.
    struct CtxSync(Arc<CudaContext>);

    impl Drop for CtxSync {
        fn drop(&mut self) {
            if let Err(e) = bloomery_gpu::capsync::ctx_sync_in_drop(&self.0, "gate_swap CtxSync") {
                println!("dropq: the context synchronize failed: {e}");
            }
        }
    }

    /// Host experts for the tier arm's `HostTier`: no service runs there, so
    /// one writes the zeros an empty list does.
    struct NoExperts;

    impl HostExperts for NoExperts {
        fn experts_into(
            &mut self,
            _layer: usize,
            _x: &Tensor2,
            _experts: &[(u32, f32)],
            out: &mut [f32],
        ) -> Result<(), GpuError> {
            out.fill(0.0);
            Ok(())
        }
    }

    /// An owner that drops the context's synchronize before its machine, as
    /// `Body` declares `shadows` before `hybrid`: its drop stops the machine
    /// first, as `HostTier::stop_swap` does for `Body`, `GpuModel` and the
    /// host tier.
    struct Dropq {
        _sync: CtxSync,
        machine: Option<SwapMachine>,
    }

    impl Drop for Dropq {
        fn drop(&mut self) {
            drop(self.machine.take());
        }
    }

    /// dropq: `a`'s first flip boundary issued under a closed window, so its
    /// copies wait on staging words the staging thread will not raise until
    /// the flips are due; then the machine dropped inside a [`Dropq`] under
    /// the queue arm's watchdog ([`watched`]).
    fn dropq(
        gpu: &Gpu,
        pm: &probe_kernels::LoadedModule,
        trace: &Trace,
        a: &Run,
    ) -> Result<bool, GateError> {
        let (b_first, _, _) = first_flip(a)?;
        let mut r = plain(gpu, pm, Faults::default(), DELAY)?;
        drive(
            gpu,
            &mut r,
            trace,
            0..b_first as usize,
            Copies::Prompt,
            Hold::None,
        )?;
        let driven = r.err.is_none();
        let mut m = r.machine.take().ok_or("gate_swap: no machine")?;
        let window = m.window();
        window.store(0, Ordering::Release);
        let made = m.boundary(gpu.stream(), &mut r.slots)?.made;
        std::thread::sleep(HOLD_SETTLE);
        let waiting = m.copy_stream().query() == Ok(false);
        let leaks_before = leaks()?.len();
        let owner = Dropq {
            _sync: CtxSync(Arc::clone(gpu.context())),
            machine: Some(m),
        };
        let took = watched(
            format!(
                "dropq: the drop did not finish in {QUEUE_BOUND:?}: the context synchronized \
                 while a copy waited on a staging word the machine had not released FAIL"
            ),
            || {
                let t0 = Instant::now();
                drop(owner);
                t0.elapsed()
            },
        );
        let leaked = leaks()?.len() - leaks_before;
        let ok = driven && made > 0 && waiting && took < DEADLINE && leaked == 0;
        println!(
            "dropq: boundary {b_first} made {made} flips under a closed window, the copy stream \
             {} before the drop; the machine dropped before the context synchronize in {:.1} ms \
             (deadline {DEADLINE:?}), {leaked} leaks: {}",
            if waiting { "waiting" } else { "idle" },
            took.as_secs_f64() * 1e3,
            verdict(ok)
        );
        Ok(ok)
    }

    /// dropq (tier): the tier arm of the drop order — a real [`HostTier`]
    /// holding a machine with a copy queued behind a staging word, driven to
    /// the first flip boundary through the tier's own `keep_rows` and
    /// `swap_boundary`, then dropped as the engine drops a tier: its `Drop`
    /// stops the machine before any field frees, so the drop ends within the
    /// machine's deadline with no leak.
    fn dropq_tier(gpu: &Gpu, trace: &Trace, a: &Run) -> Result<bool, GateError> {
        let (b_first, _, _) = first_flip(a)?;
        // The stacks and the map outlive the tier, as the model's weights
        // and its map copy do; the tier is dropped inside the clause.
        let stacks = Stacks::new(gpu)?;
        let source = Arc::new(Synth::new(&stacks, N_L, Faults::default()));
        let slots = seed_map(&[])?;
        let view = Arc::new(DeviceTensor::upload(
            gpu.stream(),
            &slots.stage_view(),
            L,
            E,
        )?);
        let boundary = Boundary::with_rows(
            gpu.context(),
            gpu.stream(),
            BoundaryShape {
                hidden: TIER_HIDDEN,
                n_used: K,
            },
            MAX_ROWS,
        )?;
        let mut tier = HostTier::new(boundary, slots, NoExperts, L)?;
        tier.start_swap(
            gpu.context(),
            gpu.stream(),
            view,
            source,
            cfg(DELAY, DEADLINE),
        )?;
        let window = tier.swap().ok_or("gate_swap: the tier's machine")?.window();
        for (rows, kept) in &trace.passes[..b_first as usize] {
            tier.swap_boundary(gpu.stream())?
                .ok_or("gate_swap: the tier's boundary")?;
            note_rows(
                tier.swap_tally().ok_or("gate_swap: the tier's tally")?,
                rows,
            )?;
            tier.keep_rows(KeptRows::prefix(*kept), PassKind::Driver)?;
        }
        window.store(0, Ordering::Release);
        let made = tier
            .swap_boundary(gpu.stream())?
            .ok_or("gate_swap: the tier's boundary")?
            .1
            .made;
        std::thread::sleep(HOLD_SETTLE);
        let waiting = tier
            .swap()
            .is_some_and(|m| m.copy_stream().query() == Ok(false));
        let leaks_before = leaks()?.len();
        let took = watched(
            format!(
                "dropq (tier): the drop did not finish in {QUEUE_BOUND:?}: the tier freed its \
                 boundary while a copy waited on a staging word the machine had not released — \
                 production stop_swap's order FAIL"
            ),
            || {
                let t0 = Instant::now();
                drop(tier);
                t0.elapsed()
            },
        );
        let leaked = leaks()?.len() - leaks_before;
        let ok = made > 0 && waiting && took < DEADLINE && leaked == 0;
        println!(
            "dropq (tier): a HostTier whose boundary, map and passes drove {} flips (boundary \
             {b_first} made {made} under a closed window, the copy stream {} before the drop): the \
             tier dropped, its machine stopped first, in {:.1} ms (deadline {DEADLINE:?}), \
             {leaked} leaks: {}",
            a.flips.len(),
            if waiting { "waiting" } else { "idle" },
            took.as_secs_f64() * 1e3,
            verdict(ok)
        );
        Ok(ok)
    }

    /// dropq (free): a plain [`DeviceBuffer`] free with the same copy queued
    /// behind a staging word, timed on its own thread so the main thread can
    /// still drop the machine — the only thing that would end a block. The
    /// line names the outcome either way; it never fails the clause, for the
    /// stop-before-any-free order rests on what it measures.
    fn dropq_free(
        gpu: &Gpu,
        pm: &probe_kernels::LoadedModule,
        trace: &Trace,
        a: &Run,
    ) -> Result<bool, GateError> {
        let (b_first, _, _) = first_flip(a)?;
        let mut r = plain(gpu, pm, Faults::default(), DELAY)?;
        drive(
            gpu,
            &mut r,
            trace,
            0..b_first as usize,
            Copies::Prompt,
            Hold::None,
        )?;
        let driven = r.err.is_none();
        let mut machine = Some(r.machine.take().ok_or("gate_swap: no machine")?);
        let window = machine.as_ref().ok_or("gate_swap: no machine")?.window();
        window.store(0, Ordering::Release);
        let made = machine
            .as_mut()
            .ok_or("gate_swap: no machine")?
            .boundary(gpu.stream(), &mut r.slots)?
            .made;
        std::thread::sleep(HOLD_SETTLE);
        let waiting = machine
            .as_ref()
            .is_some_and(|m| m.copy_stream().query() == Ok(false));
        let leaks_before = leaks()?.len();
        // A small synchronous allocation, its zeroing done: a plain cuMemFree
        // in `DeviceBuffer`'s drop, the free the model's graphs and heads
        // make — of memory the queued copy never touches.
        let buf = DeviceBuffer::<u32>::zeroed(gpu.stream(), 1)?;
        gpu.stream().synchronize()?;
        let (freed, back) = mpsc::channel::<Duration>();
        let freeing = std::thread::spawn(move || {
            let t0 = Instant::now();
            drop(buf);
            let _ = freed.send(t0.elapsed());
        });
        let release = format!(
            "dropq (free): the machine's drop did not finish in {QUEUE_BOUND:?}: the stop that \
             releases every staging wait did not return FAIL"
        );
        let stop = |machine: &mut Option<SwapMachine>| {
            watched(release, || {
                let t0 = Instant::now();
                drop(machine.take());
                t0.elapsed()
            })
        };
        let (free_took, blocked, machine_ms) = match back.recv_timeout(FREE_LOOK) {
            Ok(took) => {
                let _ = freeing.join();
                let ms = stop(&mut machine).as_secs_f64() * 1e3;
                (Some(took), false, ms)
            }
            Err(_) => {
                // The free is still behind the queued copy: the machine's
                // drop releases the staging waits, the copy runs, and the
                // block ends — the free's own thread reports when.
                let ms = stop(&mut machine).as_secs_f64() * 1e3;
                let took = back.recv_timeout(QUEUE_BOUND).ok();
                if took.is_some() {
                    let _ = freeing.join();
                }
                (took, true, ms)
            }
        };
        let leaked = leaks()?.len() - leaks_before;
        let ok = driven && made > 0 && waiting && leaked == 0;
        let free = if !blocked {
            format!(
                "returned in {:.0} µs while the copy was queued",
                free_took.map_or(0.0, |d| d.as_secs_f64() * 1e6)
            )
        } else {
            format!(
                "blocked for at least {FREE_LOOK:?}, the machine's drop ({machine_ms:.1} ms) {} \
                 it",
                if free_took.is_some() {
                    format!(
                        "then ended it, the free returning in {:.0} µs",
                        free_took.map_or(0.0, |d| d.as_secs_f64() * 1e6)
                    )
                } else {
                    format!("did not end it within {QUEUE_BOUND:?}")
                }
            )
        };
        println!(
            "dropq (free): boundary {b_first} made {made} flips under a closed window, the copy \
             stream {} before it; a plain free of an unrelated buffer {free}; the machine's own \
             drop took {machine_ms:.1} ms, {leaked} leaks: {}",
            if waiting { "waiting" } else { "idle" },
            verdict(ok)
        );
        Ok(ok)
    }

    // ------------------------------------------------------------- calls

    /// The trace passes a call arm runs before its call, and the call's
    /// steps: each a pick of every layer from the step's ids, then a probe.
    const CALL_BEFORE: usize = 40;
    const CALL_STEPS: Range<usize> = 40..48;
    /// Passes a call arm runs after its call.
    const CALL_AFTER: Range<usize> = 48..68;

    /// What a call arm saw: per step its value and its card sets after the
    /// step's picks, the picks, whether the held engine stream was still
    /// waiting when the arm looked, the call's report and its error.
    #[derive(Default)]
    struct CallSeen {
        values: Vec<Value>,
        sets: Vec<Vec<Vec<u32>>>,
        picks: Vec<CallPick>,
        start: Vec<Vec<u32>>,
        held_waited: Option<bool>,
        report: Option<CallReport>,
        err: Option<String>,
    }

    /// One call on `run`'s machine over `steps`, in the prefill's order: a
    /// boundary opens the call's pass; per step the ids refreshed, each
    /// layer's pick from the step's counts (at most `cap` wanted, floor 1),
    /// the engine stream's wait for each layer's landed event, the probe, a
    /// bounded drain and the readback, then each layer's reader; the call
    /// ends `kept` or not, and its pass with no row kept. `hold` holds the
    /// copy stream from before the first step's picks until after its
    /// probe's launch.
    fn call(
        gpu: &Gpu,
        run: &mut Run,
        steps: &[(Vec<[[u32; K]; L]>, usize)],
        cap: usize,
        kept: bool,
        hold: Option<&HostFlags>,
    ) -> Result<CallSeen, GateError> {
        let r = call_held(gpu, run, steps, cap, kept, hold);
        if let Some(flags) = hold {
            flags.raise(0)?;
        }
        r
    }

    /// [`call`]'s steps; `call` raises the hold's flag after.
    fn call_held(
        gpu: &Gpu,
        run: &mut Run,
        steps: &[(Vec<[[u32; K]; L]>, usize)],
        cap: usize,
        kept: bool,
        hold: Option<&HostFlags>,
    ) -> Result<CallSeen, GateError> {
        let stream = gpu.stream();
        let mut seen = CallSeen {
            start: card_sets(&run.slots),
            ..CallSeen::default()
        };
        let m = run.machine.as_mut().ok_or("gate_swap: no machine")?;
        let fail = |seen: &mut CallSeen, e: String| seen.err = Some(e);
        if let Err(e) = m.boundary(stream, &mut run.slots) {
            fail(&mut seen, format!("the call's boundary: {e}"));
            return Ok(seen);
        }
        m.begin_call(stream, CallCfg { floor: 1 })?;
        for (step, (rows, _)) in steps.iter().enumerate() {
            run.card.refresh(gpu, rows)?;
            if step == 0
                && let Some(flags) = hold
            {
                flags.clear(0)?;
                flags.enqueue_wait(m.copy_stream(), 0)?;
            }
            for (i, l) in LAYERS.enumerate() {
                let mut counts = vec![0u32; E];
                for row in rows {
                    for &id in &row[i] {
                        counts[id as usize] += 1;
                    }
                }
                match m.call_pick(stream, &mut run.slots, l, &counts, cap) {
                    Ok(p) => seen.picks.push(p),
                    Err(e) => {
                        fail(&mut seen, format!("step {step} layer {l}: {e}"));
                        return Ok(seen);
                    }
                }
            }
            for l in LAYERS {
                stream.wait(m.call_landed(l)?)?;
            }
            run.card.launch(gpu)?;
            if step == 0
                && let Some(flags) = hold
            {
                std::thread::sleep(HOLD_SETTLE);
                seen.held_waited = Some(stream.query() == Ok(false));
                flags.raise(0)?;
            }
            if let Err(e) = drain(gpu) {
                fail(&mut seen, format!("step {step}: {e}"));
                return Ok(seen);
            }
            let out = run.card.read(gpu)?;
            seen.values.push(value(rows, &out, &run.slots));
            seen.sets.push(card_sets(&run.slots));
            for l in LAYERS {
                m.call_reader(l, stream)?;
            }
        }
        match m.end_call(stream, &mut run.slots, kept) {
            Ok(rep) => seen.report = Some(rep),
            Err(e) => {
                fail(&mut seen, format!("end_call: {e}"));
                return Ok(seen);
            }
        }
        let mut tally = m.tally();
        m.end_pass(&mut tally, KeptRows::prefix(0))?;
        Ok(seen)
    }

    /// A call arm: every expert host-resident (the churn pool is in the
    /// host set), `CALL_BEFORE` passes of the trace, then its call.
    fn call_arm(
        gpu: &Gpu,
        pm: &probe_kernels::LoadedModule,
        trace: &Trace,
        faults: Faults,
        cap: usize,
        kept: bool,
        hold: Option<&HostFlags>,
    ) -> Result<(Run, CallSeen), GateError> {
        let faults = Faults {
            all_resident: true,
            ..faults
        };
        let mut r = plain(gpu, pm, faults, DELAY)?;
        drive(
            gpu,
            &mut r,
            trace,
            0..CALL_BEFORE,
            Copies::Prompt,
            Hold::None,
        )?;
        if let Some(e) = &r.err {
            return Err(format!("gate_swap: the call arm before its call: {e}").into());
        }
        let steps = &trace.passes[CALL_STEPS];
        let seen = call(gpu, &mut r, steps, cap, kept, hold)?;
        Ok((r, seen))
    }

    /// s1 scripted and s3 host map: a kept call's steps, pick by pick,
    /// equal a static replay of the card sets they left (synchronous copies,
    /// other slot numbers, a fresh map), with no stale sum and no id served
    /// twice or not at all — the host map moves at the pick, never a layer
    /// later — and every pick lands before the passes after the call, which
    /// run clean; a call not kept returns every layer to its start set, and
    /// the passes after it run clean too (its mutants: the host map admits
    /// each expert into another pair's slot; the host map moves at the
    /// layer's reader).
    fn s1_s3(
        gpu: &Gpu,
        pm: &probe_kernels::LoadedModule,
        trace: &Trace,
    ) -> Result<bool, GateError> {
        let (mut r, seen) = call_arm(gpu, pm, trace, Faults::default(), usize::MAX, true, None)?;
        let sub = Trace {
            passes: trace.passes[CALL_STEPS].to_vec(),
        };
        let replay = static_replay(gpu, pm, &sub, &seen.sets)?;
        let admitted: usize = seen.picks.iter().map(|p| p.admitted).sum();
        let landed = LAYERS.clone().all(|l| {
            r.machine
                .as_ref()
                .and_then(|m| m.ledger().landing(l))
                .is_none()
        });
        let before_after = r.values.len();
        drive(gpu, &mut r, trace, CALL_AFTER, Copies::Prompt, Hold::None)?;
        let after = r.values[before_after..].to_vec();
        let s1 = seen.err.is_none()
            && r.err.is_none()
            && admitted > 0
            && seen.values.len() == CALL_STEPS.len()
            && fnvs(&seen.values) == fnvs(&replay)
            && clean(&replay)
            && landed
            && clean(&after);
        let s3 = seen.err.is_none() && clean(&seen.values);
        let (mut n, restored) =
            call_arm(gpu, pm, trace, Faults::default(), usize::MAX, false, None)?;
        let back = n.err.is_none()
            && restored.err.is_none()
            && card_sets(&n.slots) == restored.start
            && restored
                .report
                .is_some_and(|rep| !rep.kept && rep.restored > 0);
        let before_n = n.values.len();
        drive(gpu, &mut n, trace, CALL_AFTER, Copies::Prompt, Hold::None)?;
        let back = back && n.err.is_none() && clean(&n.values[before_n..]);
        if let Some(rep) = seen.report {
            record::call_report(&rep).print();
        }
        println!(
            "s1 scripted: a kept call of {} steps, {admitted} experts admitted: step values = the \
             static replay's {}, replay clean {}, every pick landed by the end {landed}, {} passes \
             after it (stale, double, miss) {:?}; not kept: every layer back at its start set, then \
             clean {back}; errors {} / {} / {} {}",
            CALL_STEPS.len(),
            fnvs(&seen.values) == fnvs(&replay),
            clean(&replay),
            after.len(),
            errs(&after),
            show(&seen.err),
            show(&r.err),
            show(&restored.err),
            verdict(s1 && back)
        );
        println!(
            "s3 host map at the pick: the call's steps (stale, double, miss) {:?} {}",
            errs(&seen.values),
            verdict(s3)
        );
        Ok(s1 && back && s3)
    }

    /// s2 landing wait: the copy stream held from before a call's first
    /// picks (two wanted a layer) until after the first step's probe launch:
    /// the engine stream is still waiting on the layers' landed events when
    /// the arm looks, and after the release the probe reads no stale sum
    /// (its mutant: the landed event recorded before the pick's copies).
    fn s2(gpu: &Gpu, pm: &probe_kernels::LoadedModule, trace: &Trace) -> Result<bool, GateError> {
        let flags = HostFlags::new(gpu.context(), 1)?;
        // Two wanted a layer: the first step's jobs stay under the machine's
        // backlog bound while the copy stream holds the ring.
        let (r, seen) = call_arm(gpu, pm, trace, Faults::default(), 2, true, Some(&flags))?;
        let first = seen.values.first().copied().unwrap_or_default();
        let admitted = seen.picks.iter().take(L).map(|p| p.admitted).sum::<usize>();
        let ok = seen.err.is_none()
            && r.err.is_none()
            && admitted > 0
            && seen.held_waited == Some(true)
            && clean(&seen.values);
        println!(
            "s2 landing wait: copy stream held over the first step's picks ({admitted} admitted): \
             engine stream still waiting {:?}; first step (stale, double, miss) ({}, {}, {}), all \
             steps {:?}; error {} {}",
            seen.held_waited,
            first.stale,
            first.double,
            first.miss,
            errs(&seen.values),
            show(&seen.err),
            verdict(ok)
        );
        Ok(ok)
    }

    /// s4 victim refusal: a pool resident the source cannot serve from
    /// resident pages is refused by name at the pick that would evict it,
    /// with the machine unbroken and the host map unmoved (its mutant: the
    /// victims not checked).
    fn s4(gpu: &Gpu, pm: &probe_kernels::LoadedModule, trace: &Trace) -> Result<bool, GateError> {
        // Layer 2's pool residents; its spare's expert (the last seed slot)
        // must be resident for the machine to start. No pass before the
        // call: a flip would refuse the same victims at its boundary.
        let stuck: Vec<(usize, u32)> = (PINNED as u32..(N_L[0] - 1) as u32)
            .map(|e| (2, e))
            .collect();
        let faults = Faults {
            stuck,
            all_resident: true,
            ..Faults::default()
        };
        let mut r = plain(gpu, pm, faults, DELAY)?;
        if let Some(e) = &r.err {
            return Err(format!("gate_swap: s4's machine: {e}").into());
        }
        let before = r.slots.clone();
        let stream = gpu.stream();
        let m = r.machine.as_mut().ok_or("gate_swap: no machine")?;
        m.boundary(stream, &mut r.slots)?;
        m.begin_call(stream, CallCfg { floor: 1 })?;
        let (rows, _) = &trace.passes[CALL_STEPS.start];
        let mut counts = vec![0u32; E];
        for row in rows {
            for &id in &row[0] {
                counts[id as usize] += 1;
            }
        }
        let said = match m.call_pick(stream, &mut r.slots, LAYERS.start, &counts, usize::MAX) {
            Ok(p) => format!("picked, {} admitted", p.admitted),
            Err(e) => e.to_string(),
        };
        let named = said.contains("is not host-resident") && said.contains("layer 2 expert");
        let unbroken = m.broken().is_none();
        let unmoved = r.slots == before;
        let ended = m.end_call(stream, &mut r.slots, true).is_ok();
        let s4 = named && unbroken && unmoved && ended;
        println!(
            "s4 victim refusal: \"{said}\" named {named}, machine unbroken {unbroken}, host map \
             unmoved {unmoved}, the call ends {ended} {}",
            verdict(s4)
        );
        Ok(s4)
    }

    /// s5 floor: a call whose floor is raised between two picks admits no
    /// expert whose count is under the new floor, and a floor set with no
    /// call open is refused by name (its mutant: the pick reads the floor
    /// the call began with, not the one set since). `set_call_floor` refuses
    /// no floor value itself — 0 is a floor every count passes.
    fn s5(gpu: &Gpu, pm: &probe_kernels::LoadedModule) -> Result<bool, GateError> {
        // Counts built for the clause, not the trace's: the contract is the
        // floor against the ranking, and the ranking needs counts the trace
        // need not give. Layer 2's pool at the call's start is the seed's
        // experts 4..=10 (the pinned 0..=3 never a victim, the spare's
        // expert 11 off the card).
        const HOT: u32 = 20;
        const WARM: u32 = 21;
        const UNDER: u32 = 22;
        const OVER: u32 = 23;
        const FLOOR2: u32 = 2;
        let faults = Faults {
            all_resident: true,
            ..Faults::default()
        };
        let mut r = plain(gpu, pm, faults, DELAY)?;
        if let Some(e) = &r.err {
            return Err(format!("gate_swap: s5's machine: {e}").into());
        }
        let stream = gpu.stream();
        let m = r.machine.as_mut().ok_or("gate_swap: no machine")?;
        m.boundary(stream, &mut r.slots)?;
        let refused = match m.set_call_floor(0) {
            Ok(()) => "set".to_string(),
            Err(e) => e.to_string(),
        };
        let refused_named = refused.contains("SwapMachine::set_call_floor")
            && refused.contains("a prompt call open");
        m.begin_call(stream, CallCfg { floor: 1 })?;
        // The first pick at floor 1: two off-seed experts over zero-count
        // victims, counts 10 and 1.
        let mut c1 = vec![0u32; E];
        c1[HOT as usize] = 10;
        c1[WARM as usize] = 1;
        let first = m.call_pick(stream, &mut r.slots, LAYERS.start, &c1, usize::MAX)?;
        for l in LAYERS {
            m.call_reader(l, stream)?;
        }
        let between = card_sets(&r.slots);
        m.set_call_floor(FLOOR2)?;
        // The second pick at floor 2: of its off-card counts, 5 passes the
        // floor and 1 does not.
        let mut c2 = vec![0u32; E];
        c2[UNDER as usize] = 1;
        c2[OVER as usize] = 5;
        let second = m.call_pick(stream, &mut r.slots, LAYERS.start, &c2, usize::MAX)?;
        let after = card_sets(&r.slots);
        let gained: Vec<u32> = after[0]
            .iter()
            .filter(|id| !between[0].contains(id))
            .copied()
            .collect();
        let under: Vec<u32> = gained
            .iter()
            .filter(|&&id| c2[id as usize] < FLOOR2)
            .copied()
            .collect();
        let ended = m.end_call(stream, &mut r.slots, true).is_ok();
        let ok = refused_named
            && first.admitted == 2
            && second.admitted == 1
            && gained == [OVER]
            && under.is_empty()
            && ended;
        println!(
            "s5 floor: floor 1 then {FLOOR2}: the first pick admitted {} experts, the second \
             admitted {} (gained {gained:?}, of them under the floor {under:?}); set_call_floor(0) \
             with no call open \"{refused}\" named {refused_named}, the call ends {ended} {}",
            first.admitted,
            second.admitted,
            verdict(ok)
        );
        Ok(ok)
    }

    // ----------------------------------------------------------- evicted

    /// Layer 2's experts `ids`, for a fault's list.
    fn layer2(ids: Range<u32>) -> Vec<(usize, u32)> {
        ids.map(|e| (2, e)).collect()
    }

    /// A call arm with no pass before its call: every expert host-resident
    /// from the load bar `faults`' evicted and stuck ones, the call over
    /// passes `steps` of the trace (`kept` or not), then, when it ended,
    /// the passes `CALL_AFTER`. No boundary before the call lands a flip, so
    /// none takes one of the call's victims first.
    fn fresh_call(
        gpu: &Gpu,
        pm: &probe_kernels::LoadedModule,
        trace: &Trace,
        faults: Faults,
        steps: Range<usize>,
        kept: bool,
    ) -> Result<(Run, CallSeen), GateError> {
        let faults = Faults {
            all_resident: true,
            ..faults
        };
        let mut r = plain(gpu, pm, faults, DELAY)?;
        if let Some(e) = &r.err {
            return Err(format!("gate_swap: a fresh call arm's machine: {e}").into());
        }
        let seen = call(gpu, &mut r, &trace.passes[steps], usize::MAX, kept, None)?;
        if seen.err.is_none() {
            drive(gpu, &mut r, trace, CALL_AFTER, Copies::Prompt, Hold::None)?;
        }
        Ok((r, seen))
    }

    /// A fresh call arm `(r, seen)` ran to its end with the values, card
    /// sets and flips of its twin `(t, tseen)` without the fault, clean.
    fn call_twin(r: &Run, seen: &CallSeen, t: &Run, tseen: &CallSeen) -> bool {
        seen.err.is_none()
            && r.err.is_none()
            && tseen.err.is_none()
            && t.err.is_none()
            && fnvs(&seen.values) == fnvs(&tseen.values)
            && seen.sets == tseen.sets
            && fnvs(&r.values) == fnvs(&t.values)
            && r.values.len() == CALL_AFTER.len()
            && r.flips == t.flips
            && clean(&seen.values)
            && clean(&r.values)
    }

    /// The experts read in again that `run`'s boundaries reported.
    fn rereads(run: &Run) -> usize {
        run.reports.iter().map(|p| p.rereads).sum()
    }

    /// The experts read in again that the first boundary after a fresh
    /// call arm's call reported: the call's own, as no flip lands there.
    fn call_rereads(run: &Run) -> usize {
        run.reports.first().map_or(0, |p| p.rereads)
    }

    /// Experts the twin's picks admitted at layer 2.
    fn admitted_l2(seen: &CallSeen) -> usize {
        seen.picks
            .iter()
            .filter(|p| p.layer == 2)
            .map(|p| p.admitted)
            .sum()
    }

    /// evicted: a victim whose pages the page cache let go after the load
    /// or after the staging thread's prepare (a host set populated, not
    /// locked) is read back in by the machine, never refused — at a
    /// boundary (layer 2's pool, whose flips' victims the staging thread
    /// prepares and loses again), at a call's pick (layer 2's pool, not
    /// resident from the load) and at a call's end (layer 2's host experts,
    /// which a one-step call admits and its end, not kept, sends back; one
    /// step, so no later pick takes one as its victim first) each arm runs
    /// to its end with its twin's values without the fault (its mutant: no
    /// prepare in the machine's one decision, which refuses each by name).
    /// Each arm reports its victims read in again, its twin and the runs
    /// without a fault `clean_runs` none (a count that fires on a resident
    /// victim is red).
    fn evicted(
        gpu: &Gpu,
        pm: &probe_kernels::LoadedModule,
        trace: &Trace,
        a: &Run,
        clean_runs: &[&Run],
    ) -> Result<bool, GateError> {
        let pool = layer2(PINNED as u32..(N_L[0] - 1) as u32);
        let faults = Faults {
            evicted: pool.clone(),
            ..Faults::default()
        };
        let mut r = plain(gpu, pm, faults, DELAY)?;
        drive(gpu, &mut r, trace, 0..PASSES, Copies::Prompt, Hold::None)?;
        let at_boundary = r.err.is_none()
            && r.values.len() == PASSES
            && fnvs(&r.values) == fnvs(&a.values)
            && r.flips == a.flips
            && clean(&r.values)
            && rereads(&r) > 0;
        println!(
            "evicted boundary: layer 2's pool lost after the staging thread's prepare, {} passes, \
             values and flips equal the prompt run {}, {} victims read in again; error {} {}",
            r.values.len(),
            fnvs(&r.values) == fnvs(&a.values) && r.flips == a.flips,
            rereads(&r),
            show(&r.err),
            verdict(at_boundary)
        );

        let (tp, tpseen) = fresh_call(gpu, pm, trace, Faults::default(), CALL_STEPS, true)?;
        let faults = Faults {
            evicted: pool,
            ..Faults::default()
        };
        let (p, pseen) = fresh_call(gpu, pm, trace, faults, CALL_STEPS, true)?;
        let at_pick = admitted_l2(&tpseen) > 0
            && call_twin(&p, &pseen, &tp, &tpseen)
            && call_rereads(&p) > 0
            && rereads(&tp) == 0;
        println!(
            "evicted pick: layer 2's pool not resident from the load, a kept call of {} steps \
             ({} admitted at layer 2 in the twin), then {} passes: equal to the twin {}, the call's \
             victims read in again {} (twin {}); error {} {}",
            CALL_STEPS.len(),
            admitted_l2(&tpseen),
            p.values.len(),
            call_twin(&p, &pseen, &tp, &tpseen),
            call_rereads(&p),
            rereads(&tp),
            show(&pseen.err),
            verdict(at_pick)
        );

        let one = CALL_STEPS.start..CALL_STEPS.start + 1;
        let (te, teseen) = fresh_call(gpu, pm, trace, Faults::default(), one.clone(), false)?;
        let faults = Faults {
            evicted: layer2(N_L[0] as u32..E as u32),
            ..Faults::default()
        };
        let (e, eseen) = fresh_call(gpu, pm, trace, faults, one, false)?;
        let restored = teseen.report.map_or(0, |rep| rep.restored);
        let at_end = admitted_l2(&teseen) > 0
            && restored > 0
            && call_twin(&e, &eseen, &te, &teseen)
            && call_rereads(&e) > 0
            && rereads(&te) == 0;
        println!(
            "evicted end: layer 2's host experts not resident, a one-step call not kept ({} \
             admitted at layer 2, {restored} restored in the twin), then {} passes: equal to the \
             twin {}, the call's experts read in again {} (twin {}); error {} {}",
            admitted_l2(&teseen),
            e.values.len(),
            call_twin(&e, &eseen, &te, &teseen),
            call_rereads(&e),
            rereads(&te),
            show(&eseen.err),
            verdict(at_end)
        );
        let none: Vec<usize> = clean_runs.iter().map(|r| rereads(r)).collect();
        let quiet = none.iter().all(|&n| n == 0);
        println!(
            "evicted none: the runs without a fault read in again {none:?} victims (want 0 each) {}",
            verdict(quiet)
        );
        Ok(at_boundary && at_pick && at_end && quiet)
    }

    pub fn run() -> Result<(), GateError> {
        if !set_leak_sink(note_leak) {
            return Err("gate_swap: a leak sink was set before the gate's".into());
        }
        let gpu = Gpu::new()?;
        // SAFETY: the module is this binary's own, loaded once into the
        // card's context before any launch.
        let pm = unsafe { bloomery_gpu::shared_module!(probe_kernels, gpu.context())? };
        let trace = trace();
        println!(
            "gate_swap: layers {LAYERS:?} of {E} experts, top-{K}, card slots {N_L:?}, parts \
             {PART_BYTES:?} B, {PASSES} passes, rule every 2 cap 6 margin 1 min 2 decay 0.9, one \
             spare, delay {DELAY}, {PINNED} pinned"
        );
        let mut a = plain(&gpu, &pm, Faults::default(), DELAY)?;
        drive(&gpu, &mut a, &trace, 0..PASSES, Copies::Prompt, Hold::None)?;
        let mut b = plain(&gpu, &pm, Faults::default(), DELAY)?;
        drive(&gpu, &mut b, &trace, 0..PASSES, Copies::Prompt, Hold::None)?;
        let mut h = plain(&gpu, &pm, Faults::default(), DELAY)?;
        drive(&gpu, &mut h, &trace, 0..PASSES, Copies::Held, Hold::None)?;

        let mut ok = c1(&a, &b, &h);
        ok &= c2(&gpu, &pm, &trace, &a)?;
        ok &= c3(&a, &h);
        ok &= c3_engine(&gpu, &pm, &trace, &a)?;
        ok &= c3_event(&gpu, &pm, &trace)?;
        ok &= c5(&[&a, &b, &h]);
        ok &= c6(&trace, &a)?;
        ok &= clock(&gpu, &pm, &trace)?;
        ok &= pinned(&a);
        ok &= priority(&gpu, &a)?;
        record::residency_pass(&a.reports.last().copied().unwrap_or_default()).print();
        ok &= c7(&gpu, &pm, &trace, &a)?;
        ok &= tier(&gpu, &pm, &trace, &a)?;
        ok &= staging_failure(&gpu, &pm, &trace, &a)?;
        ok &= tally_clause(&gpu, &pm, &trace)?;
        ok &= ahead(&gpu, &pm, &trace, &a)?;
        ok &= keep_clause(&gpu)?;
        ok &= broken(&gpu, &pm, &trace, &a)?;
        ok &= panic_clause(&gpu, &pm, &trace, &a)?;
        ok &= refusal(&gpu, &pm, &trace)?;
        ok &= fault_code()?;
        ok &= stall(&gpu, &pm, &trace)?;
        ok &= placement(&gpu, &pm)?;
        let light = queue(&gpu, &pm, &trace, &a, LIGHT)?;
        ok &= light;
        if light {
            ok &= queue(&gpu, &pm, &trace, &a, INFLATE)?;
        } else {
            println!(
                "queue (heavy): not run — the light arm failed, and the heavy arm would block the \
                 process on the same defect FAIL"
            );
        }
        ok &= dropq(&gpu, &pm, &trace, &a)?;
        ok &= dropq_tier(&gpu, &trace, &a)?;
        ok &= dropq_free(&gpu, &pm, &trace, &a)?;
        ok &= s1_s3(&gpu, &pm, &trace)?;
        ok &= s2(&gpu, &pm, &trace)?;
        ok &= s4(&gpu, &pm, &trace)?;
        ok &= s5(&gpu, &pm)?;
        ok &= evicted(&gpu, &pm, &trace, &a, &[&a, &b, &h])?;
        drop((a, b, h));
        let all: Vec<LeakReason> = leaks()?.iter().map(|l| l.reason).collect();
        let only_stall = all == [LeakReason::Join];
        println!(
            "leaks: every machine dropped, leaks reported {all:?} (want the stall arm's [Join] \
             alone) {}",
            verdict(only_stall)
        );
        ok &= only_stall;

        if ok {
            println!(
                "gate_swap: PASS — the same history gives the same values and flips whatever the \
                 copies' timing, equal to a static replay of its card sets and to an independent \
                 rule over the kept rows; late copies are waited for by the engine stream, and a \
                 copy into a freed slot by the boundary before it; the host and card maps never \
                 serve an id twice or not at all; a reset with flips in flight returns to the \
                 seed; tier entries never move; a staging failure, a panic, a broken machine, a \
                 bad tally, a non-resident victim and a host wait past its deadline are each a \
                 named error, and a dropped machine leaves no copy waiting on the card; an owner \
                 that syncs or frees after its machine, and the host tier itself, drop within \
                 the machine's deadline (dropq), and a plain free against a queued copy is named \
                 (dropq free); pinned seed experts never move; a prompt call's picks equal a static replay of their card sets, land before any read, move the host map at the pick, refuse a victim the host cannot serve and take a floor raised between two picks at the next; a victim whose pages the page cache let go is read in again and counted at a boundary, a pick and a call's end."
            );
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
