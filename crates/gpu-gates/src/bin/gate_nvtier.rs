//! `gate_nvtier` — the NVMe expert tier's RAM arena on the real Qwen3.8
//! file: a load whose room forces the split dial to give the tier an arena
//! against the same load with every host expert resident, through the
//! clauses of the tier's design (nvtier R2):
//!
//! - `bits` (clause 1): both arms pin `BLOOMERY_CARD_BUDGET`'s default, the
//!   context and one explicit residency word (the unset rule's answer on the
//!   paged plan); [`PROMPT`] positions (one prompt call) then [`STEPS`]
//!   greedy steps — every greedy token equal and the prompt's and each
//!   step's logits row bit for bit the all-RAM arm's.
//! - `counters` (clause 2): misses > 0, fills > 0, evictions > 0, the
//!   arena's resident bytes at or under its budget at every step, and no
//!   buffered read on the box's direct-capable mount
//!   ([`NvTierStats::buffered_reads`]); the all-RAM arm attached no tier and
//!   prints no record.
//! - `audit` (clause 3, `--audit`): just before compute, every NVMe-tier
//!   pick's slot is `memcmp`d against a fresh buffered read of the same file
//!   range at the six sampled offsets of `nvtier::audit_offsets`
//!   ([`NvTier::audit_on`]) — a torn or zeroed span refuses by name.
//! - `refuse` (clause 4): the room under the tier's floor
//!   (`HostRoomFloor`), an arena budget under the arena's floor at load
//!   (`ArenaFloor`: one byte, `(k − 1)` slot sets, the floor's own slots
//!   less the lane claim's, each by name), the floor's own budget loading
//!   with exactly its slots and the placement's slot set, and the
//!   direct-read probe of a mount without direct IO, by name.
//! - `residency` (clause 5): the paged plan's unset rule resolves `off`
//!   ([`residency38`]: the plan pages through the tier's arena, where a
//!   promotion copies an expert from its slot when the arena holds it and
//!   through the file mapping, cold after the tier's drops, when it does
//!   not), and both arms run a set
//!   `mid-p0-s1` with an empty churn pool (`ChurnPool::of` leaves the paged
//!   stacks out: the arena serves their victims): flips land on the paged
//!   arm, and its `unresident` and `faulting` are 0 — bug catchers,
//!   structurally unreachable once the seam serves the victims the arena
//!   holds from its books. Its `late` flips are at most [`LATE_MAX`] (the
//!   PIN below) and printed, one line each from the machine's debug ring,
//!   with the five terms that sum to a late copy's issue→boundary time —
//!   the lane queue, the victim's preparation, the staging wait, the source
//!   read, the copy stream — and the ring is held to them: it carries every
//!   late flip the passes counted, or its own overflow count names the rest
//!   (a bug catcher). The flips that land at one boundary are a plan; the
//!   `plan` lines print the [`PLANS_SHOWN`] with the least window / span,
//!   the window the lane had (the first flip's issue to the boundary) over
//!   the span it took (that issue to the last copy staged), and the `plans`
//!   line the least ratio over every plan, the plans under [`PLAN_MARGIN`],
//!   and the steps' wall and sums: the boundaries' wait, the lane's
//!   preparation and staging, the victims the lane prepared, the slots the
//!   tier filled and the landings freed.
//! - `room` (clause 6): the loaded tier's arena, the plan's host need and
//!   the churn pool the residency holds fit the room together, and the
//!   plan's headroom is what the room leaves past the arena and the need.
//! - `footprint` (clause 7, the paged arm, off under `--audit`, whose
//!   buffered reads fill the page cache by design): what the page cache
//!   holds of the tier's drop runs ([`NvTier::drop_region`]), read with
//!   `mincore` over the probe's own open of the file, not the engine's
//!   mapping, three times, each after the tier's dropper has run what its
//!   readers queued ([`NvTier::flush`]). Every page-cache read of the file
//!   in a run is a fault through a mapping (the tier's fills bypass the
//!   cache, `O_DIRECT`), and a fault's readahead stays within `R` pages a
//!   window: the fault path asks for `R`, so the device's larger IO size
//!   never widens it. The read-around opens `R/2` pages before a reader's
//!   first fault; once the reader runs on, each window's marker sits on its
//!   first page, so a reader at page `p` has the cache read through
//!   `p + 2R - 1` — `R/2` before a reader and `2R` past it. Each read splits
//!   a run's pages into its interior and an edge band of `2R` pages at each
//!   of its two ends ([`edge_band`]): a run ends where a byte the host keeps
//!   begins — a held expert, or a tensor beside the stack — and the host's
//!   own reads of those bytes reach that far into it, reads the tier does
//!   not govern. The band's pages are printed, not judged. The interior is
//!   judged: (a) after the tier's own drop of every run before the prompt,
//!   no page held through a second drop — the premise: no other process
//!   maps the file (every gate that maps it holds the V4.1 load lock this
//!   one holds), and this process maps no page of a run once the drop ran
//!   (`MADV_DONTNEED` goes first). The kernel's drop passes over a page it
//!   holds busy at that instant (one it cannot lock, or one holding a
//!   reference past the page cache's own), and the next drop takes it; a
//!   page held through both is the defect: another mapper, or a page the
//!   drop cannot take. The first read's interior pages are printed, each
//!   with its layer, stack row, expert and the expert's class (a card id,
//!   an NVMe-tier id) and its distance to its run's nearer end. (b) After
//!   the prompt call, at most the readahead its readers had in flight when
//!   each layer's drop ran: `2R` pages a
//!   reading thread a paged layer, the readers no more than the host's
//!   CPUs. (c) With the residue dropped again so the two readers stay
//!   apart, after the [`STEPS`] steps, at most the readahead spill of the
//!   flips' copies, which a lane's drop of its own id does not name: a copy
//!   reads an expert's three parts in order through the mapping, each with
//!   `R/2` before it and `2R` past it — `5R/2` a part, `15R/2` a flip. `R`
//!   is the model file's device's `read_ahead_kb` (sysfs) in pages: the box
//!   reads 128 KiB, 32 pages at 4 KiB, so a band of 64 pages and 240 a flip.
//!   The line also prints each read's band pages, and the prompt's drops and
//!   their wall against the prompt's.
//! - `stepunion` (clause 8, a load of its own): the step port's union of
//!   two columns on a paged plan — the plan made for two slots at the same
//!   room and context (the two arms' plan counts one sequence, and a second
//!   on it is refused by name), under that plan's unset residency, `off`
//!   (clause 5's answer, the default the clause judges: the card's split
//!   stays the plan's, so the same tokens stepped alone sum their experts in
//!   the same split; a rule that picks `mid` there is refused by name with
//!   its pick), a prompt in
//!   each slot, then [`STEP_COLS`] passes of both slots' tokens as one
//!   two-column step against the same token sequences stepped alone on the
//!   same load (each slot reset, prompted again and fed the recorded
//!   tokens): both prompts' argmax and every token and logits row bit for
//!   bit the solo steps'. The footprint, read before and after the passes
//!   with the prompts' residue dropped twice first, grows in the interiors by
//!   at most the flips that landed in them × `15R/2` — none land under `off`,
//!   so by none: the step port's unions read the arena, and a two-column
//!   decode leaves no mapping page behind (the old reader keeps every page
//!   it reads). The arena's misses and fills move over the passes: under
//!   `off` no lane fills it, so only the step port's unions can. A second
//!   such load runs the set `mid-p0-s1` (`stepunion mid`): its flips land
//!   under the passes, the lane's victim fills and copies racing the
//!   two-column chain's reads of the arena, and the passes' bits are those of
//!   the same passes on an all-RAM two-slot load under the same word, as
//!   clause 1 holds its pair — a flip moves the card's split with the
//!   schedule, so the solo steps' schedule would sum differently, and the
//!   two loads run one. Flips landed (as many as the all-RAM load's), none
//!   unresident or faulting, the arena served the passes; the arm's `late` is
//!   printed.
//! - `exclusive` (clause 9): the tiers hold an expert once. A promotion of
//!   an expert the arena holds filled reads its slot (the file's bytes at
//!   the arena's skew, the slot pinned against the LRU for the read), and the
//!   landing frees the slot ([`NvTierStats::releases`], the residency
//!   machine's `release_host`). Over the paged arm's steps: every slot read
//!   lends [`EXPERT_PARTS`] parts, the mapping is read only for an expert
//!   the arena held unfilled when its copy opened, the landings' frees and
//!   the copies' opens tally within the flips in flight at the span's two
//!   ends, and the dropper ran no more drops than those mapping reads
//!   queued. Over the run: the slots the books hold — fills less evictions
//!   less releases — are the ones the seam finds filled and the resident
//!   bytes hold, within the fills the lane's threads had in flight at the
//!   read.
//! - `census` (the instrument the next rounds read): each phase's
//!   `tier census` record — the paged arm's prompt call and its steps, the
//!   `stepunion` arms' passes — with its picks a token by where they were
//!   served (the card, the arena, the mapping), the bytes it moved between
//!   the tiers (the arena's fills, the residency machine's promotions and
//!   demotions), the dropper's calls and wall and the pool's dispatch waits,
//!   a token. The bug catchers: every phase's card, arena and file picks sum
//!   to its picks, and its tokens are the ones the gate ran in it (every
//!   layer of a paged plan's map pages ids, so every walk serves each).
//! - `pool` (clause 10): no phase's dispatches waited for another caller's
//!   job. The chain is the pool's one dispatcher; the lane's victim fills
//!   read on the lane's own threads ([`NvTier::ensure_here`]), where a fill
//!   through the pool took the dispatch mutex the chain's host compute holds.
//!
//! PIN(2026-10-10): clause 5's `late` is judged at most [`LATE_MAX`] = 4 of the flips that landed, where the design's rule judges 0 for a plan whose lane span is at most its window / 1.5; derived, not measured, from the 2026-10-09 train log's late-flip print on the A6000 (552 flips landed, 8 late):
//! the lane serves a plan's F = 24 flips one after the other, S = queue / ahead a flip, against the plan's window, so a plan's tail is `max(0, F - floor(window / S))`, 1 / 5 / 2 flips at the three plans that were late, the 8 observed. A promotion that reads its arena slot in place of the cold
//! mapping, and a victim's fill off the pool, cut S, and the chain's dispatch waits leave the windows: the margin window / (F x S) is 1.22 / 1.30 / 1.66 at the three plans, under 1.5 at two of them, so 0 is not derived. The pessimistic corner, a cold read cheaper than derived so that S falls by only a
//! tenth, leaves a tail of 4 at the worst plan and 0 at the rest. The `plan` lines print the margins a run measures: after the change every run read `late` 0, and every plan read [`PLAN_MARGIN`] or more in all but one run (a loaded box, one plan at 1.12), so 0 is not pinned; a landing batch whose runs read every plan at [`PLAN_MARGIN`] or more takes the bound to 0.
//!
//! PIN(2026-10-09): clause 7 judges the runs' interiors, the `2R` bands at
//! their ends printed, and its base judges the interior pages held through
//! a second drop, the first drop's residue printed: a drop takes no page
//! the kernel holds busy at its instant, and the next drop takes it.
//!
//! PIN(2026-10-08): the page-cache design's two named refusals are dropped
//! — a paged tier under `BLOOMERY_HOST_LOCK=1` (the lock walk pins model
//! pages the tier no longer flushes) and an r8 resident copy under a paged
//! tier (`MADV_DONTNEED` cannot zero a copy the arena never advises) —
//! because the arena never advises the model mapping, which the
//! `nvtier::advice_disjoint` helper holds, and the gate's `advice` line checks: the arena's returned pages and the
//! model mapping's are disjoint ranges, and a mutant that madvises a model
//! page turns the check red.
//!
//! PIN(2026-10-09): the arena still never advises the model mapping; the
//! tier now drops the model mapping's pages of the ids it serves after each
//! prompt union and each lane copy (pagedrop: on a small host a long
//! prompt's union filled the page cache and pushed the arena to swap). The
//! `advice` line holds both: the arena's range disjoint from the mapped
//! tensors, and every drop run inside one routed stack of its layer and
//! disjoint from every byte of an id the tier does not serve (the plan's
//! host segment) — a run is built from whole pages inward of those bytes, so
//! the check holds by construction, and a mutant that rounds a run outward
//! at a held id turns it red. The HOST_LOCK note above stays true: the lock
//! pins the host segment's pages, which no drop names.
//!
//! The FAIL-first mutants of each clause are named in the design's §7.6
//! table.

#[path = "shared/gate_card.rs"]
mod gate_card;

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use app::Session;
use bloomery_gpu::arch::qwen3moe::Body38;
use bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for;
use bloomery_gpu::host::census::TierCensus;
use bloomery_gpu::host::lane::{LANE_THREADS, LaneStats};
use bloomery_gpu::host::nvtier::{DropRuns, NvTier, NvTierStats, advice_disjoint};
use bloomery_gpu::host::swap::{LateLog, PassReport, Residency, SwapSource};
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::residency38::{CARD38, Lever38, residency38};
use bloomery_gpu_gates::{Fnv1a64, GateError, checks_failed, data_dir, exit_with, verdict};
use bloomery_levers::{CARD_DONTNEED, HOST_LOCK, HOST_POPULATE, HostCfg, RESIDENCY38_SPARES};
use gguf::Split;
use model::arch::qwen35moe::place::{Experts, PlanInputs, machine_for_experts};
use model::placement::churn::ChurnPool;
use model::placement::host_lock::page_bytes;
use model::placement::workstation::{HostNeed, HostRead};
use model::placement::{ARENA_COLUMNS, ArenaFloor, Machine, PlanLevers};
use runtime::{Out, Target, Want};

const NAME: &str = "gate_nvtier";

/// The model the gates open (`BLOOMERY_REF_MODEL` under the qwen4exp
/// profile).
const MODEL: &str = refset::arch::qwen4exp::MODEL;
/// The stores' positions, `gate_qwen38_residency`'s.
const CTX: usize = 3072;
/// Prompt positions: one prompt call above the stream floor.
const PROMPT: usize = 512;
/// Greedy steps after the prompt.
const STEPS: usize = 96;
/// The paged arm's room, a 32 GB machine's.
const ROOM: u64 = 27 << 30;
/// The `stepunion` arm's prompts, two different short windows of the prose
/// corpus: the clause needs each slot's own routing from its prompt, not a
/// long walk.
const STEP_PROMPT: usize = 32;
/// The `stepunion` arm's two-column passes, and as many solo steps a slot.
/// Priced against the gate's wall: the arm's load is the bulk of its cost,
/// and a pass of two columns costs about two one-column steps, so the passes
/// and the replay add a few of the arm's steps' worth beside it; 24 passes
/// give each slot 24 tokens and the arena its first fills over every paged
/// layer.
const STEP_COLS: usize = 24;
/// The late flips the residency clause prints, one line each; the ring
/// holds the rest and counts them.
const LATE_SHOWN: usize = 32;
/// The most flips of the paged arm's passes whose copy may miss its landing
/// (the PIN in the module doc).
const LATE_MAX: usize = 4;
/// The landing boundaries' plans the residency clause prints, the worst
/// ones: a plan is the flips that land at one boundary.
const PLANS_SHOWN: usize = 8;
/// The parts of an expert the arena's slot holds and a promotion copies: the
/// gate, up and down stacks.
const EXPERT_PARTS: u64 = 3;
/// The margin the design asks of a plan's window over the lane's span for a
/// judged 0 of `late` (printed beside each plan).
const PLAN_MARGIN: f64 = 1.5;

fn main() -> std::process::ExitCode {
    exit_with(NAME, run())
}

/// The first `n` ids of `$BLOOMERY_DATA/qwen4exp/corpus-prose.ids`, the
/// Qwen3.8 tokenizer's prose corpus.
fn prose38(n: usize) -> Result<Vec<u32>, GateError> {
    let path = data_dir().join("qwen4exp").join("corpus-prose.ids");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let ids = text
        .split_whitespace()
        .take(n)
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()?;
    if ids.len() < n {
        return Err(format!("{}: {} ids, the gate reads {n}", path.display(), ids.len()).into());
    }
    Ok(ids)
}

/// What one arm saw: every call's argmax, the prompt's and each step's
/// logits digest, the residency boundaries' counts the paged arm holds to
/// 0, the flips that landed, the late flips the machine's debug ring holds,
/// the arena's counters at the end and the most resident bytes it held at
/// any step, and the arena itself when the load attached one.
struct ArmRun {
    tokens: Vec<u32>,
    digests: Vec<u64>,
    unresident: usize,
    faulting: usize,
    late: usize,
    landed: usize,
    late_flips: LateLog,
    /// The tier census of the prompt call and of the steps, each with the
    /// tokens the gate ran in it, on a load with a tier.
    census: Vec<(&'static str, usize, TierCensus)>,
    resident_max: u64,
    stats: Option<NvTierStats>,
    tier: Option<Arc<NvTier>>,
    seam: Option<Seam>,
    footprint: Option<Footprint>,
    /// Every boundary's report in order, the prompt call's and the steps',
    /// and where the steps' begin.
    passes: Vec<PassReport>,
    steps_from: usize,
    /// Each step's wall, nanoseconds.
    step_ns: Vec<u64>,
    /// The tier's and the lane's counters at the steps' start and end, on a
    /// load with a tier.
    steps: Option<Span>,
}

/// The counters of a span: the tier's and the lane's at its start and at
/// its end.
struct Span {
    tier: [NvTierStats; 2],
    lane: [LaneStats; 2],
}

impl Span {
    /// What the tier's counter `of` moved over the span.
    fn tier(&self, of: fn(&NvTierStats) -> u64) -> u64 {
        of(&self.tier[1]) - of(&self.tier[0])
    }
}

/// Resident pages over the tier's drop runs (`footprint`): those in a run's
/// interior and those in the edge band at either of its ends, and the runs'
/// pages.
#[derive(Clone, Copy, Default)]
struct Pages {
    interior: u64,
    edge: u64,
    total: u64,
}

/// The first base read's interior pages `footprint` checks one by one
/// against a second drop; a page past them counts held.
const HELD_CHECKED: usize = 1 << 16;

/// The first base read's interior pages `footprint` prints, one line each.
const PAGES_SHOWN: usize = 32;

/// One resident page of a drop run's interior: its layer, the plan's stack
/// row, the shard, the page's file offset, and its distance in pages to its
/// run's nearer end.
#[derive(Clone, Copy)]
struct Page {
    layer: usize,
    row: usize,
    shard: usize,
    at: u64,
    from_end: u64,
}

/// The interior pages a read found: the first [`HELD_CHECKED`] of them, and
/// how many came past those.
#[derive(Default)]
struct Interior {
    pages: Vec<Page>,
    past: u64,
}

impl Interior {
    fn note(&mut self, page: Page) {
        if self.pages.len() < HELD_CHECKED {
            self.pages.push(page);
        } else {
            self.past += 1;
        }
    }
}

/// The paged arm's footprint: after the tier's own drop before the prompt,
/// after the prompt and after the steps, over its paged layers, read with
/// an edge band of `band` pages; the base read's interior pages, which of
/// them a second drop of the tier's left (`held`) and which the prompt's
/// read found (`at_prompt`); each layer's card experts, the plan's first
/// ids of its stacks (`card`); and the prompt's drops (calls, wall)
/// against the prompt's wall.
struct Footprint {
    band: u64,
    base: Pages,
    first: Interior,
    held: Vec<bool>,
    at_prompt: Vec<bool>,
    card: Vec<u64>,
    prompt: Pages,
    steps: Pages,
    layers: usize,
    prompt_drops: u64,
    prompt_drop_ns: u64,
    prompt_ns: u64,
}

/// The residency seam's answers over the ids the arena serves, against the arena's own books,
/// and the tier's counters just before and just after the read: the lane's threads keep filling
/// while it runs.
struct Seam {
    checked: usize,
    filled: usize,
    wrong: usize,
    counters: [NvTierStats; 2],
}

/// The plan of `file` on `machine` with `room` given, or the machine's own
/// reading when `None`.
fn inputs_of(file: &Split, room: Option<u64>) -> Result<PlanInputs, GateError> {
    let mut inputs = PlanInputs::describe(file)?;
    if let Some(room) = room {
        inputs.room = (room, HostRead::Given);
    }
    Ok(inputs)
}

/// One arm: the model loaded under `residency` with `room` given (or the
/// machine's reading), its [`PROMPT`] ids as one prompt call and [`STEPS`]
/// greedy steps, the residency boundaries' reports summed; on a load with a
/// tier and a `band` (none under `--audit`), the footprint read through
/// `probe`, the gate's own open of the file, with that edge band.
#[allow(
    clippy::too_many_arguments,
    reason = "one arm's load, its inputs and its instrument, each named"
)]
fn arm(
    path: &Path,
    machine: &Machine,
    room: Option<u64>,
    residency: Residency,
    host: HostCfg,
    ids: &[u32],
    audit: bool,
    probe: &Split,
    band: Option<u64>,
) -> Result<ArmRun, GateError> {
    let file = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let inputs = inputs_of(&file, room)?;
    let ub = ubatch_for(CTX)?;
    let plan = inputs.plan_with(machine, CTX as u64, &PlanLevers::default(), Experts::Card)?;
    let card = plan.n_l.clone();
    if (plan.host.nvme_expert_bytes > 0) != room.is_some() {
        return Err(format!(
            "the {} arm's plan pages {} B on the NVMe tier: {}",
            if room.is_some() { "paged" } else { "all-RAM" },
            plan.host.nvme_expert_bytes,
            if room.is_some() {
                "the dial gave the room no tier — check the floor and the room"
            } else {
                "the host's own room does not hold the host leg (BLOOMERY_HOST_ROOM set?)"
            }
        )
        .into());
    }
    let model = Body38::open_placed_residency(file, &plan, &inputs, 0, host, ub, residency, 1)?;
    let mut s = Session::from_model(model, CTX as u32);
    s.model_mut().body_parts(NAME)?.2.log_residency(0);
    let tier = s.model().body(NAME)?.nvme_tier().cloned();
    if audit {
        tier.as_ref()
            .ok_or("--audit on a load that built no arena")?
            .audit_on()?;
    }
    // The load's boundaries and their late flips go together, so the ring
    // the residency clause holds to the passes' `late` covers the same
    // passes.
    take_passes(&mut s)?;
    s.model_mut().body_parts(NAME)?.2.take_late_flips();
    // `footprint`: the tier's drop runs, its own definition; the page cache
    // emptied of them by the tier's own drop first, so a page an earlier run
    // left warm is not this one's.
    let foot = match (&tier, band) {
        (Some(t), Some(band)) if !audit => {
            let regions = (0..plan.model.layers)
                .filter(|&l| t.covers(l))
                .map(|l| Ok((l, t.drop_region(l)?)))
                .collect::<Result<Vec<_>, GateError>>()?;
            drop_all(t, &regions)?;
            Some((Arc::clone(t), regions, band))
        }
        _ => None,
    };
    // The base read's interior pages, then the tier's drop once more: a
    // page it leaves too is held, one it takes was busy at the first drop.
    let mut first = Interior::default();
    let (base, held) = match &foot {
        Some((t, regions, band)) => {
            let base = resident(probe, regions, *band, Some(&mut first))?;
            drop_all(t, regions)?;
            (base, still_resident(probe, &first.pages)?)
        }
        None => (Pages::default(), Vec::new()),
    };
    let drops0 = tier.as_ref().map(|t| t.stats());
    let mut run = ArmRun {
        tokens: Vec::with_capacity(STEPS + 1),
        digests: Vec::with_capacity(STEPS + 1),
        unresident: 0,
        faulting: 0,
        late: 0,
        landed: 0,
        late_flips: LateLog::default(),
        census: Vec::new(),
        resident_max: 0,
        stats: None,
        tier: tier.clone(),
        seam: None,
        footprint: None,
        passes: Vec::new(),
        steps_from: 0,
        step_ns: Vec::with_capacity(STEPS),
        steps: None,
    };
    let note = |s: &mut Session<Body38>, run: &mut ArmRun| -> Result<(), GateError> {
        for (_, r) in take_passes(s)? {
            run.unresident += r.unresident.count();
            run.faulting += r.faulting.count();
            run.late += r.late;
            run.landed += r.landed;
            run.passes.push(r);
        }
        if let Some(t) = &tier {
            run.resident_max = run.resident_max.max(t.stats().resident_bytes);
        }
        Ok(())
    };
    let c0 = census_of(&s)?;
    let t0 = Instant::now();
    let out = s.prompt(ids, Want::Logits)?;
    let prompt_ns = t0.elapsed().as_nanos() as u64;
    let Out::Logits { argmax, row } = out else {
        return Err("a prompt asked for its logits answered an argmax".into());
    };
    run.digests.push(Fnv1a64::default().f32s(row).value());
    let mut next = argmax;
    run.tokens.push(next);
    note(&mut s, &mut run)?;
    // The readers queue their drops for the tier's dropper: every read
    // below waits for the queue first.
    if let Some((t, ..)) = &foot {
        t.flush()?;
    }
    let drops1 = tier.as_ref().map(|t| t.stats());
    let c1 = census_of(&s)?;
    // The prompt's residue dropped through the tier, so the steps' read
    // names only what the steps' own readers left.
    let (prompt_pages, at_prompt) = match &foot {
        Some((t, regions, band)) => {
            let p = resident(probe, regions, *band, None)?;
            let at = still_resident(probe, &first.pages)?;
            drop_all(t, regions)?;
            (p, at)
        }
        None => (Pages::default(), Vec::new()),
    };
    // The steps' census starts past the footprint's own drops.
    let c1b = census_of(&s)?;
    run.steps_from = run.passes.len();
    let span0 = match (&tier, lane_of(&s)?) {
        (Some(t), Some(l)) => Some((t.stats(), l)),
        _ => None,
    };
    for _ in 0..STEPS {
        let t = Instant::now();
        let out = s.step(next, Want::Logits)?;
        run.step_ns.push(t.elapsed().as_nanos() as u64);
        if let Out::Logits { row, .. } = out {
            run.digests.push(Fnv1a64::default().f32s(row).value());
        }
        next = out.argmax();
        run.tokens.push(next);
        note(&mut s, &mut run)?;
    }
    if let Some((t, ..)) = &foot {
        t.flush()?;
    }
    run.stats = tier.as_ref().map(|t| t.stats());
    if let (Some((t0, l0)), Some(t1), Some(l1)) = (span0, run.stats, lane_of(&s)?) {
        run.steps = Some(Span {
            tier: [t0, t1],
            lane: [l0, l1],
        });
    }
    if let (Some(c0), Some(c1), Some(c1b), Some(c2)) = (c0, c1, c1b, census_of(&s)?) {
        run.census = vec![
            ("prompt", ids.len(), c1.since(&c0)),
            ("steps", STEPS, c2.since(&c1b)),
        ];
    }
    if let (Some((_, regions, band)), Some(d0), Some(d1)) = (&foot, drops0, drops1) {
        run.footprint = Some(Footprint {
            band: *band,
            base,
            first,
            held,
            at_prompt,
            card,
            prompt: prompt_pages,
            steps: resident(probe, regions, *band, None)?,
            layers: regions.len(),
            prompt_drops: d1.drops - d0.drops,
            prompt_drop_ns: d1.drop_ns - d0.drop_ns,
            prompt_ns,
        });
    }
    run.seam = seam_of(&s, plan.model.layers, plan.model.experts as u32)?;
    run.late_flips = s.model_mut().body_parts(NAME)?.2.take_late_flips();
    if record::nvtier_of(tier.as_deref()).is_some() != room.is_some() {
        return Err(
            "the nvtier record is printed on a load with no tier or missing on one with it".into(),
        );
    }
    Ok(run)
}

/// The seam's answer `host_resident` gives for every id the arena serves, held to the arena's
/// books (`slot_filled`): the file mapping's page cache residency never answers for it, however
/// warm the mapping is. `None` on a load with no arena or no residency machine.
fn seam_of(s: &Session<Body38>, layers: usize, experts: u32) -> Result<Option<Seam>, GateError> {
    let b = s.model().body(NAME)?;
    let (Some(tier), Some(source)) = (b.nvme_tier(), b.residency_source()) else {
        return Ok(None);
    };
    let mut seam = Seam {
        checked: 0,
        filled: 0,
        wrong: 0,
        counters: [tier.stats(); 2],
    };
    for l in 0..layers {
        for id in 0..experts {
            if !tier.serves(l, id) {
                continue;
            }
            let held = tier.slot_filled(l, id);
            seam.checked += 1;
            seam.filled += usize::from(held);
            seam.wrong += usize::from(source.host_resident(l, id)? != held);
        }
    }
    seam.counters[1] = tier.stats();
    Ok(Some(seam))
}

/// What the residency machine's lane did since the load
/// ([`bloomery_gpu::host::swap::SwapMachine::lane_stats`]): `None` on a load
/// with no machine.
fn lane_of(s: &Session<Body38>) -> Result<Option<LaneStats>, GateError> {
    Ok(s.model()
        .body(NAME)?
        .hybrid()
        .swap()
        .map(|m| m.lane_stats()))
}

/// The load's tier census since the load (`HostTier::census`): `None` on a
/// load with no tier.
fn census_of(s: &Session<Body38>) -> Result<Option<TierCensus>, GateError> {
    let b = s.model().body(NAME)?;
    Ok(b.hybrid().census(b.nvme_tier().map(|t| &**t)))
}

/// The `tier census` record of span `c`, phase `phase`.
fn census_record(phase: &str, c: &TierCensus) -> Record {
    c.fields().iter().fold(
        Record::new(&record::TIER_CENSUS).w("phase", phase),
        |r, &(k, v)| r.u(k, v),
    )
}

/// The tier's own drop of every run of `regions` ([`NvTier::drop_layer`]).
fn drop_all(tier: &NvTier, regions: &[(usize, DropRuns)]) -> Result<(), GateError> {
    for &(layer, _) in regions {
        tier.drop_layer(layer)?;
    }
    Ok(())
}

/// The set `mid-p0-s1` the paged arms and the two-slot `mid` arms run.
const MID: Residency = Residency::Mid {
    pinned: 0,
    spares: RESIDENCY38_SPARES,
};

/// A prompt in each of the two slots of `s`: slot 0's the first window of
/// the corpus, slot 1's the next — a routing of its own; each slot's argmax.
fn prompt_slots(s: &mut Session<Body38>, ids: &[u32]) -> Result<[u32; 2], GateError> {
    let windows = [&ids[..STEP_PROMPT], &ids[STEP_PROMPT..2 * STEP_PROMPT]];
    let mut out = [0u32; 2];
    for (slot, p) in windows.iter().enumerate() {
        s.select_slot(slot)?;
        out[slot] = s.prompt(p, Want::Argmax)?.argmax();
    }
    Ok(out)
}

/// [`STEP_COLS`] passes of both slots' tokens as one two-column step
/// ([`Session::step_slots`], the step port's union), the first pass fed
/// `prompts` and each next the tokens the pass before answered: every token
/// and logits-row digest (slot-major within a pass), and the tokens each
/// pass was fed.
#[allow(
    clippy::type_complexity,
    reason = "one pass's three records, each named"
)]
fn two_columns(
    s: &mut Session<Body38>,
    prompts: [u32; 2],
) -> Result<(Vec<u32>, Vec<u64>, Vec<[u32; 2]>), GateError> {
    let mut tokens = Vec::with_capacity(2 * STEP_COLS);
    let mut digests = Vec::with_capacity(2 * STEP_COLS);
    let mut fed: Vec<[u32; 2]> = Vec::with_capacity(STEP_COLS);
    let mut next = prompts;
    for _ in 0..STEP_COLS {
        fed.push(next);
        let out = s.step_slots(&[(0, &next[..1]), (1, &next[1..])])?;
        for row in s.model().slots_logits()? {
            digests.push(Fnv1a64::default().f32s(&row).value());
        }
        let &[a, b] = out.ids.as_slice() else {
            return Err(format!(
                "a two-column pass answered {} tokens, not one a slot",
                out.ids.len()
            )
            .into());
        };
        tokens.extend([a, b]);
        next = [a, b];
    }
    Ok((tokens, digests, fed))
}

/// The `stepunion` arm's run: each slot's prompt argmax and the two-column
/// passes' tokens and logits-row digests (slot-major within a pass, a pass's
/// rows in its row order) against the same of the token sequences stepped
/// alone; the flips that landed during the passes; the arena's misses and
/// fills over them; and the footprint read just before and just after them,
/// over the arm's paged layers.
struct StepRun {
    prompts: [u32; 2],
    tokens: Vec<u32>,
    digests: Vec<u64>,
    solo_prompts: [u32; 2],
    solo_tokens: Vec<u32>,
    solo_digests: Vec<u64>,
    flips: usize,
    misses: u64,
    fills: u64,
    /// The tier census of the passes.
    census: TierCensus,
    before: Pages,
    after: Pages,
    layers: usize,
}

/// The `stepunion` clause's own arm: a load of its own on the paged plan
/// made for **two** slots ([`PlanInputs::plan_with_slots`] — the two arms'
/// plan counts one, and a second sequence on it is refused by name) at the
/// same room and context, under the unset rule's residency on that plan
/// (`off`, clause 5's answer: the card's split stays the plan's, so a slot
/// stepped alone sums its experts in the split its two-column pass did),
/// serving two slots: a prompt in each, then [`STEP_COLS`] passes of both
/// slots' tokens as one two-column step ([`Session::step_slots`], the step
/// port's union), then each slot reset, prompted again and fed the recorded
/// tokens one column a step on the same load. Refused by name: a two-slot
/// plan that pages nothing or builds no arena (with the plan's numbers), an
/// arena whose carve falls under the two-column floor (`ArenaFloor::of` at
/// two columns: the arm's call is wider than the served one the dial sizes
/// for), and an unset rule that picks `mid` on it (with its pick: the clause
/// judges the `off` default, never a forced one).
/// The footprint is read before and after the passes, the prompts' residue
/// dropped twice first (the second drop takes the pages the first found
/// busy); `band` `None` (under `--audit`) reads none.
fn stepunion_arm(
    path: &Path,
    machine: &Machine,
    host: HostCfg,
    ids: &[u32],
    probe: &Split,
    band: Option<u64>,
) -> Result<StepRun, GateError> {
    let file = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let inputs = inputs_of(&file, Some(ROOM))?;
    let plan = inputs.plan_with_slots(
        machine,
        CTX as u64,
        &PlanLevers::default(),
        Experts::Card,
        2,
    )?;
    if plan.host.nvme_expert_bytes == 0 || plan.host.nvme_arena_bytes == 0 {
        return Err(format!(
            "the stepunion arm's two-slot plan at the room {ROOM} B pages {} B on the NVMe tier \
             with an arena of {} B: the clause needs a paged plan with an arena",
            plan.host.nvme_expert_bytes, plan.host.nvme_arena_bytes
        )
        .into());
    }
    // The residency the unset rule picks on this plan, its `residency
    // unset` record (the pick and its why) printed: the clause judges that
    // default, `off`, and refuses by name a plan whose rule picks another.
    let residency = residency38(&plan, Lever38::Unset(None), Record::print)?;
    if let Residency::Mid { pinned, spares } = residency {
        return Err(format!(
            "the stepunion arm's plan picks mid-p{pinned}-s{spares} under the unset rule; the \
             clause judges the off default"
        )
        .into());
    }
    println!(
        "stepunion plan: two slots at the room {ROOM} B page {} B on the NVMe tier, arena {} B, \
         residency off (the unset rule's pick, its record above)",
        plan.host.nvme_expert_bytes, plan.host.nvme_arena_bytes
    );
    let ub = ubatch_for(CTX)?;
    let model = Body38::open_placed_residency(file, &plan, &inputs, 0, host, ub, residency, 2)?;
    let mut s = Session::from_model(model, CTX as u32);
    s.add_slots(2)?;
    let tier = s
        .model()
        .body(NAME)?
        .nvme_tier()
        .cloned()
        .ok_or("the stepunion arm's two-slot plan built no arena")?;
    // The arena serves the arm's widest call, two columns a step: the
    // placement's floor at two columns against the slots the tier carved.
    let wide = ArenaFloor::of(&plan, 2)?;
    let carved = tier.stats().slots_per_layer;
    if carved < wide.slots {
        return Err(format!(
            "the stepunion arm's arena carves {carved} slots a paged layer, under its two-column \
             floor of {wide}"
        )
        .into());
    }
    println!("stepunion floor: {carved} slots a paged layer serve two columns ({wide})");
    let regions = (0..plan.model.layers)
        .filter(|&l| tier.covers(l))
        .map(|l| Ok((l, tier.drop_region(l)?)))
        .collect::<Result<Vec<_>, GateError>>()?;
    let prompts = prompt_slots(&mut s, ids)?;
    // The prompts' readers done and their residue dropped twice, so the
    // read after the passes names only the passes' own readers.
    tier.flush()?;
    drop_all(&tier, &regions)?;
    drop_all(&tier, &regions)?;
    take_passes(&mut s)?;
    let before = match band {
        Some(b) => resident(probe, &regions, b, None)?,
        None => Pages::default(),
    };
    let t0 = tier.stats();
    let c0 = census_of(&s)?.ok_or("the stepunion arm's load took no census")?;
    let (tokens, digests, fed) = two_columns(&mut s, prompts)?;
    let flips = take_passes(&mut s)?.iter().map(|(_, r)| r.landed).sum();
    let t1 = tier.stats();
    tier.flush()?;
    let census = census_of(&s)?
        .ok_or("the stepunion arm's load took no census")?
        .since(&c0);
    let after = match band {
        Some(b) => resident(probe, &regions, b, None)?,
        None => Pages::default(),
    };
    // The same token sequences stepped alone on the same load.
    for slot in 0..2 {
        s.select_slot(slot)?;
        s.reset()?;
    }
    let solo_prompts = prompt_slots(&mut s, ids)?;
    let mut solo_tokens = Vec::with_capacity(2 * STEP_COLS);
    let mut solo_digests = Vec::with_capacity(2 * STEP_COLS);
    for pass in &fed {
        for (slot, &id) in pass.iter().enumerate() {
            s.select_slot(slot)?;
            let Out::Logits { argmax, row } = s.step(id, Want::Logits)? else {
                return Err("a stepunion solo step asked for logits answered an argmax".into());
            };
            solo_digests.push(Fnv1a64::default().f32s(row).value());
            solo_tokens.push(argmax);
        }
    }
    Ok(StepRun {
        prompts,
        tokens,
        digests,
        solo_prompts,
        solo_tokens,
        solo_digests,
        flips,
        misses: t1.misses - t0.misses,
        fills: t1.fills - t0.fills,
        census,
        before,
        after,
        layers: regions.len(),
    })
}

/// The `stepunion mid` arm's run: each slot's prompt argmax and the two-column
/// passes' tokens and logits-row digests; the flips that landed during the
/// passes, how many were late and the experts found unresident or faulting
/// over them; and, on a load with a tier, the arena's misses and fills over
/// the passes and their tier census.
struct MidRun {
    prompts: [u32; 2],
    tokens: Vec<u32>,
    digests: Vec<u64>,
    flips: usize,
    late: usize,
    unresident: usize,
    faulting: usize,
    served: Option<(u64, u64, TierCensus)>,
}

/// The `stepunion mid` clause's arm: a load made for **two** slots under the
/// set `mid-p0-s1`, on the paged plan at [`ROOM`] when `room` is given and
/// on the all-RAM plan when it is not, a prompt in each slot and then
/// [`STEP_COLS`] two-column passes: flips land under them, and on the paged
/// load the lane's victim fills and copies race the two-column chain's reads
/// of the arena. Both loads run one schedule, so the paged one's bits are
/// held to the all-RAM one's, as clause 1 holds the one-slot pair. Refused by
/// name: a plan whose paging does not match `room`, and on the paged load an
/// arena under the two-column floor.
fn slots_mid_arm(
    path: &Path,
    machine: &Machine,
    host: HostCfg,
    ids: &[u32],
    room: Option<u64>,
) -> Result<MidRun, GateError> {
    let file = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let inputs = inputs_of(&file, room)?;
    let plan = inputs.plan_with_slots(
        machine,
        CTX as u64,
        &PlanLevers::default(),
        Experts::Card,
        2,
    )?;
    let kind = if room.is_some() { "paged" } else { "all-RAM" };
    if (plan.host.nvme_expert_bytes > 0) != room.is_some()
        || (room.is_some() && plan.host.nvme_arena_bytes == 0)
    {
        return Err(format!(
            "the stepunion mid {kind} arm's two-slot plan pages {} B on the NVMe tier with an \
             arena of {} B",
            plan.host.nvme_expert_bytes, plan.host.nvme_arena_bytes
        )
        .into());
    }
    let residency = residency38(&plan, Lever38::Set(MID, "mid-p0-s1"), Record::print)?;
    println!(
        "stepunion mid plan: two slots, the {kind} plan: {} B on the NVMe tier, arena {} B, \
         residency the set mid-p0-s1",
        plan.host.nvme_expert_bytes, plan.host.nvme_arena_bytes
    );
    let ub = ubatch_for(CTX)?;
    let model = Body38::open_placed_residency(file, &plan, &inputs, 0, host, ub, residency, 2)?;
    let mut s = Session::from_model(model, CTX as u32);
    s.add_slots(2)?;
    s.model_mut().body_parts(NAME)?.2.log_residency(0);
    let tier = s.model().body(NAME)?.nvme_tier().cloned();
    if let Some(t) = &tier {
        let wide = ArenaFloor::of(&plan, 2)?;
        let carved = t.stats().slots_per_layer;
        if carved < wide.slots {
            return Err(format!(
                "the stepunion mid arm's arena carves {carved} slots a paged layer, under its \
                 two-column floor of {wide}"
            )
            .into());
        }
    }
    let prompts = prompt_slots(&mut s, ids)?;
    take_passes(&mut s)?;
    let t0 = tier.as_ref().map(|t| t.stats());
    let c0 = census_of(&s)?;
    let (tokens, digests, _) = two_columns(&mut s, prompts)?;
    let passes = take_passes(&mut s)?;
    let served = match (&tier, t0, c0) {
        (Some(t), Some(t0), Some(c0)) => {
            let t1 = t.stats();
            let census = census_of(&s)?
                .ok_or("the stepunion mid arm's load took no census")?
                .since(&c0);
            Some((t1.misses - t0.misses, t1.fills - t0.fills, census))
        }
        _ => None,
    };
    Ok(MidRun {
        prompts,
        tokens,
        digests,
        flips: passes.iter().map(|(_, r)| r.landed).sum(),
        late: passes.iter().map(|(_, r)| r.late).sum(),
        unresident: passes.iter().map(|(_, r)| r.unresident.count()).sum(),
        faulting: passes.iter().map(|(_, r)| r.faulting.count()).sum(),
        served,
    })
}

/// The page cache's residency over the runs of `regions`, read with
/// `mincore` through `probe`'s mapping of each shard (the gate's own open of
/// the file: a file page's residency is the page cache's, whichever mapping
/// asks): the resident pages in each run's interior and in the `band` pages
/// at either of its ends ([`edge_band`]), and the runs' pages; each interior
/// page noted in `list` when one is given.
fn resident(
    probe: &Split,
    regions: &[(usize, DropRuns)],
    band: u64,
    mut list: Option<&mut Interior>,
) -> Result<Pages, GateError> {
    let page = page_bytes().map_err(|e| format!("the page size: {e}"))?;
    let band = usize::try_from(band)?;
    let mut pages = Pages::default();
    let mut vec = Vec::new();
    for (layer, rows) in regions {
        for (row, (shard, runs)) in rows.iter().enumerate() {
            let map = probe
                .shard(*shard)
                .ok_or_else(|| format!("layer {layer}: shard {shard} is not in the probe"))?
                .mapping();
            for run in runs {
                let span = usize::try_from(run.start)
                    .ok()
                    .zip(usize::try_from(run.end).ok())
                    .and_then(|(a, b)| map.get(a..b))
                    .ok_or_else(|| {
                        format!(
                            "layer {layer} shard {shard}: the run {run:?} is past the \
                             mapping's {} B",
                            map.len()
                        )
                    })?;
                mincore_into(span, &mut vec, page as usize).map_err(|e| {
                    format!("mincore over layer {layer} shard {shard} bytes {run:?}: {e}")
                })?;
                let n = vec.len();
                for (i, _) in vec.iter().enumerate().filter(|&(_, &v)| v & 1 != 0) {
                    if i < band || i + band >= n {
                        pages.edge += 1;
                    } else {
                        pages.interior += 1;
                        if let Some(list) = list.as_deref_mut() {
                            list.note(Page {
                                layer: *layer,
                                row,
                                shard: *shard,
                                at: run.start + i as u64 * page,
                                from_end: i.min(n - 1 - i) as u64,
                            });
                        }
                    }
                }
                pages.total += n as u64;
            }
        }
    }
    Ok(pages)
}

/// Whether each of `pages` is in the page cache now: `mincore` over its one
/// page through `probe`'s mapping of its shard.
fn still_resident(probe: &Split, pages: &[Page]) -> Result<Vec<bool>, GateError> {
    let page = usize::try_from(page_bytes().map_err(|e| format!("the page size: {e}"))?)?;
    let mut vec = Vec::with_capacity(1);
    let mut out = Vec::with_capacity(pages.len());
    for p in pages {
        let map = probe
            .shard(p.shard)
            .ok_or_else(|| format!("layer {}: shard {} is not in the probe", p.layer, p.shard))?
            .mapping();
        let at = usize::try_from(p.at)?;
        let span = map.get(at..at + page).ok_or_else(|| {
            format!(
                "layer {} shard {}: the page at {} is past the mapping's {} B",
                p.layer,
                p.shard,
                p.at,
                map.len()
            )
        })?;
        mincore_into(span, &mut vec, page).map_err(|e| {
            format!(
                "mincore over layer {} shard {} at {}: {e}",
                p.layer, p.shard, p.at
            )
        })?;
        out.push(vec[0] & 1 != 0);
    }
    Ok(out)
}

/// The page cache's residency of `span`'s pages into `vec`, one byte a
/// page, bit 0 set for a resident one: `mincore`, which reads the page
/// tables and the page cache and faults nothing in. `span` starts on a page
/// boundary inside the probe's live read-only mapping of a shard.
fn mincore_into(span: &[u8], vec: &mut Vec<u8>, page: usize) -> std::io::Result<()> {
    vec.clear();
    vec.resize(span.len().div_ceil(page), 0u8);
    // SAFETY: `span` lies inside the probe's live read-only mapping of a
    // shard and starts on a page boundary (the mapping starts on one, and a
    // drop run and each of its pages are whole pages of file offsets); `vec`
    // holds one byte for each of its pages. mincore reads the page tables and
    // the page cache and writes only `vec`.
    let rc = unsafe {
        libc::mincore(
            span.as_ptr().cast_mut().cast(),
            span.len(),
            vec.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// The expert whose bytes hold `page`: its offset in the routed stack of
/// its layer that holds it (a `blk.<l>.` tensor whose name holds `_exps`)
/// in the probe's own tensor table, the stack cut into `experts` equal
/// parts. Refused by name: an offset in no routed stack of the layer.
fn expert_at(probe: &Split, page: &Page, experts: u32) -> Result<u32, GateError> {
    let g = probe.shard(page.shard).ok_or_else(|| {
        format!(
            "layer {}: shard {} is not in the probe",
            page.layer, page.shard
        )
    })?;
    let base = g.mapping().as_ptr() as usize;
    let prefix = format!("blk.{}.", page.layer);
    for t in g.iter_tensors() {
        let Ok(d) = g.data(t) else { continue };
        let t0 = (d.as_ptr() as usize - base) as u64;
        let t1 = t0 + d.len() as u64;
        if t.name.starts_with(&prefix) && t.name.contains("_exps") && (t0..t1).contains(&page.at) {
            let per = (t1 - t0) / u64::from(experts);
            return Ok(u32::try_from((page.at - t0) / per)?);
        }
    }
    Err(format!(
        "layer {} shard {}: the page at {} lies in no routed stack of the layer",
        page.layer, page.shard, page.at
    )
    .into())
}

/// The readahead window of the device that holds `path`, in pages: its
/// queue's `read_ahead_kb` in sysfs (a partition's is its disk's), refused
/// by name when the device has none.
fn readahead_pages(path: &Path) -> Result<u64, GateError> {
    use std::os::unix::fs::MetadataExt;
    let dev = std::fs::metadata(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .dev();
    // glibc's dev_t encoding (gnu_dev_major, gnu_dev_minor).
    let major = ((dev >> 8) & 0xfff) | ((dev >> 32) & !0xfff);
    let minor = (dev & 0xff) | ((dev >> 12) & !0xff);
    let at = Path::new("/sys/dev/block").join(format!("{major}:{minor}"));
    let kb = [
        at.join("queue/read_ahead_kb"),
        at.join("../queue/read_ahead_kb"),
    ]
    .iter()
    .find_map(|f| std::fs::read_to_string(f).ok())
    .ok_or_else(|| {
        format!(
            "{}: no read_ahead_kb for its device {major}:{minor}",
            path.display()
        )
    })?;
    let kb: u64 = kb.trim().parse().map_err(|e| {
        format!(
            "the read_ahead_kb {:?} of device {major}:{minor}: {e}",
            kb.trim()
        )
    })?;
    let page = page_bytes().map_err(|e| format!("the page size: {e}"))?;
    Ok(kb * 1024 / page)
}

/// The edge band at each end of a drop run, in pages, for a readahead
/// window of `ra` pages: the kernel's reach from a reader of the kept bytes
/// a run ends at into the run, `2R` pages (the module doc's clause 7). The
/// footprint prints its pages and judges the interior.
fn edge_band(ra: u64) -> u64 {
    2 * ra
}

/// The drop runs against the probe's own tensor table (`advice` (b) and
/// (c)): every run of every paged layer lies inside one routed stack of its
/// layer (a `blk.<l>.` tensor whose name holds `_exps`), so it names no byte
/// of another tensor, and meets no byte of an id the tier does not serve
/// (the plan's host segment: an id's bytes are its stack's `experts`-th part,
/// in id order). The runs checked, and the first breach by name.
fn drop_advice(
    probe: &Split,
    tier: &NvTier,
    layers: usize,
    experts: u32,
) -> Result<(usize, Option<String>), GateError> {
    let mut checked = 0;
    for l in (0..layers).filter(|&l| tier.covers(l)) {
        for (shard, runs) in tier.drop_region(l)? {
            let g = probe
                .shard(shard)
                .ok_or_else(|| format!("layer {l}: shard {shard} is not in the probe"))?;
            let base = g.mapping().as_ptr() as usize;
            let spans: Vec<(&str, u64, u64)> = g
                .iter_tensors()
                .filter_map(|t| {
                    let d = g.data(t).ok()?;
                    let at = (d.as_ptr() as usize - base) as u64;
                    Some((t.name.as_str(), at, at + d.len() as u64))
                })
                .collect();
            for run in &runs {
                checked += 1;
                let Some(&(name, t0, t1)) =
                    spans.iter().find(|s| s.1 <= run.start && run.start < s.2)
                else {
                    return Ok((
                        checked,
                        Some(format!(
                            "layer {l} shard {shard}: the run {run:?} starts in no tensor"
                        )),
                    ));
                };
                if run.end > t1
                    || !name.starts_with(&format!("blk.{l}."))
                    || !name.contains("_exps")
                {
                    return Ok((
                        checked,
                        Some(format!(
                            "layer {l}: the run {run:?} is not inside one routed stack of the layer \
                             ({name}, bytes {t0}..{t1})"
                        )),
                    ));
                }
                if (t1 - t0) % u64::from(experts) != 0 {
                    return Err(
                        format!("{name}: {} B do not cut into {experts} experts", t1 - t0).into(),
                    );
                }
                let per = (t1 - t0) / u64::from(experts);
                for id in (0..experts).filter(|&id| !tier.serves(l, id)) {
                    let (a, b) = (t0 + u64::from(id) * per, t0 + u64::from(id + 1) * per);
                    if a < run.end && run.start < b {
                        return Ok((
                            checked,
                            Some(format!(
                                "layer {l}: a drop run names a held byte — {run:?} meets expert \
                             {id}'s bytes {a}..{b} of {name}"
                            )),
                        ));
                    }
                }
            }
        }
    }
    Ok((checked, None))
}

/// The boundaries' reports since the last take.
fn take_passes(
    s: &mut Session<Body38>,
) -> Result<Vec<(bloomery_gpu::host::PassKind, PassReport)>, GateError> {
    let (_, _, b) = s.model_mut().body_parts(NAME)?;
    Ok(b.take_residency_passes())
}

/// The flips that land together at one boundary — a plan, issued at one
/// boundary — with the window the lane had from the first one's issue to the
/// boundary and the span it took from that issue to the last copy staged.
struct Plan {
    boundary: u64,
    landed: usize,
    late: usize,
    window_us: u64,
    span_us: u64,
}

impl Plan {
    /// How many spans the window holds: 1 is the lane just keeping up.
    fn margin(&self) -> f64 {
        self.window_us as f64 / self.span_us.max(1) as f64
    }
}

/// The plans of `passes`: each boundary that landed flips whose stamps the
/// machine held ([`PassReport::lane_span_us`]).
fn plans_of(passes: &[PassReport]) -> Vec<Plan> {
    passes
        .iter()
        .filter(|r| r.landed > 0 && r.lane_span_us > 0)
        .map(|r| Plan {
            boundary: r.boundary,
            landed: r.landed,
            late: r.late,
            window_us: r.window_us,
            span_us: r.lane_span_us,
        })
        .collect()
}

fn run() -> Result<(), GateError> {
    let levers = bloomery_levers::at_main(&[HOST_POPULATE, HOST_LOCK, CARD_DONTNEED])?;
    record::at_main(NAME, record::GATE_NVTIER);
    let host = levers.host();
    let audit = std::env::args().any(|a| a == "--audit");
    let path = Path::new(MODEL);
    let card = gate_card::init()?;
    let probe = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let inputs = inputs_of(&probe, Some(ROOM))?;
    let machine = machine_for_experts(
        card,
        inputs.spec.layers.len(),
        ubatch_for(CTX)? as u64,
        Experts::Card,
    );
    let plan = inputs.plan_with(&machine, CTX as u64, &PlanLevers::default(), Experts::Card)?;
    let mut pass = true;

    // `residency` (clause 5): the unset rule on the paged plan resolves
    // `off`.
    let unset = residency38(&plan, Lever38::Unset(None), Record::print)?;
    let unset_off = matches!(unset, Residency::Off);
    println!(
        "residency: the unset rule on the paged plan resolves {}: {}",
        if unset_off { "off" } else { "mid" },
        verdict(unset_off)
    );
    pass &= unset_off;
    // The residency word both arms run: a set `mid-p0-s1`, so the paged
    // arm's flips, the seam and a lane's drops stay held.
    let residency = residency38(&plan, Lever38::Set(MID, "mid-p0-s1"), Record::print)?;
    // `room`'s plan terms: the host need, the churn pool the residency
    // holds beside it, and the headroom the split left.
    // PIN(2026-10-09): `HostNeed::bytes` counts the plan's NVMe tier arena,
    // so the clause reads `plan_bytes` — the need with the arena out — and
    // adds the arena beside it: need + arena + pool fits the room, and the
    // headroom is what the room leaves past the arena and the need. Reading
    // `bytes` here counts the arena twice, once inside the need and once as
    // the clause's own term.
    let need = HostNeed::of(&plan, 0).plan_bytes();
    let pool = match residency {
        Residency::Mid { pinned, .. } => {
            ChurnPool::of(&plan, CARD38, pinned)
                .map_err(|e| format!("the churn pool: {e}"))?
                .bytes
        }
        Residency::Off => 0,
    };
    let headroom = plan.host.headroom_bytes;
    println!(
        "plan: room {ROOM} B pages {} B on the NVMe tier, arena {} B",
        plan.host.nvme_expert_bytes, plan.host.nvme_arena_bytes
    );
    pass &= refuse_clauses(path, &machine, &plan)?;
    let (layers, experts) = (plan.model.layers, plan.model.experts as u32);
    drop(plan);

    let ids = prose38(PROMPT)?;
    // `footprint`'s instrument: the model file's device's readahead window
    // and the edge band it gives; `--audit` reads no footprint.
    let window = if audit {
        None
    } else {
        Some(readahead_pages(path)?)
    };
    let band = window.map(edge_band);
    let mut paged = arm(
        path,
        &machine,
        Some(ROOM),
        residency,
        host,
        &ids,
        audit,
        &probe,
        band,
    )?;
    let ram = arm(
        path, &machine, None, residency, host, &ids, false, &probe, None,
    )?;

    let tokens_eq = paged.tokens == ram.tokens;
    let digests_eq = paged.digests == ram.digests;
    let bits = tokens_eq && digests_eq && !paged.tokens.is_empty();
    println!(
        "bits: {} greedy tokens equal {tokens_eq}, {} logits rows bit for bit {digests_eq}: {}",
        paged.tokens.len(),
        paged.digests.len(),
        verdict(bits)
    );
    pass &= bits;

    let tier = paged.tier.as_ref().ok_or("the paged arm built no arena")?;
    // `room`: the loaded tier's arena, the plan's host need and the churn
    // pool fit the room together, and the plan's headroom is what the room
    // leaves past the arena and the need.
    let arena = tier.budget();
    let room = headroom == i128::from(ROOM) - i128::from(arena) - i128::from(need)
        && need + arena + pool <= ROOM;
    println!(
        "room: need {need} B + arena {arena} B + churn pool {pool} B in the room {ROOM} B, \
         headroom {headroom} B: {}",
        verdict(room)
    );
    pass &= room;
    // `advice` (the PINs above): the arena's pages and the mapped shards'
    // are disjoint, and every drop run is inside one routed stack of its
    // layer and clear of every held byte.
    let mapped: Vec<(usize, usize)> = (0..probe.shard_count())
        .filter_map(|i| probe.shard(i))
        .flat_map(|g| g.iter_tensors().filter_map(|t| g.data(t).ok()))
        .map(|d| (d.as_ptr() as usize, d.as_ptr() as usize + d.len()))
        .collect();
    let arena_clear = !mapped.is_empty() && advice_disjoint(&[tier.range()], &mapped);
    let (runs, breach) = drop_advice(&probe, tier, layers, experts)?;
    let advice = arena_clear && runs > 0 && breach.is_none();
    println!(
        "advice: the arena's range is disjoint from {} mapped tensors {arena_clear}; {runs} drop runs \
         each inside one routed stack of its layer and clear of every held byte {}: {}",
        mapped.len(),
        breach.as_deref().unwrap_or("true"),
        verdict(advice)
    );
    pass &= advice;
    // `footprint` (clause 7): what the drop runs' interiors keep in the page
    // cache once their readers are done, held to the readahead the header
    // derives; the edge bands are printed.
    if let Some(ra) = window {
        let f = paged
            .footprint
            .as_ref()
            .ok_or("the paged arm read no footprint")?;
        let cpus = std::thread::available_parallelism().map_or(1, |n| n.get()) as u64;
        let prompt_bound = f.layers as u64 * cpus * 2 * ra;
        let steps_bound = paged.landed as u64 * 15 * ra / 2;
        // A page past the ones checked one by one counts held.
        let held = f.held.iter().filter(|&&h| h).count() as u64 + f.first.past;
        let base_ok = f.base.total > 0 && held == 0;
        let prompt_ok = f.prompt.interior <= prompt_bound;
        let steps_ok = f.steps.interior <= steps_bound;
        let footprint = base_ok && prompt_ok && steps_ok;
        for ((p, h), at) in f
            .first
            .pages
            .iter()
            .zip(&f.held)
            .zip(&f.at_prompt)
            .take(PAGES_SHOWN)
        {
            let id = expert_at(&probe, p, experts)?;
            let class = if u64::from(id) < f.card.get(p.layer).copied().unwrap_or(0) {
                "a card id"
            } else if tier.serves(p.layer, id) {
                "an NVMe-tier id"
            } else {
                "a host id"
            };
            println!(
                "footprint base page: layer {} row {} shard {} offset {} expert {id} ({class}), \
                 {} pages from its run's nearer end; held through the second drop {h}, \
                 resident at the prompt's read {at}",
                p.layer, p.row, p.shard, p.at, p.from_end
            );
        }
        if f.first.pages.len() > PAGES_SHOWN {
            println!(
                "footprint base pages: {} more checked, not shown",
                f.first.pages.len() - PAGES_SHOWN
            );
        }
        println!(
            "footprint: {} pages in the drop runs over {} paged layers, R {ra} pages \
             (read_ahead_kb), an edge band of {} pages at each run end (printed, not judged); \
             in the interiors {} after the tier's own drop (printed), {held} of them held \
             through a second drop ({} past the {HELD_CHECKED} checked one by one, counted \
             held; want 0: {}), {} after the prompt \
             (at most {} layers x {cpus} readers x 2R = {prompt_bound}: {}), {} after {STEPS} \
             steps (at most {} flips x 15R/2 = {steps_bound}: {}); in the bands {} / {} / {}; \
             the prompt's drops {} calls, {:.1} ms on the dropper's thread beside the \
             prompt's {:.1} ms ({:.2} %): {}",
            f.base.total,
            f.layers,
            f.band,
            f.base.interior,
            f.first.past,
            if base_ok {
                "ok"
            } else {
                "held after the tier's own drop: another mapper, or a page the drop cannot take"
            },
            f.prompt.interior,
            f.layers,
            if prompt_ok {
                "ok"
            } else {
                "the union's pages stay"
            },
            f.steps.interior,
            paged.landed,
            if steps_ok {
                "ok"
            } else {
                "a promotion's pages stay"
            },
            f.base.edge,
            f.prompt.edge,
            f.steps.edge,
            f.prompt_drops,
            f.prompt_drop_ns as f64 / 1e6,
            f.prompt_ns as f64 / 1e6,
            100.0 * f.prompt_drop_ns as f64 / f.prompt_ns.max(1) as f64,
            verdict(footprint)
        );
        pass &= footprint;
    } else {
        println!("footprint: off (--audit reads the file through the page cache by design)");
    }
    if let Some(r) = record::nvtier_of(Some(tier)) {
        r.print();
    }
    let s = paged.stats.ok_or("the paged arm read no counters")?;
    let counters = s.misses > 0
        && s.fills > 0
        && s.evictions > 0
        && paged.resident_max <= tier.budget()
        && s.buffered_reads == 0
        && ram.tier.is_none()
        && ram.stats.is_none();
    println!(
        "counters: misses {}, fills {}, evictions {}, resident at most {} of {} B, buffered reads \
         {}, all-RAM arm's tier {}: {}",
        s.misses,
        s.fills,
        s.evictions,
        paged.resident_max,
        tier.budget(),
        s.buffered_reads,
        if ram.tier.is_none() { "none" } else { "built" },
        verdict(counters)
    );
    pass &= counters;

    println!(
        "audit: {} (--audit {})",
        if audit {
            "every slot checked: PASS"
        } else {
            "off"
        },
        audit
    );

    let seam = paged.seam.as_ref().ok_or("the paged arm read no seam")?;
    let seam_ok =
        seam.checked > 0 && seam.filled > 0 && seam.filled < seam.checked && seam.wrong == 0;
    println!(
        "seam: host_resident answered the arena's books on {} ids ({} filled, {} unfilled), {} \
         wrong: {}",
        seam.checked,
        seam.filled,
        seam.checked - seam.filled,
        seam.wrong,
        verdict(seam_ok)
    );
    pass &= seam_ok;

    let flips = paged.landed > 0;
    // `late`, printed: one line a late flip the machine's debug ring holds,
    // where its job stood at the boundary and the five terms its
    // issue→boundary time splits into, which sum to it — the lane queue,
    // the victim's preparation and the source read (the drive), the staging
    // wait, and the copy stream (staged to this boundary, behind the lane's
    // backlog at its issue): the largest names the mechanism. The bug
    // catcher: the ring holds every late flip the passes counted, or its own
    // overflow count names the rest.
    let late = &paged.late_flips;
    let late_ok = paged.late as u64 == late.flips.len() as u64 + late.dropped;
    let us = |ns: u64| ns as f64 / 1e3;
    for f in late.flips.iter().take(LATE_SHOWN) {
        println!(
            "late flip: layer {} admit {} victim {}, issued at boundary {}, late at boundary {} \
             ({:?}); {:.1} µs from its issue: the lane queue {:.1}, the victim's prepare {:.1}, \
             the staging wait {:.1}, the source read {:.1}, the copy stream {:.1}; {} copies in \
             the lane's backlog at its issue",
            f.layer,
            f.admit,
            f.victim,
            f.issued_at,
            f.landed_at,
            f.at,
            us(f.budget_ns),
            us(f.queue_ns),
            us(f.prepare_ns),
            us(f.wait_ns),
            us(f.read_ns),
            us(f.copy_ns),
            f.ahead
        );
    }
    if late.flips.len() > LATE_SHOWN {
        println!(
            "late flips: {} more held in the ring, not shown",
            late.flips.len() - LATE_SHOWN
        );
    }
    // The landing boundaries' plans, printed: what the lane had (the window)
    // beside what it took (the span) — a plan whose window is under
    // `PLAN_MARGIN` spans is one the bound is for, and the run's sums beside
    // them say where the lane's time went.
    let steps = paged.steps.as_ref().ok_or("the paged arm read no steps")?;
    let mut plans = plans_of(&paged.passes);
    plans.sort_by(|a, b| a.margin().total_cmp(&b.margin()));
    for p in plans.iter().take(PLANS_SHOWN) {
        println!(
            "plan: landed at boundary {} ({} flips, {} late): the window {:.1} ms from the first \
             issue, the lane's span {:.1} ms to the last copy staged, window / span {:.2}",
            p.boundary,
            p.landed,
            p.late,
            p.window_us as f64 / 1e3,
            p.span_us as f64 / 1e3,
            p.margin()
        );
    }
    let in_steps = &paged.passes[paged.steps_from..];
    let sum = |of: fn(&PassReport) -> u64| in_steps.iter().map(of).sum::<u64>() as f64 / 1e3;
    let mean = |v: &[u64]| v.iter().sum::<u64>() as f64 / v.len().max(1) as f64 / 1e6;
    let thin = plans.iter().filter(|p| p.margin() < PLAN_MARGIN).count();
    println!(
        "plans: {} landing plans over {} boundaries; window / span at least {:.2}, {} under \
         {PLAN_MARGIN} (the margin the design asks for a judged 0); over the {STEPS} steps: a \
         step {:.1} ms, the boundaries' wait {:.1} ms, the lane's prepare {:.1} ms and stage \
         {:.1} ms (thread time), {} victims the lane prepared, {} slots the tier filled, {} \
         slots the landings freed",
        plans.len(),
        paged.passes.len(),
        plans.first().map_or(0.0, Plan::margin),
        thin,
        mean(&paged.step_ns),
        sum(|r| r.wait_us),
        sum(|r| r.prepare_us),
        sum(|r| r.stage_us),
        steps.lane[1].victims - steps.lane[0].victims,
        steps.tier(|t| t.fills),
        in_steps.iter().map(|r| r.released).sum::<usize>()
    );
    let late_in = paged.late <= LATE_MAX;
    let residency_ok = flips && paged.unresident == 0 && paged.faulting == 0;
    println!(
        "residency: under the set mid-p0-s1, flips landed {} (the seam ran: {flips}), unresident \
         {}, faulting {}: {}; late {} (at most {LATE_MAX}: {}), the ring holds {} and names {} \
         dropped (want {} together): {}",
        paged.landed,
        paged.unresident,
        paged.faulting,
        verdict(residency_ok),
        paged.late,
        verdict(late_in),
        late.flips.len(),
        late.dropped,
        paged.late,
        verdict(late_ok)
    );
    pass &= residency_ok && late_ok && late_in;

    // `exclusive` (clause 9): a promotion reads the expert's arena slot, and
    // the landing frees it. Over the steps: every slot read lends its three
    // parts (the file's bytes at the arena's skew) and a mapping read is the
    // lane's alone, since an id the arena holds unfilled when its copy opens
    // reads the file; the landings' frees and the copies' opens tally within
    // the flips in flight at the span's two ends; and the dropper ran no more
    // drops than the mapping's reads queued (a slot read leaves no page).
    // Over the run: the arena's books follow the frees — fills less
    // evictions less releases is what the seam finds filled and what the
    // resident bytes hold, within the fills the lane's threads had in flight
    // when the counters were read.
    let in_start = paged.passes[..paged.steps_from]
        .last()
        .map_or(0, |r| r.in_flight)
        + in_steps.first().and_then(|r| r.picked).unwrap_or(0);
    let in_start = in_start as u64;
    let in_end = in_steps.last().map_or(0, |r| r.in_flight) as u64;
    let slot_reads = steps.tier(|t| t.slot_reads);
    let mapping_reads = steps.tier(|t| t.mapping_reads);
    let slot_parts = steps.tier(|t| t.slot_parts);
    let freed = steps.tier(|t| t.releases);
    let found_none = steps.tier(|t| t.release_misses);
    let drops = steps.tier(|t| t.drops);
    let opened = slot_reads + mapping_reads;
    let landed_here = freed + found_none;
    // A copy opened before an end of the span can lend its parts after it,
    // at most `LANE_THREADS` copies at each end.
    let parts_slack = EXPERT_PARTS * LANE_THREADS as u64;
    let reads_ok = slot_reads > 0 && slot_parts.abs_diff(EXPERT_PARTS * slot_reads) <= parts_slack;
    let frees_ok = freed > 0 && landed_here <= opened + in_start && opened <= landed_here + in_end;
    let drops_ok = drops <= mapping_reads + in_start;
    let covered = (0..layers).filter(|&l| tier.covers(l)).count().max(1) as u64;
    let slot_bytes = tier.slot_set() / covered;
    let held_in = |c: &NvTierStats| (c.fills as i64) - (c.evictions as i64) - (c.releases as i64);
    let (held_a, held_b) = (held_in(&seam.counters[0]), held_in(&seam.counters[1]));
    let slack = LANE_THREADS as i64;
    let books_ok = (held_a.min(held_b) - slack..=held_a.max(held_b) + slack)
        .contains(&(seam.filled as i64))
        && seam.counters.iter().all(|c| {
            c.resident_bytes
                .abs_diff((held_in(c).max(0) as u64) * slot_bytes)
                <= slack as u64 * slot_bytes
        });
    let exclusive = reads_ok && frees_ok && drops_ok && books_ok;
    println!(
        "exclusive: over {STEPS} steps the lane's copies opened {opened} reads, {slot_reads} of an \
         arena slot and {mapping_reads} of the file mapping, the slot reads lending {slot_parts} \
         parts ({EXPERT_PARTS} a read: {}); the landings freed {freed} slots and found none to \
         free at {found_none}, against the {opened} opens within the {in_start} flips in flight \
         at the start and {in_end} at the end: {}; the dropper ran {drops} drops against the \
         {mapping_reads} mapping reads' ({}); over the run fills - evictions - releases = \
         {held_a} slots before the seam's read and {held_b} after, the seam finds {} filled and \
         the arena holds {} B at {slot_bytes} B a slot, within the {slack} fills the lane's \
         threads can have in flight at a read: {}: {}",
        verdict(reads_ok),
        verdict(frees_ok),
        verdict(drops_ok),
        seam.filled,
        seam.counters[1].resident_bytes,
        verdict(books_ok),
        verdict(exclusive)
    );
    pass &= exclusive;

    // `stepunion` (clause 8): the step port's union of two columns on a
    // paged plan — a load of its own made for two slots (the two arms' plan
    // counts one), a prompt in each slot, then two-column passes against the
    // same sequences stepped alone; and a second such load under the set
    // `mid-p0-s1`, whose flips land under the passes. The two arms' loads and
    // arena go first.
    let mut census = std::mem::take(&mut paged.census);
    drop((paged, ram));
    let step = stepunion_arm(path, &machine, host, &ids, &probe, band)?;
    let solo_bits = !step.tokens.is_empty()
        && step.prompts == step.solo_prompts
        && step.tokens == step.solo_tokens
        && step.digests == step.solo_digests;
    let grown = step.after.interior.saturating_sub(step.before.interior);
    let step_bound = window.map(|ra| step.flips as u64 * 15 * ra / 2);
    let step_pages_ok = step_bound.is_none_or(|bound| grown <= bound);
    let served = step.misses > 0 && step.fills > 0;
    let stepunion = solo_bits && step_pages_ok && served;
    println!(
        "stepunion: {STEP_COLS} two-column passes on a two-slot load, {} tokens and logits rows \
         and both prompts' argmax bit for bit the same sequences stepped alone: {}; {}; misses \
         +{}, fills +{} over the passes (the arena served them: {served}): {}",
        step.tokens.len(),
        verdict(solo_bits),
        match step_bound {
            Some(bound) => format!(
                "{} interior pages over {} paged layers before the passes, {} after, +{grown} \
                 (at most {} flips x 15R/2 = {bound}): {}; in the edge bands {} before, {} \
                 after (printed, not judged)",
                step.before.interior,
                step.layers,
                step.after.interior,
                step.flips,
                verdict(step_pages_ok),
                step.before.edge,
                step.after.edge
            ),
            None => "the footprint's read is off (--audit reads the file through the page cache \
                     by design)"
                .to_string(),
        },
        step.misses,
        step.fills,
        verdict(stepunion)
    );
    pass &= stepunion;

    // `stepunion` under a set `mid-p0-s1`: the flips land while the two-column
    // chain runs, so a lane's victim fill races the chain's own reads of the
    // arena (the claims, the pins and the frees all under the books' one
    // lock). The paged load's bits are the all-RAM load's, as clause 1's
    // one-slot pair: one schedule, and the fills' and the copies' timing
    // moves no bit.
    let mid = slots_mid_arm(path, &machine, host, &ids, Some(ROOM))?;
    let mid_ram = slots_mid_arm(path, &machine, host, &ids, None)?;
    let mid_bits = !mid.tokens.is_empty()
        && mid.prompts == mid_ram.prompts
        && mid.tokens == mid_ram.tokens
        && mid.digests == mid_ram.digests
        && mid.flips == mid_ram.flips;
    let (mid_misses, mid_fills, mid_census) = mid.served.ok_or("the paged mid arm took no tier")?;
    let mid_flips = mid.flips > 0 && mid.unresident == 0 && mid.faulting == 0;
    let mid_served = mid_misses > 0 && mid_fills > 0;
    let stepmid = mid_bits && mid_flips && mid_served && mid_ram.served.is_none();
    println!(
        "stepunion mid: {STEP_COLS} two-column passes on a two-slot load under the set mid-p0-s1, \
         flips landed {} (the lane filled and copied under the chain: {}; the all-RAM load's {}), \
         unresident {}, faulting {}, late {} (printed, not judged); {} tokens and logits rows and \
         both prompts' argmax bit for bit the all-RAM load's: {}; misses +{mid_misses}, fills \
         +{mid_fills} over the passes (the arena served them: {mid_served}): {}",
        mid.flips,
        mid.flips > 0,
        mid_ram.flips,
        mid.unresident,
        mid.faulting,
        mid.late,
        mid.tokens.len(),
        verdict(mid_bits),
        verdict(stepmid)
    );
    pass &= stepmid;

    // `census`: each phase's tier census — the paged arm's prompt call and
    // steps, the stepunion arm's two-column passes — printed as its record,
    // its picks by where they were served and the bytes the phase moved
    // between the tiers, and held to one bug catcher: every phase's card,
    // arena and file picks sum to its picks (a reader that serves picks it
    // does not count leaves them out).
    census.push(("slots", 2 * STEP_COLS, step.census));
    census.push(("slots_mid", 2 * STEP_COLS, mid_census));
    let mut census_ok = census.len() == 4;
    for (phase, tokens, c) in &census {
        census_record(phase, c).print();
        census_ok &= c.picks > 0 && c.adds_up() && c.tokens == *tokens as u64;
        let per = (*tokens).max(1) as f64;
        println!(
            "census {phase}: {tokens} tokens run (the census counts {}); a token {:.1} picks: card \
             {:.1}, arena {:.1}, file {:.1} (card + arena + file = picks: {}); fills {:.2} MB, \
             promotions {:.2} MB, demotions {:.2} MB; the dropper's {:.1} calls, {:.3} ms; {:.1} \
             dispatches waited, {:.3} ms",
            c.tokens,
            c.picks as f64 / per,
            c.card as f64 / per,
            c.arena as f64 / per,
            c.file as f64 / per,
            c.adds_up(),
            c.fill_bytes as f64 / per / 1e6,
            c.promote_bytes as f64 / per / 1e6,
            c.demote_bytes as f64 / per / 1e6,
            c.drops as f64 / per,
            c.drop_ns as f64 / per / 1e6,
            c.dispatch_waits as f64 / per,
            c.dispatch_wait_ns as f64 / per / 1e6,
        );
    }
    println!(
        "census: {} phases, each one's card, arena and file picks summing to its picks and its \
         tokens the ones the gate ran: {}",
        census.len(),
        verdict(census_ok)
    );
    pass &= census_ok;

    // `pool` (clause 10): no phase's dispatch waited for another caller's
    // job. The chain is the pool's one dispatcher: the lane's victim fills
    // read on the lane's own threads (`NvTier::ensure_here`), where a fill on
    // the pool took the dispatch mutex the chain's host compute holds.
    let waits: Vec<String> = census
        .iter()
        .map(|(phase, _, c)| format!("{phase} {}", c.dispatch_waits))
        .collect();
    let pool_ok = census.iter().all(|(_, _, c)| c.dispatch_waits == 0);
    println!(
        "pool: dispatches that waited for another caller's job, by phase: {} (want 0 in every \
         one): {}",
        waits.join(", "),
        verdict(pool_ok)
    );
    pass &= pool_ok;

    if pass {
        println!("{NAME}: every clause passed");
        Ok(())
    } else {
        Err(checks_failed())
    }
}

/// `refuse` (clause 4): each refusal by name before any load. The room
/// under the tier's floor is the plan's; the arena budgets under the
/// arena's floor — one byte, the routed width less one slot sets, the
/// floor's own slots less one — are the tier's own build over the paged
/// plan with that arena, beside the floor's own budget, which loads with
/// its slots; the direct-read probe is `DirectFile::open` on a tmpfs file
/// when the mount refuses direct IO (a mount that accepts it names that
/// fact and the clause holds nothing there).
fn refuse_clauses(
    path: &Path,
    machine: &Machine,
    plan: &model::placement::Plan<'_>,
) -> Result<bool, GateError> {
    let mut low = inputs_of(&Split::open(path)?, Some(1 << 30))?;
    low.room = (1 << 30, HostRead::Given);
    let floor = match low.plan_with(machine, CTX as u64, &PlanLevers::default(), Experts::Card) {
        Err(e) => e.to_string().contains("the NVMe expert tier's floor"),
        Ok(_) => false,
    };
    println!(
        "refuse floor: a 1 GiB room is refused by name (HostRoomFloor): {}",
        verdict(floor)
    );

    let split = Arc::new(Split::open(path)?);
    let with_arena = |bytes: u64| {
        let mut p = plan.clone();
        p.host.nvme_arena_bytes = bytes;
        NvTier::of_paged(&p, &split, true)
    };
    let slot = match with_arena(1) {
        Err(e) => e.to_string().contains("carves 0 slots"),
        Ok(_) => false,
    };
    println!(
        "refuse slot: an arena of 1 B is refused by name (carves 0 slots): {}",
        verdict(slot)
    );
    // The arena's floor at load (`slots_of`, the placement's `ArenaFloor`):
    // an arena of the routed width less one slot sets, and of the floor's
    // own slots less one (the lane claim's), each refused by name with the
    // floor's terms; the floor's own bytes load, carve exactly its slots,
    // and carve the slot set the placement counts — the window geometry's
    // one owner.
    let least = ArenaFloor::of(plan, ARENA_COLUMNS)?;
    let named = |slots: u64| match with_arena(slots * least.slot_set) {
        Err(e) => {
            let text = e.to_string();
            let named = text.contains(&format!("carves {slots} slots a paged layer"))
                && text.contains(&format!(
                    "under the floor of {} slots ({} × {} ids a call + {} lane claims",
                    least.slots, least.k, least.columns, least.lanes
                ));
            println!("refuse floor: {} B: {text}", slots * least.slot_set);
            named
        }
        Ok(_) => false,
    };
    let floor_k = named(least.k - 1);
    println!(
        "refuse floor-k: an arena of {} × {} B (the routed width less one) is refused at load \
         by name: {}",
        least.k - 1,
        least.slot_set,
        verdict(floor_k)
    );
    let floor_lane = named(least.slots - 1);
    println!(
        "refuse floor-lane: an arena of {} × {} B (the floor less its lane claim) is refused at \
         load by name: {}",
        least.slots - 1,
        least.slot_set,
        verdict(floor_lane)
    );
    let accepted = match with_arena(least.bytes) {
        Ok(Some(t)) => {
            let s = t.stats();
            println!(
                "accept floor: an arena of {} B carves {} slots a paged layer over {} B a slot \
                 set (the floor: {least})",
                least.bytes,
                s.slots_per_layer,
                t.slot_set()
            );
            s.slots_per_layer == least.slots && t.slot_set() == least.slot_set
        }
        Ok(None) => false,
        Err(e) => {
            println!("accept floor: the floor's own arena was refused: {e}");
            false
        }
    };
    println!(
        "accept floor: the floor's arena loads with its slots and the placement's slot set: {}",
        verdict(accepted)
    );

    let shm = Path::new("/dev/shm").join(format!("{NAME}-probe-{}", std::process::id()));
    std::fs::write(&shm, vec![0u8; engram::direct::DIRECT_ALIGN])?;
    let opened = engram::direct::DirectFile::open(&shm);
    std::fs::remove_file(&shm)?;
    let probe = match opened {
        Err(e) => {
            let named = e.to_string().to_ascii_lowercase().contains("direct");
            println!(
                "refuse probe: a tmpfs file is refused ({e}): {}",
                verdict(named)
            );
            named
        }
        Ok(_) => {
            println!(
                "refuse probe: this kernel's tmpfs accepts O_DIRECT, so no mount here refuses \
                 it; the probe's refusal is engram's hw test's (hw_engram_direct_file_reads_aligned_spans)"
            );
            true
        }
    };
    Ok(floor && slot && floor_k && floor_lane && accepted && probe)
}
