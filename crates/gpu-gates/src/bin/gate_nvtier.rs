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
//!   (`HostRoomFloor`), an arena budget under one slot's span, and the
//!   direct-read probe of a mount without direct IO, each by name.
//! - `residency` (clause 5): the paged plan's unset rule resolves `mid`
//!   with an empty churn pool (`ChurnPool::of` leaves the paged stacks
//!   out: the arena serves their victims), flips land on the
//!   paged arm, and its `unresident` and `faulting` are 0 and its `late`
//!   flips 0 — bug catchers, structurally unreachable once the seam serves
//!   the victims the arena holds from its books.
//! - `room` (clause 6): the loaded tier's arena, the plan's host need and
//!   the churn pool the residency holds fit the room together, and the
//!   plan's headroom is what the room leaves past the arena and the need.
//! - `footprint` (clause 7, the paged arm, off under `--audit`, whose
//!   buffered reads fill the page cache by design): the page cache holds no
//!   page of the tier's drop runs ([`NvTier::drop_region`]) past a folio
//!   from a run's ends ([`FOLIO_MAX`]) — after the tier's own drop of every
//!   run before the prompt (the premise: no other process maps the file;
//!   every gate that maps it holds the V4.1 load lock this one holds), after
//!   the prompt call (its union and its lanes dropped what they read), and,
//!   with the residue dropped again so the two readers stay apart, after the
//!   [`STEPS`] steps (the promotions' lanes dropped theirs). The page cache
//!   is read with `mincore` over the probe's own open of the file, not the
//!   engine's mapping. The line also prints the prompt's drops and their
//!   wall against the prompt's.
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
use bloomery_gpu::host::nvtier::{DropRuns, NvTier, NvTierStats, advice_disjoint};
use bloomery_gpu::host::swap::{PassReport, Residency, SwapSource};
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::residency38::{CARD38, Lever38, residency38};
use bloomery_gpu_gates::{Fnv1a64, GateError, checks_failed, data_dir, exit_with, verdict};
use bloomery_levers::{CARD_DONTNEED, HOST_LOCK, HOST_POPULATE, HostCfg};
use gguf::Split;
use model::arch::qwen35moe::place::{Experts, PlanInputs, machine_for_experts};
use model::placement::churn::ChurnPool;
use model::placement::host_lock::page_bytes;
use model::placement::workstation::{HostNeed, HostRead};
use model::placement::{Machine, PlanLevers};
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
/// The largest page-cache folio, in bytes: x86-64 builds them up to PMD
/// order (2 MiB), and `POSIX_FADV_DONTNEED` skips a folio that reaches past
/// the range it is given, so a run's first and last folio may keep pages a
/// correct drop named. `footprint` judges the pages past it from a run's
/// ends and prints the rest.
const FOLIO_MAX: u64 = 2 << 20;

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
/// 0, the flips that landed, the arena's counters at the end and the most
/// resident bytes it held at any step, and the arena itself when the load
/// attached one.
struct ArmRun {
    tokens: Vec<u32>,
    digests: Vec<u64>,
    unresident: usize,
    faulting: usize,
    late: usize,
    landed: usize,
    resident_max: u64,
    stats: Option<NvTierStats>,
    tier: Option<Arc<NvTier>>,
    seam: Option<Seam>,
    footprint: Option<Footprint>,
}

/// Resident pages over the tier's drop runs (`footprint`): those past
/// [`FOLIO_MAX`] from a run's ends (the interior, judged), those nearer
/// (the edges, printed), and the runs' pages.
#[derive(Clone, Copy, Default)]
struct Pages {
    interior: u64,
    edge: u64,
    total: u64,
}

/// The paged arm's footprint: after the tier's own drop before the prompt,
/// after the prompt and after the steps; and the prompt's drops (calls,
/// wall) against the prompt's wall.
struct Footprint {
    base: Pages,
    prompt: Pages,
    steps: Pages,
    prompt_drops: u64,
    prompt_drop_ns: u64,
    prompt_ns: u64,
}

/// The residency seam's answers over the ids the arena serves, against the arena's own books.
struct Seam {
    checked: usize,
    filled: usize,
    wrong: usize,
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
/// tier and no `--audit`, the footprint read through `probe`, the gate's own
/// open of the file.
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
) -> Result<ArmRun, GateError> {
    let file = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let inputs = inputs_of(&file, room)?;
    let ub = ubatch_for(CTX)?;
    let plan = inputs.plan_with(machine, CTX as u64, &PlanLevers::default(), Experts::Card)?;
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
    take_passes(&mut s)?;
    // `footprint`: the tier's drop runs, its own definition; the page cache
    // emptied of them by the tier's own drop first, so a page an earlier run
    // left warm is not this one's.
    let foot = match &tier {
        Some(t) if !audit => {
            let regions = (0..plan.model.layers)
                .filter(|&l| t.covers(l))
                .map(|l| Ok((l, t.drop_region(l)?)))
                .collect::<Result<Vec<_>, GateError>>()?;
            drop_all(t, &regions)?;
            Some((Arc::clone(t), regions))
        }
        _ => None,
    };
    let base = match &foot {
        Some((_, regions)) => resident(probe, regions)?,
        None => Pages::default(),
    };
    let drops0 = tier.as_ref().map(|t| t.stats());
    let mut run = ArmRun {
        tokens: Vec::with_capacity(STEPS + 1),
        digests: Vec::with_capacity(STEPS + 1),
        unresident: 0,
        faulting: 0,
        late: 0,
        landed: 0,
        resident_max: 0,
        stats: None,
        tier: tier.clone(),
        seam: None,
        footprint: None,
    };
    let note = |s: &mut Session<Body38>, run: &mut ArmRun| -> Result<(), GateError> {
        for (_, r) in take_passes(s)? {
            run.unresident += r.unresident.count();
            run.faulting += r.faulting.count();
            run.late += r.late;
            run.landed += r.landed;
        }
        if let Some(t) = &tier {
            run.resident_max = run.resident_max.max(t.stats().resident_bytes);
        }
        Ok(())
    };
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
    let drops1 = tier.as_ref().map(|t| t.stats());
    // The prompt's residue dropped through the tier, so the steps' read
    // names only what the steps' own readers left.
    let prompt_pages = match &foot {
        Some((t, regions)) => {
            let p = resident(probe, regions)?;
            drop_all(t, regions)?;
            p
        }
        None => Pages::default(),
    };
    for _ in 0..STEPS {
        let out = s.step(next, Want::Logits)?;
        if let Out::Logits { row, .. } = out {
            run.digests.push(Fnv1a64::default().f32s(row).value());
        }
        next = out.argmax();
        run.tokens.push(next);
        note(&mut s, &mut run)?;
    }
    run.stats = tier.as_ref().map(|t| t.stats());
    if let (Some((_, regions)), Some(d0), Some(d1)) = (&foot, drops0, drops1) {
        run.footprint = Some(Footprint {
            base,
            prompt: prompt_pages,
            steps: resident(probe, regions)?,
            prompt_drops: d1.drops - d0.drops,
            prompt_drop_ns: d1.drop_ns - d0.drop_ns,
            prompt_ns,
        });
    }
    run.seam = seam_of(&s, plan.model.layers, plan.model.experts as u32)?;
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
    Ok(Some(seam))
}

/// The tier's own drop of every run of `regions` ([`NvTier::drop_layer`]).
fn drop_all(tier: &NvTier, regions: &[(usize, DropRuns)]) -> Result<(), GateError> {
    for &(layer, _) in regions {
        tier.drop_layer(layer)?;
    }
    Ok(())
}

/// The page cache's residency over the runs of `regions`, read with
/// `mincore` through `probe`'s mapping of each shard (the gate's own open of
/// the file: a file page's residency is the page cache's, whichever mapping
/// asks): the pages past [`FOLIO_MAX`] from a run's ends, those nearer, and
/// the runs' pages.
fn resident(probe: &Split, regions: &[(usize, DropRuns)]) -> Result<Pages, GateError> {
    let page = page_bytes().map_err(|e| format!("the page size: {e}"))?;
    let band = FOLIO_MAX / page;
    let mut pages = Pages::default();
    let mut vec = Vec::new();
    for (layer, rows) in regions {
        for (shard, runs) in rows {
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
                            "layer {layer} shard {shard}: the run {run:?} is past the mapping's                              {} B",
                            map.len()
                        )
                    })?;
                let n = span.len().div_ceil(page as usize);
                vec.clear();
                vec.resize(n, 0u8);
                // SAFETY: `span` lies inside the probe's live read-only
                // mapping of the shard and starts on a page boundary (the
                // mapping starts on one, and a drop run is whole pages of
                // file offsets); `vec` holds one byte for each of its pages.
                // mincore reads the page tables and the page cache and
                // writes only `vec`.
                let rc = unsafe {
                    libc::mincore(
                        span.as_ptr().cast_mut().cast(),
                        span.len(),
                        vec.as_mut_ptr(),
                    )
                };
                if rc != 0 {
                    return Err(format!(
                        "mincore over layer {layer} shard {shard} bytes {run:?}: {}",
                        std::io::Error::last_os_error()
                    )
                    .into());
                }
                let n = n as u64;
                for (i, _) in vec.iter().enumerate().filter(|&(_, &v)| v & 1 != 0) {
                    let i = i as u64;
                    if i < band || i + band >= n {
                        pages.edge += 1;
                    } else {
                        pages.interior += 1;
                    }
                }
                pages.total += n;
            }
        }
    }
    Ok(pages)
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
                            "layer {l}: the run {run:?} is not inside one routed stack of the layer                              ({name}, bytes {t0}..{t1})"
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
                                "layer {l}: a drop run names a held byte — {run:?} meets expert                                  {id}'s bytes {a}..{b} of {name}"
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

    // The residency word both arms run: the unset rule's answer on the
    // paged plan, which resolves `mid` with the churn pool 0.
    let residency = residency38(&plan, Lever38::Unset(None), Record::print)?;
    // `room`'s plan terms: the host need, the churn pool the residency
    // holds beside it, and the headroom the split left.
    let need = HostNeed::of(&plan, 0).bytes();
    let pool = match residency {
        Residency::Mid { pinned, .. } => {
            ChurnPool::of(&plan, CARD38, pinned)
                .map_err(|e| format!("the churn pool: {e}"))?
                .bytes
        }
        Residency::Off => 0,
    };
    let headroom = plan.host.headroom_bytes;
    let Residency::Mid { pinned, spares } = residency else {
        return Err("the paged plan's unset residency rule did not resolve mid".into());
    };
    println!("residency: mid-p{pinned}-s{spares} (the unset rule on the paged plan): PASS");
    println!(
        "plan: room {ROOM} B pages {} B on the NVMe tier, arena {} B",
        plan.host.nvme_expert_bytes, plan.host.nvme_arena_bytes
    );
    pass &= refuse_clauses(path, &machine, &plan)?;
    let (layers, experts) = (plan.model.layers, plan.model.experts as u32);
    drop(plan);

    let ids = prose38(PROMPT)?;
    let paged = arm(
        path,
        &machine,
        Some(ROOM),
        residency,
        host,
        &ids,
        audit,
        &probe,
    )?;
    let ram = arm(path, &machine, None, residency, host, &ids, false, &probe)?;

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
        "advice: the arena's range is disjoint from {} mapped tensors {arena_clear}; {runs} drop runs          each inside one routed stack of its layer and clear of every held byte {}: {}",
        mapped.len(),
        breach.as_deref().unwrap_or("true"),
        verdict(advice)
    );
    pass &= advice;
    // `footprint` (clause 7): no page of the drop runs past a folio from a
    // run's ends stays in the page cache once its reader is done.
    if audit {
        println!("footprint: off (--audit reads the file through the page cache by design)");
    } else {
        let f = paged
            .footprint
            .as_ref()
            .ok_or("the paged arm read no footprint")?;
        let why = if f.base.interior > 0 {
            "the runs stay resident after the tier's own drop: another process maps the file"
        } else if f.prompt.interior > 0 {
            "the union's pages stay"
        } else if f.steps.interior > 0 {
            "a promotion's pages stay"
        } else {
            "none stay"
        };
        let footprint = f.base.total > 0 && why == "none stay";
        println!(
            "footprint: {} pages in the drop runs; resident past {FOLIO_MAX} B of a run's ends {}              after the tier's own drop, {} after the prompt, {} after {STEPS} steps (at the edges {} /              {} / {}); the prompt's drops {} calls, {:.1} ms of its {:.1} ms ({:.2} %); {why}: {}",
            f.base.total,
            f.base.interior,
            f.prompt.interior,
            f.steps.interior,
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
    let residency_ok = flips && paged.unresident == 0 && paged.faulting == 0 && paged.late == 0;
    println!(
        "residency: flips landed {} (the seam ran: {flips}), unresident {}, faulting {}, late {}: {}",
        paged.landed,
        paged.unresident,
        paged.faulting,
        paged.late,
        verdict(residency_ok)
    );
    pass &= residency_ok;

    if pass {
        println!("{NAME}: every clause passed");
        Ok(())
    } else {
        Err(checks_failed())
    }
}

/// `refuse` (clause 4): each refusal by name before any load. The room
/// under the tier's floor is the plan's; the arena budget under one slot is
/// the tier's own build over the paged plan with a one-byte arena; the
/// direct-read probe is `DirectFile::open` on a tmpfs file when the mount
/// refuses direct IO (a mount that accepts it names that fact and the clause
/// holds nothing there).
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

    let mut tiny = plan.clone();
    tiny.host.nvme_arena_bytes = 1;
    let split = Arc::new(Split::open(path)?);
    let slot = match NvTier::of_paged(&tiny, &split, true) {
        Err(e) => e.to_string().contains("under one slot"),
        Ok(_) => false,
    };
    println!(
        "refuse slot: an arena of 1 B is refused by name (under one slot): {}",
        verdict(slot)
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
    Ok(floor && slot && probe)
}
