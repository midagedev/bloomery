//! Load gate ② (`docs/v41-placement.md` §7 ②): V4.1-Flash onto this machine's
//! cards as a plan of design §5 places it, proven against the plan. A
//! correctness run: every time and memory figure it prints is a runtime value.
//!
//! The plan is built from `model::placement::workstation`, the figures gate ①
//! plans from. Before any upload every plan card is found by name
//! (`Gpu::for_card` — never by ordinal: `BLOOMERY_CARD=both` exposes the 3090
//! first) and must have free, once its context exists, the bytes the plan puts
//! on it besides the context; a missing card or a short one refuses the run.
//! Then, per card in stage order:
//!
//! 1. Resident bytes: every segment `Weights::load_placed` uploaded holds the
//!    plan's buffers for it, byte for byte, nothing else is resident, and the
//!    card's total is the plan's dense + expert bytes.
//! 2. Device memory: `cuMemGetInfo` after the context and after the load. What
//!    is left free must cover the card's headroom + KV + scratch
//!    (`CardTotals`; the gate allocates neither KV nor scratch). The plan
//!    counts the allocator's rounding as its own term, so this holds exactly
//!    when the context, plus whatever the load took beyond its resident bytes
//!    and that term, fits what the plan set aside for the context. The residue
//!    of that gap over the plan's rounding term must be zero: the term's rule
//!    for small allocations is fitted to this loader's allocation list, and a
//!    changed list would otherwise vanish into the context's slack. Printed:
//!    the bytes the load asked for next to what `cuMemGetInfo` lost, the
//!    context's cost, the allocation count and both granules.
//! 3. Read-back: the first whole tensor of each card format on the card and
//!    the first expert-prefix stack segment, read back and compared bit for
//!    bit with a packing of the same file bytes written here from the file
//!    formats' definitions — not the loader's code.
//!
//! Host half, after the cards, through the engine's own load step
//! (`HostResidency::at_load` with the engine's levers, the call
//! `GpuModel::load_placed` makes):
//!
//! 4. Resident host set: before anything is loaded the shard files are
//!    dropped from the page cache (`posix_fadvise(DONTNEED)`, the pages
//!    another process maps excepted — the count still resident is printed),
//!    so a lazy mapping cannot pass by luck; after the host step `mincore`
//!    must find every page of the plan's host set resident. Red with
//!    `BLOOMERY_HOST_POPULATE=0`. `Cached` of `/proc/meminfo` before the
//!    load, after the cards and after the host step is printed, never
//!    asserted: with `BLOOMERY_CARD_DONTNEED` at its default the cards'
//!    file bytes leave the page cache as they are uploaded.
//!
//! With `--lock` (every host segment) or `--lock-layers L0..L1` (the host
//! segments of those layers, which narrows check 4's set too), the engine's
//! lever `BLOOMERY_HOST_LOCK=1` must be set, and the host step locks the set:
//! `VmLck` must grow by the page-rounded union of the locked ranges — derived
//! here from the headers — exactly. Lock wall time per shard and in total is
//! printed, never asserted.
//!
//! `--plan a|b` picks design §5's plan, b by default: plan (b) splits the
//! layers over both cards (`workstation::plan_b`), and this gate is the tree's
//! one run of that two-card staging, the exercise ahead of the two-card tier
//! (T2 of `docs/plan.md`'s target profile). The engine itself opens one card:
//! `body::open` in `bloomery_gpu_deepseek41` refuses a placement on more than
//! one. Plan (a) is the one-card plan.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_load_v41: built without the `gpu` feature; see `just stage-gpu-load-v41`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_load_v41", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use std::collections::BTreeMap;
    use std::fmt;
    use std::ops::Range;
    use std::time::Instant;

    use bloomery_gpu::Gpu;
    use bloomery_gpu::hybrid::{HostLevers, HostResidency, host_levers};
    use bloomery_gpu::weights::{DevWeight, Weights};
    use bloomery_gpu_gates::{GateError, bits_equal, bytes_to_words, checks_failed, verdict};
    use cuda_core::CudaStream;
    use gguf::Split;
    use model::arch::deepseek41::place::PlanInputs;
    use model::placement::host_lock::{HostSet, PageDrop, page_bytes};
    use model::placement::{
        Card, CardFormat, CardTotals, Device, ExpertList, Format, ModelTensor, ModelTensors, Plan,
        Segment, workstation,
    };

    /// Design §5's two plans.
    #[derive(Clone, Copy)]
    enum PlanId {
        A,
        B,
    }

    impl fmt::Display for PlanId {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(match self {
                PlanId::A => "a",
                PlanId::B => "b",
            })
        }
    }

    /// Which host segments the host half locks.
    enum Lock {
        All,
        Layers(Range<usize>),
    }

    impl Lock {
        fn keeps(&self, t: &ModelTensor) -> bool {
            match self {
                Lock::All => true,
                Lock::Layers(r) => t.layer.is_some_and(|l| r.contains(&l)),
            }
        }
    }

    /// The host segments the host step walks: `lock`'s, or every one.
    fn keeps(lock: Option<&Lock>, t: &ModelTensor) -> bool {
        lock.is_none_or(|l| l.keeps(t))
    }

    struct Args {
        plan: PlanId,
        lock: Option<Lock>,
    }

    fn parse_args() -> Result<Args, GateError> {
        const USAGE: &str = "usage: gate_load_v41 [--plan a|b] [--lock | --lock-layers L0..L1]";
        let mut args = Args {
            plan: PlanId::B,
            lock: None,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--plan" => {
                    args.plan = match it.next().as_deref() {
                        Some("a") => PlanId::A,
                        Some("b") => PlanId::B,
                        other => return Err(format!("--plan {other:?}: {USAGE}").into()),
                    }
                }
                "--lock" => args.lock = Some(Lock::All),
                "--lock-layers" => {
                    let v = it.next().unwrap_or_default();
                    let range = v
                        .split_once("..")
                        .and_then(|(a, b)| Some(a.parse().ok()?..b.parse().ok()?))
                        .filter(|r: &Range<usize>| !r.is_empty());
                    let Some(range) = range else {
                        return Err(format!("--lock-layers {v:?}: {USAGE}").into());
                    };
                    args.lock = Some(Lock::Layers(range));
                }
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        Ok(args)
    }

    pub fn run() -> Result<(), GateError> {
        let args = parse_args()?;
        let levers = host_levers()?;
        if args.lock.is_some() && !levers.lock {
            return Err(
                "--lock and --lock-layers go through the engine's lever: set BLOOMERY_HOST_LOCK=1"
                    .into(),
            );
        }
        println!(
            "levers: BLOOMERY_HOST_POPULATE={} BLOOMERY_HOST_LOCK={} BLOOMERY_CARD_DONTNEED={}",
            u8::from(levers.populate),
            u8::from(levers.lock),
            u8::from(levers.card_dontneed)
        );
        let path = workstation::model_v41();
        let split = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
        let inputs = PlanInputs::read(&split)?;
        let machine = match args.plan {
            PlanId::A => workstation::plan_a(inputs.model.layers),
            PlanId::B => workstation::plan_b(inputs.model.layers),
        };
        let plan = inputs
            .plan(&machine, workstation::CTX_MAX)
            .map_err(|e| format!("plan ({}): {e}", args.plan))?;
        println!(
            "plan ({}) of {path}: {} layers, ctx_max {}",
            args.plan, plan.model.layers, plan.ctx_max
        );
        for (card, t) in plan.machine.cards.iter().zip(&plan.cards) {
            println!(
                "  {} layers {:?}{}: dense {} experts {} ({}) rounding {} KV {} scratch {} context {} headroom {}",
                card.name,
                card.layers,
                if card.head { " +head" } else { "" },
                t.dense_bytes,
                t.expert_bytes,
                t.experts,
                t.rounding_bytes,
                t.kv_bytes,
                t.scratch_bytes,
                t.context_bytes,
                t.headroom_bytes
            );
        }
        let h = &plan.host;
        println!(
            "  host: experts {} ({}) tables {} ring shadows {} (page-locked; this gate allocates \
             none) reserves {} headroom {}",
            h.expert_bytes,
            h.experts,
            h.table_bytes,
            h.shadow_bytes,
            h.reserve_bytes,
            h.headroom_bytes
        );

        let cards = open_cards(&plan)?;
        let lock = args.lock.as_ref();
        let set = HostSet::of(&split, &plan, |t| keeps(lock, t))?;
        evict(&split, &set)?;
        let cached_before = meminfo_cached()?;
        let mut ok = true;
        for (c, on) in cards.iter().enumerate() {
            ok &= check_card(&split, &plan, c, on)?;
        }
        let cached_cards = meminfo_cached()?;
        ok &= check_host(&split, &plan, lock, levers, [cached_before, cached_cards])?;
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: gate_load_v41 — plan ({}): every card holds its segments at the plan's bytes, \
             its free memory covers the plan's headroom, KV and scratch, and its samples read back \
             as the file's bytes{}",
            args.plan,
            if args.lock.is_some() {
                "; the host set is resident and locked, the lock the derived page union"
            } else {
                "; the host set is resident"
            }
        );
        Ok(())
    }

    /// A plan card's device, and its memory once its context exists.
    struct OnCard {
        gpu: Gpu,
        total: u64,
        free_ctx: u64,
    }

    /// Every plan card's device, found by name, each with the bytes the plan
    /// puts on it besides the context free — or the refusal, before any upload.
    fn open_cards(plan: &Plan<'_>) -> Result<Vec<OnCard>, GateError> {
        let mut out = Vec::with_capacity(plan.machine.cards.len());
        let mut short = Vec::new();
        for (card, t) in plan.machine.cards.iter().zip(&plan.cards) {
            let gpu = Gpu::for_card(&card.name).map_err(|e| {
                format!(
                    "refusing before any upload: plan card {} is not here: {e}",
                    card.name
                )
            })?;
            let (free, total) = gpu.mem_info()?;
            let (free, total) = (free as u64, total as u64);
            let need =
                t.dense_bytes + t.expert_bytes + t.rounding_bytes + t.kv_bytes + t.scratch_bytes;
            println!(
                "card {} is {}: {total} B, {free} B free after its context; the plan puts {need} B \
                 on it besides the context (resident + rounding + KV + scratch)",
                card.name,
                gpu.device_name()?
            );
            if free < need {
                short.push(format!(
                    "{}: {free} B free, the plan needs {need} B",
                    card.name
                ));
            }
            out.push(OnCard {
                gpu,
                total,
                free_ctx: free,
            });
        }
        if !short.is_empty() {
            return Err(format!("refusing before any upload: {}", short.join("; ")).into());
        }
        Ok(out)
    }

    /// Load card `c`'s segments and run checks 1–3 on them.
    fn check_card(
        split: &Split,
        plan: &Plan<'_>,
        c: usize,
        on: &OnCard,
    ) -> Result<bool, GateError> {
        let card = &plan.machine.cards[c];
        let start = Instant::now();
        let w = Weights::load_placed(on.gpu.stream(), split, plan, c)?;
        let wall = start.elapsed();
        let (free_load, _) = on.gpu.mem_info()?;
        let resident = w.resident_bytes() as u64;
        println!(
            "card {}: loaded {resident} B in {:.1} s (runtime value)",
            card.name,
            wall.as_secs_f64()
        );
        let one = check_resident(plan, c, &w);
        let mem = Memory {
            total: on.total,
            free_ctx: on.free_ctx,
            free_load: free_load as u64,
            resident,
            granularity: on.gpu.allocation_granularity()? as u64,
        };
        let two = check_memory(card, &plan.cards[c], &mem, &w);
        let three = check_readback(split, plan, c, &on.gpu, &w)?;
        Ok(one && two && three)
    }

    /// Check 1: each segment on card `c` holds the plan's buffers for it,
    /// nothing else is resident, and the total is the card's dense + expert
    /// bytes.
    fn check_resident(plan: &Plan<'_>, c: usize, w: &Weights) -> bool {
        let name = &plan.machine.cards[c].name;
        let (mut segments, mut bad) = (0usize, Vec::new());
        for row in &plan.rows {
            let t = &plan.model.tensors[row.tensor];
            for seg in row.segments.iter().filter(|s| s.device == Device::Card(c)) {
                segments += 1;
                let want = seg.buffer_bytes(t, plan.model.experts);
                match (w.get(&t.name).map(buffers), want) {
                    (Some(got), Ok(want)) if got == want => {}
                    (Some(got), want) => bad.push(format!(
                        "{}: buffers {got:?} B resident, the plan's segment {want:?}",
                        t.name
                    )),
                    (None, _) => bad.push(format!("{}: not resident", t.name)),
                }
            }
        }
        let resident_names = w.names().count();
        if resident_names != segments {
            bad.push(format!(
                "{resident_names} tensors resident, the plan has {segments} segments here"
            ));
        }
        let t = &plan.cards[c];
        let (got, want) = (w.resident_bytes() as u64, t.dense_bytes + t.expert_bytes);
        if got != want {
            bad.push(format!(
                "total {got} B, the plan's dense + experts {want} B"
            ));
        }
        println!(
            "check 1 {name}: {segments} segments, {got} B resident, plan dense + experts {want} B: {}",
            verdict(bad.is_empty())
        );
        for b in &bad {
            eprintln!("FAIL: check 1 {name}: {b}");
        }
        bad.is_empty()
    }

    /// One card's device memory around its load.
    struct Memory {
        total: u64,
        free_ctx: u64,
        free_load: u64,
        resident: u64,
        granularity: u64,
    }

    /// Check 2: what the load leaves free covers the plan's headroom, KV and
    /// scratch, and what the load took beyond its resident bytes is the plan's
    /// rounding term exactly. The other costs are printed, never asserted: the
    /// plan's usable bytes are the card's free bytes with no process on it, so
    /// the context took usable − free after it.
    fn check_memory(card: &Card, t: &CardTotals, m: &Memory, w: &Weights) -> bool {
        let floor = t.headroom_bytes + i128::from(t.kv_bytes) + i128::from(t.scratch_bytes);
        let pass = i128::from(m.free_load) >= floor;
        println!(
            "check 2 {}: {} B free after the load, the plan's headroom + KV + scratch {floor} B: {}",
            card.name,
            m.free_load,
            verdict(pass)
        );
        let (usable, free_ctx, free_load, resident) = (
            i128::from(card.usable_bytes),
            i128::from(m.free_ctx),
            i128::from(m.free_load),
            i128::from(m.resident),
        );
        let took = free_ctx - free_load;
        let gap = took - resident;
        let residue = gap - i128::from(t.rounding_bytes);
        let exact = residue == 0;
        println!(
            "check 2 {} rounding: the load asked for {resident} B resident and cuMemGetInfo lost {took} B \
             across it, {gap} B beyond the request; the plan's rounding term {} B [derived], residue \
             {residue} B: {}",
            card.name,
            t.rounding_bytes,
            verdict(exact)
        );
        let context = usable - free_ctx;
        println!(
            "  the context took {context} B (usable {usable} − free after it); with the residue {} B \
             against the plan's context {} B",
            context + residue,
            t.context_bytes
        );
        let total = i128::from(m.total);
        let overhead = workstation::spec_of(card).map_or("?".to_string(), |s| {
            (total - free_load - resident - i128::from(s.driver_reserve_bytes)).to_string()
        });
        println!(
            "  cuMemGetInfo total {total} B; total − free − resident − driver reserve = {overhead} B"
        );
        let allocations: usize = w
            .names()
            .filter_map(|n| w.get(n))
            .map(|dw| buffers(dw).len())
            .sum();
        println!(
            "  {allocations} allocations; granule {} B in the plan, allocation granularity {} B on \
             the device",
            card.granule_bytes, m.granularity
        );
        pass && exact
    }

    /// The device buffers a resident weight holds, in bytes: measured from
    /// the buffers themselves (`DevWeight::buffer_bytes`), never recomputed
    /// from the format, so check 1 compares what was allocated with the
    /// plan's arithmetic rather than that arithmetic with itself.
    fn buffers(dw: &DevWeight) -> Vec<u64> {
        dw.buffer_bytes().into_iter().map(|b| b as u64).collect()
    }

    /// A read-back sample: a plan tensor, its card format and its segment on
    /// the card.
    struct Sample<'p> {
        t: &'p ModelTensor,
        format: CardFormat,
        seg: &'p Segment,
    }

    impl Sample<'_> {
        fn experts(&self) -> Option<&ExpertList> {
            self.seg.experts.as_ref()
        }
    }

    /// Card `c`'s samples: the first whole tensor of each card format, in plan
    /// order, and the first expert stack segment.
    fn samples<'p>(plan: &'p Plan<'p>, c: usize) -> Vec<Sample<'p>> {
        let model: &'p ModelTensors = plan.model;
        let mut out: Vec<Sample<'p>> = Vec::new();
        for row in &plan.rows {
            let t = &model.tensors[row.tensor];
            for seg in row.segments.iter().filter(|s| s.device == Device::Card(c)) {
                let Format::Card(format) = seg.format else {
                    continue;
                };
                // One expert segment of any format; one whole tensor per format.
                let taken = out.iter().any(|s| match (s.experts(), &seg.experts) {
                    (Some(_), Some(_)) => true,
                    (None, None) => s.format == format,
                    _ => false,
                });
                if !taken {
                    out.push(Sample { t, format, seg });
                }
            }
        }
        out
    }

    /// Resident planes, as the device holds them or as they are packed here.
    enum Planes {
        Words(Vec<u32>),
        F32(Vec<f32>),
        Q8(Vec<u32>, Vec<u16>),
    }

    /// The planes `format` makes of `bytes`, a run of whole rows of the file,
    /// packed from the file formats' definitions: a K-quant row stream as
    /// little-endian words; a q8_0 block (f16 scale, 32 codes) as its scale's
    /// f16 bits and its codes four to a little-endian word; f32 as is; bf16 as
    /// the high half of an f32.
    fn host_planes(format: CardFormat, bytes: &[u8]) -> Result<Planes, GateError> {
        Ok(match format {
            CardFormat::KQuant => Planes::Words(bytes_to_words(bytes)),
            CardFormat::Q8_0Planes => {
                let blocks = bytes.as_chunks::<34>().0;
                let mut qs = Vec::with_capacity(blocks.len() * 8);
                let mut d = Vec::with_capacity(blocks.len());
                for b in blocks {
                    d.push(u16::from_le_bytes([b[0], b[1]]));
                    qs.extend(
                        b[2..]
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|w| u32::from_le_bytes(*w)),
                    );
                }
                Planes::Q8(qs, d)
            }
            CardFormat::F32 => Planes::F32(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect(),
            ),
            CardFormat::Bf16AsF32 => Planes::F32(
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16))
                    .collect(),
            ),
            CardFormat::Q5_0 | CardFormat::Q5_1 | CardFormat::Bf16Raw => {
                return Err(format!("no host packing of {format:?} here: V4.1 has none").into());
            }
        })
    }

    /// The planes of a resident weight, read back.
    fn device_planes(dw: &DevWeight, stream: &CudaStream) -> Result<Planes, GateError> {
        Ok(match dw {
            DevWeight::KQuant { w, .. } | DevWeight::Q5_0 { w, .. } | DevWeight::Q5_1 { w, .. } => {
                Planes::Words(w.buf().to_host_vec(stream)?)
            }
            DevWeight::Q8_0 { qs, d, .. } => {
                Planes::Q8(qs.buf().to_host_vec(stream)?, d.buf().to_host_vec(stream)?)
            }
            DevWeight::F32 { w, .. } => Planes::F32(w.buf().to_host_vec(stream)?),
            DevWeight::Q8_0Derived { .. } => {
                return Err("a derived weight under a file tensor's name".into());
            }
        })
    }

    /// `None` when `got` is `want` bit for bit, else what differs.
    fn difference(got: &Planes, want: &Planes) -> Option<String> {
        let first = |a: &[u32], b: &[u32]| {
            a.iter()
                .zip(b)
                .position(|(x, y)| x != y)
                .unwrap_or(a.len().min(b.len()))
        };
        match (got, want) {
            (Planes::Words(g), Planes::Words(w)) => (g != w).then(|| {
                format!(
                    "{} vs {} words, first difference at {}",
                    g.len(),
                    w.len(),
                    first(g, w)
                )
            }),
            (Planes::F32(g), Planes::F32(w)) => {
                (!bits_equal(g, w)).then(|| format!("{} vs {} f32 not bit-equal", g.len(), w.len()))
            }
            (Planes::Q8(gq, gd), Planes::Q8(wq, wd)) => (gq != wq || gd != wd)
                .then(|| format!("codes {} scales {}", verdict(gq == wq), verdict(gd == wd))),
            _ => Some("the resident variant is not the format's".to_string()),
        }
    }

    /// Check 3: each sample of card `c` reads back as the host packing of the
    /// file bytes its segment holds.
    fn check_readback(
        split: &Split,
        plan: &Plan<'_>,
        c: usize,
        gpu: &Gpu,
        w: &Weights,
    ) -> Result<bool, GateError> {
        let name = &plan.machine.cards[c].name;
        gpu.context().bind_to_thread()?;
        let mut pass = true;
        for s in samples(plan, c) {
            let (shard, info) = split
                .find(&s.t.name)
                .ok_or_else(|| format!("{} is not in the split", s.t.name))?;
            let bytes = split.shard(shard).ok_or("shard out of range")?.data(info)?;
            let rows: u64 = info.dims[1..].iter().product();
            let row_bytes = info.nbytes / rows;
            // The segment's rows in slot order, from the list's ids alone.
            let held_ranges: Vec<(u64, u64)> = match s.experts() {
                None => vec![(0, rows)],
                Some(e) => {
                    let per = rows / plan.model.experts;
                    e.ids()
                        .iter()
                        .map(|&id| (u64::from(id) * per, (u64::from(id) + 1) * per))
                        .collect()
                }
            };
            let held_rows: u64 = held_ranges.iter().map(|&(a, b)| b - a).sum();
            let mut held = Vec::new();
            for &(ra, rb) in &held_ranges {
                let (a, b) = (
                    usize::try_from(ra * row_bytes)?,
                    usize::try_from(rb * row_bytes)?,
                );
                let run = bytes
                    .get(a..b)
                    .ok_or_else(|| format!("{} holds fewer than {b} bytes", s.t.name))?;
                held.extend_from_slice(run);
            }
            let len = held.len();
            let want = host_planes(s.format, &held)?;
            let dw = w
                .get(&s.t.name)
                .ok_or_else(|| format!("{} is not resident", s.t.name))?;
            let diff = difference(&device_planes(dw, gpu.stream())?, &want);
            // One run prints as itself; a hot list's scattered ids print as
            // counts, not as every range.
            let experts = s.experts().map_or(String::new(), |e| match e.runs().len() {
                1 => format!(" experts {e}"),
                n => format!(" experts {} in {n} ranges", e.len()),
            });
            println!(
                "check 3 {name}: {} {}{experts}, {held_rows} rows, {len} file bytes: {}",
                s.t.name,
                Format::Card(s.format),
                verdict(diff.is_none())
            );
            if let Some(d) = diff {
                eprintln!("FAIL: check 3 {name}: {}: {d}", s.t.name);
                pass = false;
            }
        }
        Ok(pass)
    }

    /// Per shard, the pages of the page-rounded union of the host segments
    /// `lock` keeps, from the headers alone.
    fn page_union(
        split: &Split,
        plan: &Plan<'_>,
        lock: &Lock,
        page: u64,
    ) -> Result<BTreeMap<usize, u64>, GateError> {
        let mut spans: BTreeMap<usize, Vec<(u64, u64)>> = BTreeMap::new();
        for row in &plan.rows {
            let t = &plan.model.tensors[row.tensor];
            if !lock.keeps(t) {
                continue;
            }
            for seg in &row.segments {
                if seg.device != Device::Host || seg.format != Format::HostFile {
                    continue;
                }
                let (shard, info) = split
                    .find(&t.name)
                    .ok_or_else(|| format!("{} is not in the split", t.name))?;
                let base =
                    split.shard(shard).ok_or("shard out of range")?.data_base() + info.offset;
                let per = info.nbytes / plan.model.experts;
                let held: Vec<(u64, u64)> = match &seg.experts {
                    None => vec![(0, info.nbytes)],
                    Some(e) => e
                        .ids()
                        .iter()
                        .map(|&id| (u64::from(id) * per, (u64::from(id) + 1) * per))
                        .collect(),
                };
                let entry = spans.entry(shard).or_default();
                for (a, b) in held {
                    entry.push(((base + a) / page, (base + b).div_ceil(page)));
                }
            }
        }
        let mut pages = BTreeMap::new();
        for (shard, mut s) in spans {
            s.sort_unstable();
            let (mut n, mut end) = (0, 0);
            for (a, b) in s {
                let a = a.max(end);
                if b > a {
                    n += b - a;
                    end = b;
                }
            }
            pages.insert(shard, n);
        }
        Ok(pages)
    }

    /// `VmLck` of this process, in bytes.
    fn vm_lck() -> Result<u64, GateError> {
        let status = std::fs::read_to_string("/proc/self/status")?;
        let kb = status
            .lines()
            .find_map(|l| l.strip_prefix("VmLck:"))
            .and_then(|v| v.trim().strip_suffix("kB"))
            .ok_or("/proc/self/status has no VmLck line in kB")?;
        Ok(kb.trim().parse::<u64>()? * 1024)
    }

    /// `Cached` of `/proc/meminfo`, in bytes.
    fn meminfo_cached() -> Result<u64, GateError> {
        let info = std::fs::read_to_string("/proc/meminfo")?;
        let kb = info
            .lines()
            .find_map(|l| l.strip_prefix("Cached:"))
            .and_then(|v| v.trim().strip_suffix("kB"))
            .ok_or("/proc/meminfo has no Cached line in kB")?;
        Ok(kb.trim().parse::<u64>()? * 1024)
    }

    /// Check 4's precondition: every shard file out of the page cache, so the
    /// host set is resident afterwards only if the load read it in. Pages
    /// another process maps stay; the count left is printed.
    fn evict(split: &Split, set: &HostSet) -> Result<(), GateError> {
        let mut release = PageDrop::new(split);
        for s in 0..split.shard_count() {
            let len = split.shard(s).ok_or("shard out of range")?.mapping().len() as u64;
            release.release(s, 0..len)?;
        }
        let (resident, pages) = set.resident(split)?;
        println!(
            "evict: the shard files' {} B advised out of the page cache; host set {resident} of \
             {pages} pages still resident before the load (pages another process maps stay)",
            release.bytes()
        );
        Ok(())
    }

    /// The host half: the engine's host step over the host segments `lock`
    /// keeps (every one without it), then check 4 — `mincore` finds the
    /// whole set resident — and, when the step locked, `VmLck` against the
    /// derived page union. `cached` is `Cached` before the load and after the
    /// cards.
    fn check_host(
        split: &Split,
        plan: &Plan<'_>,
        lock: Option<&Lock>,
        levers: HostLevers,
        cached: [u64; 2],
    ) -> Result<bool, GateError> {
        let page = page_bytes();
        let before = vm_lck()?;
        let host = HostResidency::at_load(split, plan, |t| keeps(lock, t), levers)?;
        let after = vm_lck()?;
        let cached_host = meminfo_cached()?;
        let set = host.set();
        match host.populated() {
            Some(w) => println!(
                "host_populate={} in {:.1} s over {} shards (runtime value)",
                w.bytes(),
                w.wall().as_secs_f64(),
                w.shards().len()
            ),
            None => println!("host_populate=off"),
        }
        let delta = |a: u64, b: u64| i128::from(b) - i128::from(a);
        println!(
            "meminfo Cached: {} B before the load, {} B after the cards ({:+} B, card_dontneed={}), \
             {} B after the host step ({:+} B; the host set is {} B)",
            cached[0],
            cached[1],
            delta(cached[0], cached[1]),
            u8::from(levers.card_dontneed),
            cached_host,
            delta(cached[1], cached_host),
            set.bytes()
        );
        let (resident, pages) = set.resident(split)?;
        let four = resident == pages;
        println!(
            "check 4 host: {resident} of {pages} host-set pages resident after the host step \
             ({} B): {}",
            pages * page,
            verdict(four)
        );
        if !four {
            eprintln!(
                "FAIL: check 4 host: {} host-set pages not resident after the load",
                pages - resident
            );
        }
        let Some(held) = host.lock() else {
            return Ok(four);
        };
        let union = page_union(split, plan, lock.unwrap_or(&Lock::All), page)?;
        let union_bytes = union.values().sum::<u64>() * page;
        for s in held.shards() {
            println!(
                "lock shard {}: {} spans, {} B in {:.1} s (runtime value); derived union {} B",
                s.shard,
                s.spans,
                s.bytes,
                s.wall.as_secs_f64(),
                union.get(&s.shard).map_or(0, |p| p * page)
            );
        }
        println!(
            "host_lock={} vm_lck={after} over {} shards in {:.1} s (runtime value)",
            held.bytes(),
            held.shards().len(),
            held.wall().as_secs_f64()
        );
        let grew = after.checked_sub(before);
        let vm_ok = grew == Some(union_bytes) && held.bytes() == union_bytes;
        println!(
            "host VmLck: {before} B before, {after} B after the lock; the derived page-rounded union \
             {union_bytes} B, the host set {} B: {}",
            set.bytes(),
            verdict(vm_ok)
        );
        Ok(four && vm_ok)
    }
}
