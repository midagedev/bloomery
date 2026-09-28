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
//! it with the source's sum under another rule, so the pass's value depends
//! on which experts the card holds, as the engine's bits do. A pass's value
//! is the FNV-1a of every routed id's (id, side, parts); a card sum that is
//! not its expert's is `stale`, an id both sides serve `double`, one neither
//! serves `miss`.
//!
//! The trace: 120 passes of the synthetic router (a hot set of 8 drifting
//! every 20 passes, never the seed's first 4), every fifth pass two rows of
//! which the first is kept. The rule: every 2, cap 6, margin 1, min count 2,
//! decay 0.9, one spare, live delay 3, 4 pinned seed experts a layer.
//!
//! Clauses (adaptres §4; c6 needs the pair pass and is R3's):
//! Each pass runs in the engine's order: the ids refreshed by a
//! synchronizing copy, the boundary, the graph, then the engine stream
//! drained by polling with a bound (a stream waiting on a copy nothing
//! stages is a named failure, not a hang) and read back.
//!
//! - c1: the same history twice with the copies prompt, and once with the
//!   copy stream held (c3's run), gives the same pass values and the same
//!   flips, and the run makes flips on every card layer.
//! - c2: the adaptive run equals, pass for pass, a static replay that puts
//!   each pass's card sets in place with synchronous copies into slots by
//!   ascending id (other slot numbers than the machine's) and a fresh map.
//! - c3: with the staging window closed for the whole run, a copy is staged
//!   only once its flip is due (its live boundary reached), so every landing
//!   is late and the engine stream waits for it; no card sum is stale, no id
//!   served twice or not at all.
//! - c4: (its mutant: the card's map word written when the flip is made) is
//!   red in c3's run.
//! - c5: every pass of every run has no double and no miss (its mutant: the
//!   host map one pass late).
//! - c7: a reset brings every layer's card set back to its seed (`residency
//!   reset` diff=0, the live sets and the slot ledger's), and the trace after
//!   it gives the fresh run's values.
//! - refusal: a flip whose victim the source cannot bring to the host is
//!   refused by name at the boundary it would land at; a load whose spare
//!   slot gives up such an expert is refused at construction.
//! - pinned: no flip evicts a pinned seed expert and each keeps its slot for
//!   the whole run, while flips evict other seed experts.
//! - stall: a victim that takes longer to prepare than the machine's
//!   deadline is a named error at the boundary it would land at, returned
//!   before the preparation ends.

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
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::time::{Duration, Instant};

    use bloomery_gpu::host::slots::{HOST, Slot, SlotMap};
    use bloomery_gpu::host::swap::{
        MachineCfg, PassReport, Piece, SlotState, SwapMachine, SwapSource, Transform,
    };
    use bloomery_gpu::{Gpu, GpuError, Graph};
    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use cuda_core::{DeviceBuffer, LaunchConfig1D, sys};
    use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
    use cuda_host::cuda_module;
    use runtime::swaprule::{Flip, SwapParams};

    const LAYERS: Range<usize> = 2..6;
    const L: usize = LAYERS.end - LAYERS.start;
    const E: usize = 32;
    const K: usize = 4;
    const MAX_ROWS: usize = 2;
    const NIDS: usize = MAX_ROWS * K;
    const N_L: [usize; L] = [12, 12, 10, 0];
    const PART_BYTES: [usize; 3] = [1024, 1024, 1536];
    const PARTS: usize = PART_BYTES.len();
    const PINNED: usize = 4;
    const PASSES: usize = 120;
    /// The machine's bound on every host wait in the arms that must pass.
    const DEADLINE: Duration = Duration::from_secs(10);
    /// The stall arm's bound, and how long its slow victim takes to prepare.
    const STALL_DEADLINE: Duration = Duration::from_millis(300);
    const SLOW: Duration = Duration::from_secs(2);
    /// The gate's own bound on a pass's engine stream.
    const ENGINE_DRAIN: Duration = Duration::from_secs(30);
    const DELAY: u64 = 3;
    /// The host's rule: its sum is the card's under this mask, so a pass's
    /// value says which side served each id.
    const HOST_MASK: u32 = 0x1234_5678;
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

    /// The synthetic source: every expert's bytes on the host; the stacks'
    /// addresses; residency per expert, the card's experts not resident at
    /// load (their pages dropped), preparing makes an expert resident unless
    /// it is in `stuck`.
    struct Synth {
        bytes: Vec<Vec<u8>>,
        bases: Vec<[sys::CUdeviceptr; PARTS]>,
        caps: [usize; L],
        resident: Vec<AtomicBool>,
        stuck: Vec<(usize, u32)>,
        slow: Vec<(usize, u32)>,
        prepared: AtomicU32,
    }

    impl Synth {
        fn new(
            stacks: &Stacks,
            caps: [usize; L],
            stuck: Vec<(usize, u32)>,
            slow: Vec<(usize, u32)>,
        ) -> Synth {
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
                .map(|x| AtomicBool::new(x % E >= caps[x / E]))
                .collect();
            Synth {
                bytes,
                bases: stacks.bases(),
                caps,
                resident,
                stuck,
                slow,
                prepared: AtomicU32::new(0),
            }
        }
    }

    impl SwapSource for Synth {
        fn part_bytes(&self) -> &[usize] {
            &PART_BYTES
        }

        fn source(&self, layer: usize, id: u32, part: usize) -> Result<Piece<'_>, GpuError> {
            let at = (li(layer) * E + id as usize) * PARTS + part;
            Ok(Piece {
                bytes: &self.bytes[at],
                transform: Transform::Identity,
            })
        }

        fn dest(&self, layer: usize, part: usize, slot: u32) -> Result<sys::CUdeviceptr, GpuError> {
            if slot as usize >= self.caps[li(layer)].max(1) {
                return Err(GpuError::Shape {
                    what: "Synth::dest",
                    detail: format!("layer {layer} slot {slot}"),
                });
            }
            Ok(self.bases[li(layer)][part] + (slot as usize * PART_BYTES[part]) as u64)
        }

        fn prepare_victim(&self, layer: usize, id: u32) -> Result<(), GpuError> {
            self.prepared.fetch_add(1, Ordering::Relaxed);
            if self.slow.contains(&(layer, id)) {
                std::thread::sleep(SLOW);
            }
            if !self.stuck.contains(&(layer, id)) {
                self.resident[li(layer) * E + id as usize].store(true, Ordering::Release);
            }
            Ok(())
        }

        fn host_resident(&self, layer: usize, id: u32) -> Result<bool, GpuError> {
            Ok(self.resident[li(layer) * E + id as usize].load(Ordering::Acquire))
        }

        fn release_host(&self, layer: usize, id: u32) -> Result<u64, GpuError> {
            self.resident[li(layer) * E + id as usize].store(false, Ordering::Release);
            Ok(PART_BYTES.iter().sum::<usize>() as u64)
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

    /// The seed: experts `0..n_l` of every layer in slots `0..n_l`.
    fn seed_map() -> Result<SlotMap, GateError> {
        let mut rows = vec![HOST; L * E];
        for (i, &n) in N_L.iter().enumerate() {
            for e in 0..n {
                rows[i * E + e] = e as u32;
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

    fn params() -> SwapParams {
        SwapParams {
            every: 2,
            cap: 6,
            margin: 1.0,
            min_count: 2.0,
            decay: 0.9,
            spares: 1,
            delay: DELAY,
        }
    }

    /// A card with the probe's graph over its own buffers.
    struct Card {
        stacks: Stacks,
        view: DeviceBuffer<u32>,
        ids: DeviceBuffer<u32>,
        out: DeviceBuffer<u32>,
        graph: Option<Graph>,
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
                stacks,
                view: DeviceBuffer::from_host(s, &map.stage_view())?,
                ids: DeviceBuffer::<u32>::zeroed(s, MAX_ROWS * L * K)?,
                out: DeviceBuffer::<u32>::zeroed(s, L * PARTS * NIDS)?,
                graph: None,
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
                    let on_host = slots.slot(l, id) == Some(Slot::Host);
                    mix(id);
                    match (on_card, on_host) {
                        (true, true) => double += 1,
                        (false, false) => miss += 1,
                        _ => {}
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

    /// What an adaptive run saw: per pass its value, its report and its
    /// card sets; the flips made, with the boundary; and the machine, the
    /// map and the card, for the arms that go on.
    struct Run {
        values: Vec<Value>,
        reports: Vec<PassReport>,
        sets: Vec<Vec<Vec<u32>>>,
        flips: Vec<(u64, Flip)>,
        slot_moves: usize,
        pinned_kept: bool,
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

    fn cfg(deadline: Duration) -> MachineCfg {
        MachineCfg {
            params: params(),
            pinned: N_L
                .iter()
                .map(|&n| if n > 0 { PINNED } else { 0 })
                .collect(),
            top_k: K,
            max_rows: MAX_ROWS,
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

    /// Drive `passes` of the trace through a machine, in the engine's order:
    /// per pass the ids refreshed (a synchronizing copy), the boundary, the
    /// graph, a bounded drain and the readback, the value, then the ids
    /// noted and the pass's end with its kept rows. Held, the staging
    /// window stays closed for the whole run, so every copy is staged only
    /// once its flip is due.
    fn drive(
        gpu: &Gpu,
        run: &mut Run,
        trace: &Trace,
        passes: Range<usize>,
        copies: Copies,
    ) -> Result<(), GateError> {
        let stream = gpu.stream();
        let m = run.machine.as_mut().ok_or("gate_swap: no machine")?;
        let window = m.window();
        let mut tally = m.tally();
        if copies == Copies::Held {
            window.store(0, Ordering::Release);
        }
        let pinned_slots: Vec<Vec<(u32, Option<Slot>)>> = LAYERS
            .map(|l| {
                (0..m.pinned(l))
                    .map(|r| {
                        let e = m.seed(l)[r];
                        (e, run.slots.slot(l, e))
                    })
                    .collect()
            })
            .collect();
        for p in passes {
            let (rows, kept) = &trace.passes[p];
            run.card.refresh(gpu, rows)?;
            let report = match m.boundary(stream, &mut run.slots) {
                Ok(r) => r,
                Err(e) => {
                    run.err = Some(e.to_string());
                    window.store(1, Ordering::Release);
                    return Ok(());
                }
            };
            for f in m.rule().in_flight() {
                if f.live_at == report.boundary + DELAY {
                    run.flips.push((report.boundary, *f));
                }
            }
            run.slot_moves += report.landed;
            run.card.launch(gpu)?;
            if let Err(e) = drain(gpu) {
                run.err = Some(format!("pass {p}: {e}"));
                window.store(1, Ordering::Release);
                return Ok(());
            }
            let out = run.card.read(gpu)?;
            run.values.push(value(rows, &out, &run.slots));
            run.reports.push(report);
            run.sets.push(card_sets(&run.slots));
            for (r, row) in rows.iter().enumerate() {
                for (i, ids) in row.iter().enumerate() {
                    for (k, &id) in ids.iter().enumerate() {
                        tally.note(LAYERS.start + i, r, k, id);
                    }
                }
            }
            m.end_pass(&mut tally, *kept)?;
        }
        window.store(1, Ordering::Release);
        run.pinned_kept = LAYERS.zip(&pinned_slots).all(|(l, pins)| {
            pins.iter()
                .all(|&(e, slot)| slot.is_some() && run.slots.slot(l, e) == slot)
        });
        Ok(())
    }

    /// A fresh arm: its card, its captured probe and a machine over a source
    /// whose `stuck` experts never become host-resident and whose `slow`
    /// ones take `SLOW` to prepare, every host wait bounded by `deadline`.
    fn start(
        gpu: &Gpu,
        pm: &probe_kernels::LoadedModule,
        stuck: Vec<(usize, u32)>,
        slow: Vec<(usize, u32)>,
        deadline: Duration,
    ) -> Result<Run, GateError> {
        let mut slots = seed_map()?;
        let mut card = Card::new(gpu, &slots)?;
        card.capture(gpu, pm)?;
        let source = Arc::new(Synth::new(&card.stacks, N_L, stuck, slow));
        let machine = SwapMachine::new(
            gpu.context(),
            gpu.stream(),
            &mut slots,
            &card.view,
            source,
            cfg(deadline),
        );
        let (machine, err) = match machine {
            Ok(m) => (Some(m), None),
            Err(e) => (None, Some(e.to_string())),
        };
        Ok(Run {
            values: Vec::new(),
            reports: Vec::new(),
            sets: Vec::new(),
            flips: Vec::new(),
            slot_moves: 0,
            pinned_kept: false,
            err,
            machine,
            slots,
            card,
        })
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

    /// The static replay: per pass, the adaptive run's card sets placed by
    /// synchronous copies, each layer's set in slots by ascending id, and a
    /// fresh map from those rows.
    fn static_replay(
        gpu: &Gpu,
        pm: &probe_kernels::LoadedModule,
        trace: &Trace,
        sets: &[Vec<Vec<u32>>],
    ) -> Result<Vec<Value>, GateError> {
        let mut slots = seed_map()?;
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

    fn show(e: &Option<String>) -> String {
        e.as_ref()
            .map_or_else(|| "none".to_string(), ToString::to_string)
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        // SAFETY: the module is this binary's own, loaded once into the
        // card's context before any launch.
        let pm = unsafe { probe_kernels::load(gpu.context())? };
        let trace = trace();
        let mut ok = true;
        println!(
            "gate_swap: layers {LAYERS:?} of {E} experts, top-{K}, card slots {N_L:?}, parts \
             {PART_BYTES:?} B, {PASSES} passes, rule every 2 cap 6 margin 1 min 2 decay 0.9, one \
             spare, delay {DELAY}, {PINNED} pinned"
        );

        // c1: the same history, twice prompt and once held.
        let mut a = start(&gpu, &pm, Vec::new(), Vec::new(), DEADLINE)?;
        drive(&gpu, &mut a, &trace, 0..PASSES, Copies::Prompt)?;
        let mut b = start(&gpu, &pm, Vec::new(), Vec::new(), DEADLINE)?;
        drive(&gpu, &mut b, &trace, 0..PASSES, Copies::Prompt)?;
        let mut h = start(&gpu, &pm, Vec::new(), Vec::new(), DEADLINE)?;
        drive(&gpu, &mut h, &trace, 0..PASSES, Copies::Held)?;
        let per_layer: Vec<usize> = LAYERS
            .map(|l| {
                a.flips
                    .iter()
                    .filter(|(_, f)| f.layer + LAYERS.start == l)
                    .count()
            })
            .collect();
        let flips_everywhere = per_layer
            .iter()
            .zip(N_L)
            .all(|(&n, cap)| (cap == 0) == (n == 0));
        let c1 = a.err.is_none()
            && b.err.is_none()
            && h.err.is_none()
            && a.values.len() == PASSES
            && fnvs(&a.values) == fnvs(&b.values)
            && fnvs(&a.values) == fnvs(&h.values)
            && a.flips == b.flips
            && a.flips == h.flips
            && flips_everywhere;
        ok &= c1;
        println!(
            "c1 same history: {} passes, {} flips (per layer {per_layer:?}), landed {}; prompt twice \
             equal {} flips equal {}; held equal {} flips equal {}; errors {} / {} / {} {}",
            a.values.len(),
            a.flips.len(),
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

        // c2: the static replay of the adaptive run's card sets.
        let reference = static_replay(&gpu, &pm, &trace, &a.sets)?;
        let first_diff = fnvs(&a.values)
            .iter()
            .zip(fnvs(&reference))
            .position(|(x, y)| *x != y);
        let c2 = reference.len() == a.values.len() && first_diff.is_none() && clean(&reference);
        ok &= c2;
        println!(
            "c2 static replay (slots by ascending id, synchronous copies): {} passes, first \
             differing pass {first_diff:?}, reference errors (stale, double, miss) {:?} {}",
            reference.len(),
            errs(&reference),
            verdict(c2)
        );

        // c3 (and c4's harness): with the window closed every copy is staged
        // only once due, so every landing is late, and each is read right.
        let late: usize = h.reports.iter().map(|r| r.late).sum();
        let c3 = h.err.is_none()
            && late > 0
            && late == h.slot_moves
            && clean(&h.values)
            && fnvs(&h.values) == fnvs(&a.values);
        ok &= c3;
        println!(
            "c3 held copies (window closed all run): late landings {late} of {} (the engine stream \
             waited), errors (stale, double, miss) {:?}, values = the prompt run's {}, prompt run's \
             late landings {} {}",
            h.slot_moves,
            errs(&h.values),
            fnvs(&h.values) == fnvs(&a.values),
            a.reports.iter().map(|r| r.late).sum::<usize>(),
            verdict(c3)
        );

        // c5: no id served twice or not at all, in every run.
        let c5 = [&a, &b, &h].iter().all(|r| clean(&r.values));
        ok &= c5;
        println!(
            "c5 host map and card map agree every pass: errors (stale, double, miss) prompt {:?} \
             prompt {:?} held {:?} {}",
            errs(&a.values),
            errs(&b.values),
            errs(&h.values),
            verdict(c5)
        );

        // pinned: never a victim, never moved, while other seed experts leave.
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
        ok &= pinned;
        println!(
            "pinned: flips evicting a pinned seed expert {pinned_victim}, every pinned expert in its \
             load slot throughout {}, flips evicting an unpinned seed expert {seed_victims} {}",
            a.pinned_kept,
            verdict(pinned)
        );
        let last = a.reports.last().copied().unwrap_or_default();
        record::residency_pass(&last).print();

        // c7: reset back to the seed, then the trace again from the seed.
        let reset = {
            let m = a.machine.as_mut().ok_or("gate_swap: no machine")?;
            m.reset(gpu.stream(), &mut a.slots)
        };
        let c7 = match &reset {
            Ok(r) => {
                record::residency_reset(r).print();
                let m = a.machine.as_ref().ok_or("gate_swap: no machine")?;
                let sets_seed = LAYERS.enumerate().all(|(i, l)| {
                    let mut seed = m.seed(l).to_vec();
                    seed.sort_unstable();
                    card_sets(&a.slots)[i] == seed
                        && m.rule()
                            .live(i)
                            .is_ok_and(|live| live.collect::<Vec<_>>() == seed)
                });
                let ledger_seed = LAYERS.into_iter().all(|l| {
                    let row = m.ledger().row(l);
                    let live = row
                        .iter()
                        .filter(|s| matches!(s, SlotState::Live(_)))
                        .count();
                    let spare = row.iter().filter(|s| **s == SlotState::Spare).count();
                    live == m.seed(l).len() && spare == row.len() - live
                });
                let fresh = fnvs(&a.values[..40]);
                a.values.clear();
                a.flips.clear();
                drive(&gpu, &mut a, &trace, 0..40, Copies::Prompt)?;
                let again = fnvs(&a.values) == fresh && a.err.is_none();
                println!(
                    "c7 reset: diff {} cancelled {} copies {} dropped_bytes {}; card sets = the seed \
                     {sets_seed}, slot ledger = the seed's {ledger_seed}; 40 passes after it = the \
                     fresh run's {again}",
                    r.diff, r.cancelled, r.copies, r.dropped_bytes
                );
                r.diff == 0 && r.copies > 0 && sets_seed && ledger_seed && again
            }
            Err(e) => {
                println!("c7 reset: {e}");
                false
            }
        };
        ok &= c7;
        println!("c7 {}", verdict(c7));

        // refusal: a victim the source cannot bring to the host.
        let stuck: Vec<(usize, u32)> = (PINNED as u32..(N_L[0] - 1) as u32)
            .map(|e| (2, e))
            .collect();
        let mut r = start(&gpu, &pm, stuck, Vec::new(), DEADLINE)?;
        let built = r.err.is_none();
        drive(&gpu, &mut r, &trace, 0..PASSES, Copies::Prompt)?;
        let said = show(&r.err);
        let named = said.contains("is not host-resident") && said.contains("layer 2 expert");
        let tail = vec![(3usize, (N_L[1] - 1) as u32)];
        let at_load = start(&gpu, &pm, tail, Vec::new(), DEADLINE)?;
        let load_said = show(&at_load.err);
        let load_named = load_said.contains("is not host-resident")
            && load_said.contains("layer 3 expert 11")
            && at_load.machine.is_none();
        let refusal = built && named && load_named;
        ok &= refusal;
        println!(
            "refusal: after {} passes \"{said}\" {named}; a spare slot's expert at load \"{load_said}\" \
             {load_named} {}",
            r.values.len(),
            verdict(refusal)
        );

        // stall: a victim that takes longer to prepare than the machine's
        // deadline is a named error at the boundary it would land at, not a
        // wait without end.
        let slow = vec![(2usize, 7u32)];
        let mut st = start(&gpu, &pm, Vec::new(), slow, STALL_DEADLINE)?;
        let t0 = Instant::now();
        drive(&gpu, &mut st, &trace, 0..PASSES, Copies::Prompt)?;
        let took = t0.elapsed();
        let stall_said = show(&st.err);
        let stall = stall_said.contains("was not prepared for the host")
            && stall_said.contains("layer 2: the victim 7")
            && took < SLOW;
        ok &= stall;
        println!(
            "stall: a victim {SLOW:?} to prepare under a {STALL_DEADLINE:?} deadline: after {} passes \
             in {took:?} \"{stall_said}\" {}",
            st.values.len(),
            verdict(stall)
        );

        if ok {
            println!(
                "gate_swap: PASS — the same history gives the same values and flips whatever the \
                 copies' timing, equal to a static replay of its card sets; late copies are waited \
                 for; the host and card maps never serve an id twice or not at all; a reset returns \
                 to the seed; a non-resident victim is refused by name; a host wait past its \
                 deadline is a named error; pinned seed experts never move."
            );
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
