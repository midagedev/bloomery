//! Gate for allocator traffic: how often a steady-state decode step goes to malloc.
//!
//! No gate guards speed, but this is the one speed cause that reduces to an integer.
//! The main thread's steady-state profile had a quarter of its samples in
//! `memset` and the malloc family — every op returned a fresh zeroed `Vec` — and the
//! workers idle while the main thread does that. The count is over every thread, so a
//! worker closure that allocates per chunk shows up here too.
//!
//! `hw_` prefix: needs the box and the model file, not the oracle.
#[path = "common/model_path.rs"]
mod model_path;
#[allow(
    dead_code,
    reason = "this gate renders one layer; the sidecar and the salted writer serve the union gates"
)]
#[path = "common/r8layer.rs"]
mod r8layer;

use gguf::{GgmlType, Split};
use model::arch::deepseek2::derived::Derived;
use model::arch::deepseek2::forward::{new_cache, step};
use model::moe::{
    Claim, HostLayer, HostScratch, PinRelease, PinToken, Ticket, TicketWait, TierError, TierPin,
    TierSlots, UnionScratch,
};
use model::ops::{Lanes, ShardTensor, Tensor2};
use model::r8file::R8Source;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use threads::CcdMap;

/// The counters are the process's: one gate counts at a time.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

static CALLS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
/// The calling thread's share: the part that is serial, with every worker waiting on it.
static MAIN_CALLS: AtomicU64 = AtomicU64::new(0);

thread_local! {
    // const-initialized and destructor-free, so reading it inside the allocator cannot recurse.
    static IS_MAIN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct Counting;

fn count(size: usize) {
    CALLS.fetch_add(1, Relaxed);
    BYTES.fetch_add(size as u64, Relaxed);
    if IS_MAIN.with(|m| m.get()) {
        MAIN_CALLS.fetch_add(1, Relaxed);
    }
}

// SAFETY: every method forwards to `System` with the caller's arguments unchanged.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        count(l.size());
        // SAFETY: the caller meets `alloc`'s contract for `l` (non-zero size), and
        // `System` receives `l` unchanged.
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        count(l.size());
        // SAFETY: the caller meets `alloc_zeroed`'s contract for `l` (non-zero size),
        // and `System` receives `l` unchanged.
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
        count(new_size);
        // SAFETY: `p` came from this allocator with layout `l` — so from `System`, as every
        // allocating method forwards there — and the caller meets `realloc`'s contract
        // for `new_size`; `System` receives all three unchanged.
        unsafe { System.realloc(p, l, new_size) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: `p` came from this allocator with layout `l`, so from `System`, as every
        // allocating method forwards there.
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Allocator calls one single-token step may make, all threads. A ratchet, not a
/// target: it only moves down.
///
/// PIN(2026-09-20): 14454 calls / 24.5 MB per step before the `Tensor2` free list,
/// 11463 / 16.3 MB after (8952 of them on the calling thread); a step that grows the
/// KV blocks adds 28. What is left is per-call `Vec`s in `matmul_q_multi`, the
/// `TensorInfo` views built per expert and per head, and metadata keys formatted per
/// block — token-independent work that belongs in `Derived`.
/// PIN(2026-09-20): 11463 -> 1291 / 0.32 MB on the load-time plan round — every
/// tensor-name find, metadata read, norm-gain decode and per-head/per-expert view
/// moved into `Derived`, the quantization and flash scratch into per-thread
/// bucketed pools, and the MoE trace behind an opt-in. What is left: the routing
/// table's vectors and the per-batch bookkeeping (`ws`/`xs`/`bytes`/`outs`/`pairs`
/// of a `matmul_q_batch` call), plus the KV row a growing cache owns.
/// PIN(2026-09-21): 1291 -> 765 / 0.18 MB on the group dispatch round — the batch
/// bookkeeping (`ws`/`xs`/`outs`/`pairs`) moved to fixed-capacity stack arrays. A
/// step that grows the KV blocks adds 28.
/// PIN(2026-09-21): 765 -> 630 / 0.13 MB on the fused attention round — the
/// per-head `wv_b` gather (sixteen activation blocks and its batch bookkeeping)
/// left the step; worst warm step 658. A step that grows the KV blocks adds 28.
/// PIN(2026-09-21): 630 -> 603 / 0.13 MB on the contiguous-KV round — one flat
/// buffer per block replaces one `Vec` per cached row plus its outer `Vec`.
/// LIMIT 700 -> 660 on the same round: worst warm step measured 631 (the capacity-
/// doubling step, 864 KB); 660 keeps ~5 % headroom over that and the ratchet only
/// moves down.
const LIMIT: u64 = 660;

#[test]
#[ignore = "hw: needs the box and the model file"]
fn hw_steady_step_allocations_bounded() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let g = gguf::Gguf::open(model_path::model_path()).unwrap();
    let mut cache = new_cache(&g).unwrap();
    let derived = Derived::new(&g).unwrap();
    step(
        &g,
        &[100000, 549, 6077, 280, 7239, 317],
        &mut cache,
        &derived,
    )
    .unwrap();
    // Warm steps first: pools fill and the KV blocks grow on their own schedule.
    IS_MAIN.with(|m| m.set(true));
    let mut worst = 0u64;
    for i in 0..8 {
        let (c0, b0, m0) = (
            CALLS.load(Relaxed),
            BYTES.load(Relaxed),
            MAIN_CALLS.load(Relaxed),
        );
        step(&g, &[549], &mut cache, &derived).unwrap();
        let (c, b) = (CALLS.load(Relaxed) - c0, BYTES.load(Relaxed) - b0);
        let m = MAIN_CALLS.load(Relaxed) - m0;
        eprintln!(
            "step {i}: {c} allocator calls ({m} on the calling thread), {:.1} KB",
            b as f64 / 1024.0
        );
        if i >= 4 {
            worst = worst.max(c);
        }
    }
    assert!(
        worst <= LIMIT,
        "steady decode step made {worst} allocator calls, limit {LIMIT}"
    );
}

/// Allocator calls of the host tier's legs over a synthetic routed layer —
/// the one-column leg and a three-column step union, each a pair of pool
/// dispatches — in a steady state, under flat lanes and under CCD-major lanes
/// (the legs' `_on` forms take them): the CCD-major dispatch keeps its tables on the
/// stack, so its calls make no more than the flat ones' (a table made per
/// dispatch is two allocations a call). Each arm's count is the least of three
/// repeats of eight calls after twenty that fill the pools, so a worker's
/// first touch of a scratch is not read as a table.
#[test]
#[ignore = "hw: needs the box; reads no model file"]
fn hw_a_ccd_dispatch_allocates_no_more_than_a_flat_one() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let layer = r8layer::Layer::write("alloc-ccd", 512, 256, 16, GgmlType::Q4_K);
    let (embd, ff) = (layer.embd, layer.ff);
    let split = Split::open(&layer.source).unwrap();
    let src = R8Source::rows(&split);
    let host = HostLayer::build(src, &layer.spec()).unwrap();
    let mut hs = HostScratch::new(embd, ff, 4).unwrap();
    let mut us = UnionScratch::new_routed(embd, ff, 3, 4).unwrap();
    let x = Tensor2::from_vec(
        embd,
        1,
        (0..embd).map(|i| (i % 17) as f32 / 9.0 - 0.9).collect(),
    );
    let x3 = Tensor2::from_vec(
        embd,
        3,
        (0..3 * embd)
            .map(|i| (i % 23) as f32 / 11.0 - 1.0)
            .collect(),
    );
    let list =
        |j: u32| -> Vec<(u32, f32)> { (0..4).map(|i| ((j * 5 + i * 3) % 16, 0.25)).collect() };
    let lists = [list(0), list(1), list(2)];
    let refs: Vec<&[(u32, f32)]> = lists.iter().map(Vec::as_slice).collect();
    let (mut out1, mut out3) = (vec![0.0f32; embd], vec![0.0f32; 3 * embd]);
    let mut calls = |lanes: Lanes, n: usize| {
        for _ in 0..n {
            host.experts_into_on(lanes, src, &x, &lists[0], &mut out1, &mut hs, None)
                .unwrap();
            host.experts_step_union_into_on(lanes, src, &x3, &refs, &mut out3, &mut us)
                .unwrap();
        }
    };
    IS_MAIN.with(|m| m.set(true));
    let mut least = [u64::MAX; 2];
    let ccd = Lanes::Ccd(CcdMap::new(threads::pool().threads(), 4, 0));
    for (arm, lanes) in [(0, Lanes::Flat), (1, ccd)] {
        calls(lanes, 20);
        for _ in 0..3 {
            let c0 = CALLS.load(Relaxed);
            calls(lanes, 8);
            least[arm] = least[arm].min(CALLS.load(Relaxed) - c0);
        }
    }
    eprintln!(
        "host legs, 8 calls of one-column + 3-column step union: {} allocator calls flat, {} CCD-major",
        least[0], least[1]
    );
    assert!(
        least[1] <= least[0],
        "CCD-major lanes made {} allocator calls over 8 calls, the flat lanes {}",
        least[1],
        least[0]
    );
}

/// A [`TierSlots`] over a host layer's own bytes for the pin arm: every slot is
/// filled, and a pin takes an id buffer from a pool and hands it back cleared
/// when it drops, the way the gpu tier's arena lends them — four buffers cut
/// at the build, each with the capacity a pin can need.
///
/// With `late` set ([`PoolTier::late`]) a claim holding one of those ids is a
/// pending ticket, as a miss is: the id lends nothing until the ticket is
/// waited for, and the other ids of the claim are lent at once.
struct PoolTier {
    parts: Vec<Vec<u8>>,
    idle: Mutex<Vec<Vec<u32>>>,
    /// The `ensure_pinned` calls the legs made.
    pins: AtomicU64,
    /// Bit `id` set: the id's fill ends at the ticket's wait.
    late: u64,
    /// The late ids of the claim in flight, by bit; zero once it is waited for.
    flight: AtomicU64,
    /// The pending tickets issued.
    pending: AtomicU64,
    /// The lends made while a claim was in flight: its hits read under the
    /// misses' fill, the step's two-group path.
    early_lends: AtomicU64,
    /// The pin of the one ticket in flight, when the wait is borrowed rather
    /// than boxed. A second ticket that arrives while this is set takes the
    /// boxed wait.
    stash: Mutex<Option<PinToken>>,
}

impl PoolTier {
    fn of(split: &Split, layer: &HostLayer, ids_per_pin: usize) -> PoolTier {
        let mut parts = Vec::new();
        for e in 0..layer.n_expert() {
            for stack in layer.stacks() {
                parts.push(
                    ShardTensor::expert(stack, split, e)
                        .unwrap()
                        .bytes()
                        .to_vec(),
                );
            }
        }
        let mut idle = Vec::with_capacity(4);
        idle.extend((0..4).map(|_| Vec::with_capacity(ids_per_pin)));
        PoolTier {
            parts,
            idle: Mutex::new(idle),
            pins: AtomicU64::new(0),
            late: 0,
            flight: AtomicU64::new(0),
            pending: AtomicU64::new(0),
            early_lends: AtomicU64::new(0),
            stash: Mutex::new(None),
        }
    }

    /// The tier with `ids` late: their fill ends at the ticket's wait.
    fn late(mut self, ids: &[u32]) -> PoolTier {
        assert!(ids.iter().all(|&id| id < 64), "late ids are bits of a word");
        self.late = ids.iter().fold(0, |m, &id| m | 1u64 << id);
        self
    }

    fn pinned(&self, layer: usize, ids: &[u32]) -> TierPin<'_> {
        let mut held = self.idle.lock().unwrap().pop().unwrap_or_default();
        held.extend_from_slice(ids);
        TierPin::new(self, PinToken { layer, ids: held })
    }
}

impl PinRelease for PoolTier {
    fn release(&self, mut token: PinToken) {
        token.ids.clear();
        self.idle.lock().unwrap().push(token.ids);
    }
}

impl<'a> TicketWait<'a> for PoolTier {
    fn finish_ticket(&'a self) -> Result<TierPin<'a>, TierError> {
        self.flight.store(0, Relaxed);
        let token = self
            .stash
            .lock()
            .unwrap()
            .take()
            .expect("a step ticket settles once");
        Ok(TierPin::new(self, token))
    }

    fn abandon_ticket(&'a self) {
        self.flight.store(0, Relaxed);
        if let Some(token) = self.stash.lock().unwrap().take() {
            self.release(token);
        }
    }
}

impl TierSlots for PoolTier {
    fn ensure_as(&self, _layer: usize, _ids: &[u32], _claim: Claim) -> Result<(), TierError> {
        Ok(())
    }

    fn pin<'a>(&'a self, layer: usize, ids: &[u32]) -> Result<TierPin<'a>, TierError> {
        Ok(self.pinned(layer, ids))
    }

    fn slots(&self, _layer: usize) -> usize {
        0
    }

    fn ensure_pinned<'a>(
        &'a self,
        layer: usize,
        ids: &[u32],
        _claim: Claim,
    ) -> Result<TierPin<'a>, TierError> {
        self.pins.fetch_add(1, Relaxed);
        Ok(self.pinned(layer, ids))
    }

    fn prefetch_pinned<'a>(
        &'a self,
        layer: usize,
        ids: &[u32],
        claim: Claim,
    ) -> Result<Ticket<'a>, TierError> {
        let in_flight = ids.iter().fold(0, |m, &id| m | ((1u64 << id) & self.late));
        if in_flight == 0 {
            return self.ensure_pinned(layer, ids, claim).map(Ticket::ready);
        }
        self.pending.fetch_add(1, Relaxed);
        if self.stash.lock().unwrap().is_some() {
            let pin = self.pinned(layer, ids);
            self.flight.store(in_flight, Relaxed);
            return Ok(Ticket::new(move || {
                self.flight.store(0, Relaxed);
                Ok(pin)
            }));
        }
        let mut held = self.idle.lock().unwrap().pop().unwrap_or_default();
        held.clear();
        held.extend_from_slice(ids);
        *self.stash.lock().unwrap() = Some(PinToken { layer, ids: held });
        self.flight.store(in_flight, Relaxed);
        Ok(Ticket::bound(self))
    }

    fn slot(&self, _layer: usize, id: u32, part: usize) -> Option<&[u8]> {
        let in_flight = self.flight.load(Relaxed);
        if in_flight & (1u64 << id) != 0 {
            return None;
        }
        if in_flight != 0 {
            self.early_lends.fetch_add(1, Relaxed);
        }
        Some(&self.parts[3 * id as usize + part])
    }
}

/// Allocator calls one pin adds to a step leg's call over a paged layer, against
/// the same layer with no tier, in a steady state. The one-column leg and a
/// three-column step union each take one pin a call.
///
/// PIN(2026-10-12): 0. Derived from the code: the model side's pin was a
/// `Box<dyn Send>` over the tier's guard, one call a pin, two a call of this
/// arm's two legs and sixteen over its eight calls. The pin is now a borrow of
/// the tier and a token that holds an id buffer the tier lent, moved in and
/// back, so the model side makes none; the tier's own half (its id buffers) is
/// the gpu crate's gate (`gate-gpu-lib`).
const PIN_CALLS: u64 = 0;

/// The model side of a paged layer's step makes no allocator call of its own:
/// the legs over a layer with a tier whose slots are all the arena's (so the
/// one-column leg and the union both pin) make no more calls than the legs
/// over the same layer with none. Each arm's count is the least of three
/// repeats of eight calls after twenty that fill the pools.
#[test]
#[ignore = "hw: needs the box; reads no model file"]
fn hw_a_tier_pin_adds_no_allocator_call_to_the_step_legs() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let layer = r8layer::Layer::write("alloc-pin", 512, 256, 16, GgmlType::Q4_K);
    let (embd, ff) = (layer.embd, layer.ff);
    let split = Split::open(&layer.source).unwrap();
    let src = R8Source::rows(&split);
    let mut host = HostLayer::build(src, &layer.spec()).unwrap();
    let mut hs = HostScratch::new(embd, ff, 4).unwrap();
    let mut us = UnionScratch::new_routed(embd, ff, 3, 4).unwrap();
    let x = Tensor2::from_vec(
        embd,
        1,
        (0..embd).map(|i| (i % 17) as f32 / 9.0 - 0.9).collect(),
    );
    let x3 = Tensor2::from_vec(
        embd,
        3,
        (0..3 * embd)
            .map(|i| (i % 23) as f32 / 11.0 - 1.0)
            .collect(),
    );
    let list =
        |j: u32| -> Vec<(u32, f32)> { (0..4).map(|i| ((j * 5 + i * 3) % 16, 0.25)).collect() };
    let lists = [list(0), list(1), list(2)];
    let refs: Vec<&[(u32, f32)]> = lists.iter().map(Vec::as_slice).collect();
    let (mut out1, mut out3) = (vec![0.0f32; embd], vec![0.0f32; 3 * embd]);
    IS_MAIN.with(|m| m.set(true));
    let mut least = |host: &HostLayer| {
        let mut calls = |n: usize| {
            for _ in 0..n {
                host.experts_into_on(Lanes::Flat, src, &x, &lists[0], &mut out1, &mut hs, None)
                    .unwrap();
                host.experts_step_union_into_on(Lanes::Flat, src, &x3, &refs, &mut out3, &mut us)
                    .unwrap();
            }
        };
        calls(20);
        let mut least = u64::MAX;
        for _ in 0..3 {
            let c0 = CALLS.load(Relaxed);
            calls(8);
            least = least.min(CALLS.load(Relaxed) - c0);
        }
        least
    };
    let flat = least(&host);
    // Every id is the arena's: no host segment, so the union pins too.
    let tier = Arc::new(PoolTier::of(&split, &host, 16));
    host.attach_tier(0, 0..0, tier.clone());
    let paged = least(&host);
    eprintln!(
        "step legs, 8 calls of one-column + 3-column step union: {flat} allocator calls with no tier, {paged} with one"
    );
    assert_eq!(
        tier.pins.load(Relaxed),
        2 * (20 + 3 * 8),
        "the arm's legs took one pin each call"
    );
    assert!(
        paged <= flat + PIN_CALLS * 2 * 8, // two pins a call, eight calls
        "the legs over a paged layer made {paged} allocator calls over 8 calls, {flat} over the same layer with no tier"
    );
}

/// Allocator calls a step union adds when some ids of its claim are misses,
/// per call of the arm's two legs, against the same layer with no tier.
///
/// PIN(2026-10-14): 0. The pending ticket borrows the tier (`Ticket::bound`);
/// the boxed wait is only the overlap, which this arm does not issue. The
/// one-column miss takes the same borrowed ticket. The tier double's id
/// buffer is the pin pool's.
const MISS_CALLS: u64 = 0;

/// A steady step whose union claim holds misses takes the two-group path and
/// makes no more allocator calls than [`MISS_CALLS`] a call over the same
/// layer with no tier. The arm proves it took that path: every union call was
/// issued a pending ticket and its hits were lent while the misses were in
/// flight. Counts are the least of three repeats of eight calls after twenty.
#[test]
#[ignore = "hw: needs the box; reads no model file"]
fn hw_a_step_with_misses_pins_the_allocator_calls_of_its_miss_path() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let layer = r8layer::Layer::write("alloc-miss", 512, 256, 16, GgmlType::Q4_K);
    let (embd, ff) = (layer.embd, layer.ff);
    let split = Split::open(&layer.source).unwrap();
    let src = R8Source::rows(&split);
    let mut host = HostLayer::build(src, &layer.spec()).unwrap();
    let mut hs = HostScratch::new(embd, ff, 4).unwrap();
    let mut us = UnionScratch::new_routed(embd, ff, 3, 4).unwrap();
    let x = Tensor2::from_vec(
        embd,
        1,
        (0..embd).map(|i| (i % 17) as f32 / 9.0 - 0.9).collect(),
    );
    let x3 = Tensor2::from_vec(
        embd,
        3,
        (0..3 * embd)
            .map(|i| (i % 23) as f32 / 11.0 - 1.0)
            .collect(),
    );
    let list =
        |j: u32| -> Vec<(u32, f32)> { (0..4).map(|i| ((j * 5 + i * 3) % 16, 0.25)).collect() };
    let lists = [list(0), list(1), list(2)];
    let refs: Vec<&[(u32, f32)]> = lists.iter().map(Vec::as_slice).collect();
    let (mut out1, mut out3) = (vec![0.0f32; embd], vec![0.0f32; 3 * embd]);
    IS_MAIN.with(|m| m.set(true));
    let mut least = |host: &HostLayer| {
        let mut calls = |n: usize| {
            for _ in 0..n {
                host.experts_into_on(Lanes::Flat, src, &x, &lists[0], &mut out1, &mut hs, None)
                    .unwrap();
                host.experts_step_union_into_on(Lanes::Flat, src, &x3, &refs, &mut out3, &mut us)
                    .unwrap();
            }
        };
        calls(20);
        let mut least = u64::MAX;
        for _ in 0..3 {
            let c0 = CALLS.load(Relaxed);
            calls(8);
            least = least.min(CALLS.load(Relaxed) - c0);
        }
        least
    };
    let flat = least(&host);
    // The union lists ids 0 3 5 6 8 9 10 11 13 14: four are late, six are hits.
    let tier = Arc::new(PoolTier::of(&split, &host, 16).late(&[5, 8, 11, 14]));
    host.attach_tier(0, 0..0, tier.clone());
    let missed = least(&host);
    eprintln!(
        "step legs, 8 calls of one-column + 3-column step union: {flat} allocator calls with no tier, {missed} with misses"
    );
    assert_eq!(
        tier.pending.load(Relaxed),
        20 + 3 * 8,
        "every union call of the arm was issued a pending ticket: the arm took the miss path"
    );
    assert!(
        tier.early_lends.load(Relaxed) > 0,
        "the arm's hits were never lent while the misses were in flight: it did not take the two-group path"
    );
    assert!(
        missed <= flat + MISS_CALLS * 8, // one miss union a call, eight calls
        "the legs over a layer with misses made {missed} allocator calls over 8 calls, {flat} over the same layer with no tier"
    );
}

/// A one-column step whose list holds a miss takes the resident split and
/// adds no allocator call over the same layer with no tier. The arm's list
/// is a hit and a late id, so the ticket is pending and the hit is lent
/// while the miss is in flight.
#[test]
#[ignore = "hw: needs the box; reads no model file"]
fn hw_a_one_column_miss_allocates_nothing() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let layer = r8layer::Layer::write("alloc-col", 512, 256, 16, GgmlType::Q4_K);
    let (embd, ff) = (layer.embd, layer.ff);
    let split = Split::open(&layer.source).unwrap();
    let src = R8Source::rows(&split);
    let mut host = HostLayer::build(src, &layer.spec()).unwrap();
    let mut hs = HostScratch::new(embd, ff, 4).unwrap();
    let mut us = UnionScratch::new_routed(embd, ff, 1, 4).unwrap();
    let x = Tensor2::from_vec(
        embd,
        1,
        (0..embd).map(|i| (i % 17) as f32 / 9.0 - 0.9).collect(),
    );
    let list = [(0u32, 0.25), (5, 0.25)];
    let mut out = vec![0.0f32; embd];
    IS_MAIN.with(|m| m.set(true));
    let mut least = |host: &HostLayer, resident: bool| {
        let mut calls = |n: usize| {
            for _ in 0..n {
                if resident {
                    host.experts_into_on(
                        Lanes::Flat,
                        src,
                        &x,
                        &list,
                        &mut out,
                        &mut hs,
                        Some(&mut us),
                    )
                    .unwrap();
                } else {
                    host.experts_into_on(Lanes::Flat, src, &x, &list, &mut out, &mut hs, None)
                        .unwrap();
                }
            }
        };
        calls(20);
        let mut least = u64::MAX;
        for _ in 0..3 {
            let c0 = CALLS.load(Relaxed);
            calls(8);
            least = least.min(CALLS.load(Relaxed) - c0);
        }
        least
    };
    let flat = least(&host, false);
    let tier = Arc::new(PoolTier::of(&split, &host, 16).late(&[5]));
    host.attach_tier(0, 0..0, tier.clone());
    let missed = least(&host, true);
    assert_eq!(
        tier.pending.load(Relaxed),
        20 + 3 * 8,
        "every one-column call was issued a pending ticket"
    );
    assert!(
        tier.early_lends.load(Relaxed) > 0,
        "the hit was never lent while the miss was in flight"
    );
    assert!(
        missed <= flat,
        "the one-column miss made {missed} allocator calls over 8 calls, {flat} with no tier"
    );
}
