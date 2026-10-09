//! Gate tests for the resident pool: partition exactness, coverage,
//! repeated-call barrier health, panic propagation, the count of a contended
//! dispatch's wait, and (hw_) topology.
//!
//! The pure tests (everything but `hw_topology`) run in the default loop:
//! they need no pinning to be correct — an unpinned pool must produce
//! identical partitions and coverage. `hw_` is `#[ignore]`d per the repo
//! convention (`cargo nextest` is not installed on the box, so run this file
//! with `cargo test -p bloomery-threads --test pool -- --include-ignored
//! --nocapture` via `tools/box.sh`, i.e. `just gate-threads`).

use std::ops::Range;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc;

use threads::{CcdMap, chunks, pool};

/// Exhaustive partition check: every `n` in 0..=257 against every `threads`
/// in 1..=64 — chunks are contiguous, ascending, non-overlapping, their union
/// is exactly `0..n`, the count is exactly `threads`, and lengths differ by at
/// most 1. Also: the same `(n, threads)` yields the identical Vec twice.
#[test]
fn chunks_partition_exhaustive() {
    for n in 0..=257usize {
        for t in 1..=64usize {
            let cs = chunks(n, t);
            assert_eq!(cs.len(), t, "n={n} t={t}: chunk count");
            assert_eq!(cs[0].start, 0, "n={n} t={t}: first chunk starts at 0");
            for w in cs.windows(2) {
                assert_eq!(
                    w[0].end, w[1].start,
                    "n={n} t={t}: chunks must be contiguous, non-overlapping, ascending"
                );
            }
            assert_eq!(cs[t - 1].end, n, "n={n} t={t}: union must be exactly 0..n");
            let maxlen = cs.iter().map(Range::len).max().unwrap();
            let minlen = cs.iter().map(Range::len).min().unwrap();
            assert!(
                maxlen - minlen <= 1,
                "n={n} t={t}: lengths {minlen}..{maxlen} differ by more than 1"
            );
            assert_eq!(
                cs,
                chunks(n, t),
                "n={n} t={t}: partition must be deterministic"
            );
        }
    }
}

/// Every index of `0..n` is visited exactly once. 10944 is this model's
/// `ffn_up` row count; 31/32/33 straddle the thread count on the box.
#[test]
fn for_each_chunk_covers() {
    for &n in &[0usize, 1, 31, 32, 33, 10944] {
        let hits: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
        pool().for_each_chunk(n, |c: Range<usize>| {
            for i in c {
                hits[i].fetch_add(1, Ordering::Relaxed);
            }
        });
        for (i, h) in hits.iter().enumerate() {
            assert_eq!(
                h.load(Ordering::Relaxed),
                1,
                "n={n}: index {i} must be visited exactly once"
            );
        }
    }
}

/// 10000 consecutive dispatches on the same pool, coverage checked every
/// time — a barrier that drifts out of step (missed wake, double arrival)
/// fails here, not just once at the end.
#[test]
fn repeated_calls_stay_correct() {
    let p = pool();
    for it in 0..10000usize {
        let n = 33 + it % 7;
        let hits: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
        p.for_each_chunk(n, |c: Range<usize>| {
            for i in c {
                hits[i].fetch_add(1, Ordering::Relaxed);
            }
        });
        for (i, h) in hits.iter().enumerate() {
            assert_eq!(h.load(Ordering::Relaxed), 1, "iter {it}: index {i}");
        }
    }
}

/// A panicking chunk must (a) reach the caller and (b) leave the pool
/// usable. The body runs in a helper thread with a 30 s channel timeout so a
/// stranded barrier fails the test instead of hanging it.
#[test]
fn panic_propagates_and_pool_survives() {
    let (tx, rx) = mpsc::channel();
    let body = std::thread::spawn(move || {
        let p = pool();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            p.for_each_chunk(64, |c: Range<usize>| {
                if c.contains(&7) {
                    panic!("boom in chunk {c:?}");
                }
            });
        }));
        assert!(r.is_err(), "the worker's panic must reach the caller");
        // The very next dispatch must work — the barrier survived the panic.
        let hits: Vec<AtomicU32> = (0..64).map(|_| AtomicU32::new(0)).collect();
        p.for_each_chunk(64, |c: Range<usize>| {
            for i in c {
                hits[i].fetch_add(1, Ordering::Relaxed);
            }
        });
        for (i, h) in hits.iter().enumerate() {
            assert_eq!(h.load(Ordering::Relaxed), 1, "post-panic: index {i}");
        }
        tx.send(()).unwrap();
    });
    match rx.recv_timeout(std::time::Duration::from_secs(30)) {
        Ok(()) => body.join().expect("panic-test body"),
        Err(_) => panic!("pool stuck for 30 s after a chunk panic (barrier deadlock?)"),
    }
}

/// Nesting `for_each_chunk` inside a chunk closure deadlocks — from a worker on
/// the dispatch mutex the caller holds, from the calling thread (which runs the
/// last chunk itself) on the mutex it holds already. Either way a hang with no
/// message. The pool asserts instead, and this proves the assertion fires — on
/// the caller's own chunk too, not just the resident workers.
///
/// The watchdog matters here more than anywhere: if the assertion is ever removed,
/// this test does not fail, it hangs — so the 30 s channel is the actual gate.
#[test]
fn nested_dispatch_asserts_instead_of_hanging() {
    let (tx, rx) = mpsc::channel();
    let body = std::thread::spawn(move || {
        let p = pool();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            p.for_each_chunk(64, |_c: Range<usize>| {
                p.for_each_chunk(8, |_: Range<usize>| {});
            });
        }));
        tx.send(r.is_err()).unwrap();
    });
    match rx.recv_timeout(std::time::Duration::from_secs(30)) {
        Ok(fired) => {
            body.join().expect("nested-dispatch body");
            assert!(
                fired,
                "a nested for_each_chunk must panic, not succeed silently"
            );
        }
        Err(_) => {
            panic!("nested for_each_chunk deadlocked for 30 s — the reentrancy assert is gone")
        }
    }
}

/// A dispatch that finds another caller's job holding the pool is counted
/// with its wait: a job on another thread holds the dispatch for its chunks'
/// sleep, and this thread's dispatch, started once that job is inside its
/// chunks, waits for it — `dispatch_waits` moves by at least one and
/// `dispatch_wait_ns` by at least part of the sleep. A pool of one thread
/// takes no lock and counts nothing.
#[test]
fn a_contended_dispatch_counts_its_wait() {
    let p = pool();
    let before = p.stats();
    let (inside, entered) = mpsc::channel();
    let holder = std::thread::spawn(move || {
        pool().for_each_chunk(pool().threads(), |c: Range<usize>| {
            if c.start == 0 {
                inside.send(()).unwrap();
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        });
    });
    entered
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("the holding job entered its chunks");
    p.for_each_chunk(1, |_: Range<usize>| {});
    holder.join().expect("the holding job");
    let after = p.stats();
    if p.threads() == 1 {
        assert_eq!(
            after.dispatch_waits, before.dispatch_waits,
            "one thread: no lock"
        );
        return;
    }
    assert!(
        after.dispatch_waits > before.dispatch_waits,
        "the contended dispatch was counted: {before:?} -> {after:?}"
    );
    assert!(
        after.dispatch_wait_ns - before.dispatch_wait_ns >= 10_000_000,
        "its wait covers part of the holder's 50 ms: {before:?} -> {after:?}"
    );
}

/// Reads the real /sys topology. Asserted strongly but portably: at least one
/// CCD group, every group non-empty, no duplicate cpu ids. On the box this
/// prints 4 groups of 8 primaries + 8 siblings and threads=32 — that shape is
/// verified by eye in the gate output (`--nocapture`), not asserted, so the
/// gate also runs on other machines.
#[test]
#[ignore = "hw: reads /sys/devices/system/cpu topology, meaningful on the box"]
fn hw_topology() {
    let p = pool();
    println!(
        "hw_topology: threads={} pin_failed={}",
        p.threads(),
        p.pin_failed()
    );
    let topo = p.topology();
    for (i, g) in topo.iter().enumerate() {
        println!("hw_topology: ccd{i} cpus={g:?}");
    }
    assert!(!topo.is_empty(), "at least one CCD group");
    for (i, g) in topo.iter().enumerate() {
        assert!(!g.is_empty(), "ccd{i} must be non-empty");
    }
    let mut all: Vec<u32> = topo.iter().flatten().copied().collect();
    let n = all.len();
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), n, "no duplicate cpu ids across CCD groups");
    assert!(!p.pin_failed(), "pinning must succeed on the box");
}

/// The map's rule, restated: participant `t` is on CCD `t % ccds` as that
/// CCD's `t / ccds`-th participant. Over every thread and CCD count up to
/// 64 the participants fill each CCD's ranks exactly once, a CCD holds
/// `threads / ccds` participants (the first `threads % ccds` CCDs one more),
/// the CCD-major lanes run CCD 0's ranks first, and the lane's CCD is the
/// participant's.
#[test]
fn ccd_map_places_participants_by_the_pinning_rule() {
    for threads in 1..=64usize {
        for ccds in 1..=16usize {
            let m = CcdMap::new(threads, ccds, 0);
            let mut seen = vec![false; threads];
            for t in 0..threads {
                assert_eq!(m.ccd_of(t), t % ccds, "threads={threads} ccds={ccds} t={t}");
                assert_eq!(
                    m.rank_of(t),
                    t / ccds,
                    "threads={threads} ccds={ccds} t={t}"
                );
                assert!(m.rank_of(t) < m.width(m.ccd_of(t)));
                let lane = m.lane_of(t);
                assert!(
                    !std::mem::replace(&mut seen[lane], true),
                    "lane {lane} twice"
                );
                assert_eq!(m.lane_ccd(lane), m.ccd_of(t));
            }
            let widths: Vec<usize> = (0..ccds).map(|c| m.width(c)).collect();
            assert_eq!(widths.iter().sum::<usize>(), threads);
            for (c, &w) in widths.iter().enumerate() {
                let want = threads / ccds + usize::from(c < threads % ccds);
                assert_eq!(w, want, "threads={threads} ccds={ccds} ccd {c}");
                assert_eq!(m.first_lane(c), widths[..c].iter().sum::<usize>());
            }
            assert_eq!(m.first_lane(ccds), threads);
            assert_eq!(m.active(), ccds.min(threads));
        }
    }
}

/// A CCD's share of a matrix's units is as wide as its participants are
/// many: 30 threads on four CCDs are 8/8/7/7, so 300 units cut 80/80/70/70
/// (four equal quarters would give every CCD 75), 8 threads on three CCDs
/// are 3/3/2, so 80 units cut 30/30/20. Every span count partitions `0..n`
/// in CCD order, its edges at `n · first_lane(c) / threads` rounded down.
///
/// PIN(2026-10-10): the share was the sum of the pool's own chunks of the
/// CCD's lanes, which hands the `n % threads` remainder units to the first
/// lanes — CCD 0's, then CCD 1's — of every matrix of a dispatch: 80 units on
/// 32 threads and four CCDs were 24/24/16/16 (3/3/2/2 a participant, CCD 0 and
/// 1 +50 % of the least), 80 on 30 threads 24/24/18/14. A floor of a
/// proportional edge is within one unit of the ideal: for `x = n · w / T`,
/// `floor(a + x) − floor(a)` is `floor(x)` or `ceil(x)`, so a CCD of `w` of
/// the `T` participants holds `len` units with `|len · T − n · w| < T`, and a
/// participant `len / w` units within `1 / w` of the mean `n / T`. FAIL-first:
/// the chunk-sum share reads 24 units for CCD 0 at `n = 80`, `T = 32`, where
/// the bound allows 20 ± 1.
#[test]
fn ccd_spans_follow_the_participants_per_ccd() {
    let lens =
        |m: CcdMap, n: usize| -> Vec<usize> { (0..m.ccds()).map(|c| m.span(n, c).len()).collect() };
    assert_eq!(lens(CcdMap::new(30, 4, 0), 300), [80, 80, 70, 70]);
    assert_eq!(lens(CcdMap::new(8, 3, 0), 80), [30, 30, 20]);
    assert_eq!(lens(CcdMap::new(32, 4, 0), 2048), [512; 4]);
    assert_eq!(lens(CcdMap::new(3, 4, 0), 30), [10, 10, 10, 0]);
    // The shapes of a routed expert's matrices that no thread count divides:
    // 80 row-lane groups (640 rows of Q3_K gates and ups) on 32 threads are
    // 2.5 groups a participant on every CCD, not 3/3/2/2; 640 rows, 256 and 80
    // groups on 30 threads stay within the bound.
    assert_eq!(lens(CcdMap::new(32, 4, 0), 80), [20; 4]);
    assert_eq!(lens(CcdMap::new(30, 4, 0), 640), [170, 171, 149, 150]);
    assert_eq!(lens(CcdMap::new(30, 4, 0), 256), [68, 68, 60, 60]);
    assert_eq!(lens(CcdMap::new(30, 4, 0), 80), [21, 21, 19, 19]);
    for (threads, ccds) in [
        (1, 1),
        (8, 3),
        (24, 4),
        (30, 4),
        (32, 4),
        (7, 5),
        (64, 16),
        (3, 8),
    ] {
        let m = CcdMap::new(threads, ccds, 0);
        for n in 0..=700usize {
            let mut at = 0;
            for c in 0..ccds {
                let s = m.span(n, c);
                assert_eq!(
                    s.start, at,
                    "threads={threads} ccds={ccds} n={n} ccd {c}: contiguous"
                );
                at = s.end;
                assert_eq!(
                    s.start,
                    n * m.first_lane(c) / threads,
                    "threads={threads} ccds={ccds} n={n} ccd {c}: the proportional edge"
                );
                let w = m.width(c);
                assert!(
                    (s.len() * threads).abs_diff(n * w) < threads,
                    "threads={threads} ccds={ccds} n={n} ccd {c}: {} units for {w} of {threads} participants",
                    s.len()
                );
            }
            assert_eq!(
                at, n,
                "threads={threads} ccds={ccds} n={n}: the spans cover 0..n"
            );
        }
    }
}

/// The map against the machine: on the box every worker's pinned cpu is in
/// the L3 group of the CCD the map names for it, the caller's too once it is
/// pinned, the L3 bytes are read, and a spread over the pool's own groups is
/// what the pool reports. An independent path from the rule's restatement
/// above: the cpus come from `sched_setaffinity`'s arguments.
#[test]
#[ignore = "hw: pins the calling thread and reads /sys/devices/system/cpu topology, meaningful on the box"]
fn hw_ccd_map_matches_the_pins() {
    let p = pool();
    assert!(p.pin_caller(), "pinning must succeed on the box");
    let map = p.ccd_map().expect("every participant is pinned");
    let topo = p.topology();
    assert_eq!(map.threads(), p.threads());
    assert_eq!(map.ccds(), topo.len());
    for t in 0..p.threads() {
        let cpu = if t + 1 < p.threads() {
            p.worker_cpu(t).expect("a pinned worker has a cpu") as u32
        } else {
            u32::try_from(p.caller_cpu()).unwrap()
        };
        assert!(
            topo[map.ccd_of(t)].contains(&cpu),
            "participant {t} is pinned to cpu {cpu}, outside CCD {} {:?}",
            map.ccd_of(t),
            topo[map.ccd_of(t)]
        );
    }
    println!(
        "hw_ccd_map: threads={} ccds={} widths={:?} l3_bytes={}",
        map.threads(),
        map.ccds(),
        (0..map.ccds()).map(|c| map.width(c)).collect::<Vec<_>>(),
        map.l3_bytes()
    );
    assert!(map.l3_bytes() > 0, "the box's sysfs names its L3 size");
}
