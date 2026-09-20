//! Gate tests for the resident pool: partition exactness, coverage,
//! repeated-call barrier health, panic propagation, and (hw_) topology.
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

use threads::{chunks, pool};

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
