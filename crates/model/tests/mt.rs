//! Gate for the row-parallel `matmul_q`: the thread count must not change the
//! logits, not by one bit.
//!
//! The split axis is the output row `r` and only that: every output element
//! still sums its `k` products ascending, in the same order it always did, so
//! which worker computes which row is invisible in the arithmetic. This gate
//! exists because that invisibility is a claim that breaks silently — a gather
//! that forgets the chunk base, a chunk bound off by one, a scratch row shared
//! between workers. None of those move the answer by a visible amount on a
//! lucky prompt; a byte compare has no luck.
//!
//! `BLOOMERY_THREADS` is read once per process when the pool is built, so one
//! process cannot be both 1-threaded and 32-threaded. The dumps therefore run
//! in re-exec'd children of this very binary (the pattern `tests/profile.rs`
//! uses for `BLOOMERY_PROFILE`), each writing raw logits bytes that the parent
//! compares. `=3` is in the set on purpose: a row count that does not divide
//! the thread count is where a partition bug actually lives — the first
//! `n % threads` chunks are one row longer than the rest.
//!
//! The children's pool sizes are printed as evidence, not asserted: the
//! assertion is about the logits, and the printed `threads::pool().threads()`
//! lines are how a reader confirms the comparison really exercised the pool
//! rather than running 1 thread three times.
//!
//! `hw_` prefix: needs the box and the model file. Public API only (`forward`)
//! — a parallel round is changing `forward.rs` internals right now, and a gate
//! that reaches into them breaks on merge.
#[path = "common/oracle.rs"]
mod oracle;

use model::forward::forward;

/// The re-exec entry point. Not a test of its own: when `BLOOMERY_MT_CHILD_DUMP`
/// is absent (i.e. someone ran the file directly) it returns without doing
/// anything — all assertions live in [`hw_mt_gate`].
#[test]
#[ignore = "hw: re-exec child of hw_mt_gate; standalone it is a no-op"]
fn hw_mt_child_logits() {
    let Ok(dump) = std::env::var("BLOOMERY_MT_CHILD_DUMP") else {
        eprintln!("child helper: no BLOOMERY_MT_CHILD_DUMP, nothing to do");
        return;
    };
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(oracle::model_path()).unwrap();
    let tokens: Vec<u32> = o.tokens.iter().map(|&t| t as u32).collect();
    let logits = forward(&g, &tokens).unwrap();
    let bytes: Vec<u8> = logits.data.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&dump, bytes).unwrap();
    // The pool already exists — `forward` built it on its first `matmul_q`.
    eprintln!(
        "mt child: {} logits with threads::pool().threads() = {} dumped to {dump}",
        logits.data.len(),
        threads::pool().threads()
    );
}

/// Run `forward` on the shared prompt in a child process pinned to the given
/// `BLOOMERY_THREADS` value, returning its raw logits bytes and its stderr (the
/// stderr carries the pool-size evidence line). Each child is a fresh process,
/// which is the only way one test can compare several thread counts.
fn child_logits(threads_env: &str) -> (Vec<u8>, String) {
    let exe = std::env::current_exe().unwrap();
    let dump = std::env::temp_dir().join(format!(
        "bloomery-mt-child-{}-t{threads_env}.f32",
        std::process::id()
    ));
    let out = std::process::Command::new(exe)
        .args(["--exact", "hw_mt_child_logits", "--ignored", "--nocapture"])
        .env("BLOOMERY_MT_CHILD_DUMP", &dump)
        .env("BLOOMERY_THREADS", threads_env)
        .output()
        .expect("re-exec of this test binary");
    assert!(
        out.status.success(),
        "child at BLOOMERY_THREADS={threads_env} failed\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (
        std::fs::read(&dump).unwrap(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Byte equality of the logits across thread counts. The failure message names
/// the first differing byte so a red run says where, not just that.
fn assert_byte_identical(a: &[u8], b: &[u8], la: &str, lb: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "logit byte count {la} = {} vs {lb} = {}",
        a.len(),
        b.len()
    );
    let at = a.iter().zip(b).position(|(x, y)| x != y);
    assert_eq!(
        a,
        b,
        "logits differ between BLOOMERY_THREADS={la} and ={lb}: first differing byte \
         at index {:?} of {}",
        at,
        a.len()
    );
}

#[test]
#[ignore = "hw: needs the box and the model file; run via `just gate-mt`"]
fn hw_mt_gate() {
    let (one, one_err) = child_logits("1");
    let (wide, wide_err) = child_logits("32");
    let (odd, odd_err) = child_logits("3");

    // Evidence, printed whatever the verdict: each child reports the pool it
    // actually built. A run whose children silently shared one thread count
    // would pass the asserts below and prove nothing.
    eprint!("{one_err}");
    eprint!("{wide_err}");
    eprint!("{odd_err}");

    assert_byte_identical(&one, &wide, "1", "32");
    eprintln!(
        "logits, threads=1 vs threads=32        byte-identical ({} bytes)",
        one.len()
    );
    assert_byte_identical(&one, &odd, "1", "3");
    eprintln!(
        "logits, threads=1 vs threads=3         byte-identical ({} bytes)",
        one.len()
    );
}
