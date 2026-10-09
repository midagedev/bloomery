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
//!
//! The CCD-major lanes of the host tier's legs ([`ops::Lanes::Ccd`]) are held
//! to the same bit contract, with no model file: on a synthetic routed layer
//! the one-column leg, the step union at 1, 3 and 9 columns under both
//! deferral arms and a heterogeneous group each equal their flat-lane value
//! bit for bit, under every spread of the pool's participants over 1, 2, 3, 4
//! and 12 CCDs (the legs' `_on` forms take the lanes) — in children at 8 and 30 threads too,
//! where 8 over three CCDs is 3/3/2 and 30 over four is 8/8/7/7 — with
//! `BLOOMERY_POISON=1`, so a cell no lane computed is a NaN the compare sees.
#[path = "common/manifest.rs"]
mod manifest;
#[path = "common/model_path.rs"]
mod model_path;
#[path = "common/prompt.rs"]
mod prompt;
#[allow(
    dead_code,
    reason = "this gate renders one layer; the sidecar and the salted writer serve the union gates"
)]
#[path = "common/r8layer.rs"]
mod r8layer;

use gguf::{GgmlType, Split};
use model::arch::deepseek2::forward::forward;
use model::moe::{HostLayer, HostScratch, UnionScratch};
use model::ops::{self, Lanes, ShardTensor, Tensor2, matmul_q_group_into_on};
use model::r8file::R8Source;
use threads::CcdMap;

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
    let g = gguf::Gguf::open(model_path::model_path()).unwrap();
    let tokens: Vec<u32> = prompt::tokens();
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

/// Routed experts a list holds, and columns the widest union call takes.
const LIST: usize = 6;
const COLS: usize = 9;

/// `n` seeded values in [-1, 1), with an outlier now and then.
fn seeded(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let v = (state >> 40) as f32 / 8_388_608.0 - 1.0;
            if i % 61 == 0 { v * 8.0 } else { v }
        })
        .collect()
}

/// What every leg of the layer computes under `lanes`: the one-column leg of
/// each column's list, the step union at each width, and a group of three
/// matrices of two row counts.
fn legs(layer: &r8layer::Layer, split: &Split, lanes: Lanes) -> Vec<Vec<f32>> {
    let (embd, ff, n_expert) = (layer.embd, layer.ff, layer.n_expert);
    let src = R8Source::rows(split);
    let host = HostLayer::build(src, &layer.spec()).unwrap();
    let mut hs = HostScratch::new(embd, ff, LIST).unwrap();
    let mut us = UnionScratch::new_routed(embd, ff, COLS, LIST).unwrap();
    let x: Vec<f32> = (0..COLS)
        .flat_map(|j| seeded(embd, 0x51 + j as u64))
        .collect();
    let lists: Vec<Vec<(u32, f32)>> = (0..COLS)
        .map(|j| {
            (0..LIST)
                .map(|i| (((j * 7 + i * 5) % n_expert) as u32, 0.1 + 0.05 * i as f32))
                .collect()
        })
        .collect();
    let mut all = Vec::new();
    for j in 0..COLS {
        let xj = Tensor2::from_vec(embd, 1, x[j * embd..(j + 1) * embd].to_vec());
        let mut out = vec![f32::NAN; embd];
        host.experts_into_on(lanes, src, &xj, &lists[j], &mut out, &mut hs)
            .unwrap();
        all.push(out);
    }
    for k in [1, 3, COLS] {
        for defer in [true, false] {
            ops::set_defer_quant(Some(defer));
            let refs: Vec<&[(u32, f32)]> = lists[..k].iter().map(Vec::as_slice).collect();
            let mut out = vec![f32::NAN; embd * k];
            let view = Tensor2::from_vec(embd, k, x[..embd * k].to_vec());
            let done = host.experts_step_union_into_on(lanes, src, &view, &refs, &mut out, &mut us);
            ops::set_defer_quant(None);
            done.unwrap();
            all.push(out);
        }
    }
    // A group whose pairs differ in rows and in `k`: a gate and an up on the
    // embedding, a down on the feed-forward width.
    let ws = [
        ShardTensor::find(split, r8layer::GATE)
            .unwrap()
            .expert(split, 2)
            .unwrap(),
        ShardTensor::find(split, r8layer::UP)
            .unwrap()
            .expert(split, 5)
            .unwrap(),
        ShardTensor::find(split, r8layer::DOWN)
            .unwrap()
            .expert(split, 3)
            .unwrap(),
    ];
    let (xe, xf) = (
        Tensor2::from_vec(embd, 1, seeded(embd, 0x77)),
        Tensor2::from_vec(ff, 1, seeded(ff, 0x78)),
    );
    let mut outs = [
        Tensor2::zeros(ff, 1),
        Tensor2::zeros(ff, 1),
        Tensor2::zeros(embd, 1),
    ];
    matmul_q_group_into_on(lanes, "mt", &ws, &[&xe, &xe, &xf], &mut outs).unwrap();
    all.extend(outs.map(|o| o.data.clone()));
    all
}

/// The re-exec entry point of [`hw_ccd_lanes_gate`]: every leg under flat lanes
/// and under each spread over CCDs, which must agree bit for bit. A no-op
/// without `BLOOMERY_MT_CHILD_DUMP`, which here only says it is a child.
#[test]
#[ignore = "hw: re-exec child of hw_ccd_lanes_gate; standalone it is a no-op"]
fn hw_ccd_lanes_child() {
    if std::env::var("BLOOMERY_MT_CHILD_DUMP").is_err() {
        eprintln!("child helper: no BLOOMERY_MT_CHILD_DUMP, nothing to do");
        return;
    }
    let layer = r8layer::Layer::write("mt-ccd", 512, 256, 16, GgmlType::Q4_K);
    let split = Split::open(&layer.source).unwrap();
    let threads = threads::pool().threads();
    let want = legs(&layer, &split, Lanes::Flat);
    assert!(
        want.iter().flatten().all(|v| v.is_finite()),
        "the flat legs are finite, or the compare below sees nothing"
    );
    for ccds in [1, 2, 3, 4, 12] {
        let map = CcdMap::new(threads, ccds, 0);
        let got = legs(&layer, &split, Lanes::Ccd(map));
        let widths: Vec<usize> = (0..ccds).map(|c| map.width(c)).collect();
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let at = g
                .iter()
                .zip(w)
                .position(|(a, b)| a.to_bits() != b.to_bits());
            assert!(
                at.is_none(),
                "{threads} threads over {ccds} CCDs {widths:?}: leg {i} differs from the flat lanes at cell {at:?}"
            );
        }
        eprintln!(
            "ccd lanes: {threads} threads over {ccds} CCDs {widths:?}: {} legs bit-identical to flat",
            got.len()
        );
    }
}

/// Run [`hw_ccd_lanes_child`] in a fresh process at `threads_env` threads
/// (`None`: the machine's own count), with `BLOOMERY_POISON=1` or the lever
/// unset, returning its stderr.
fn ccd_child(threads_env: Option<&str>, poison: bool) -> String {
    let exe = std::env::current_exe().unwrap();
    let mut cmd = std::process::Command::new(exe);
    cmd.args(["--exact", "hw_ccd_lanes_child", "--ignored", "--nocapture"])
        .env("BLOOMERY_MT_CHILD_DUMP", "ccd-lanes");
    if poison {
        cmd.env("BLOOMERY_POISON", "1");
    }
    if let Some(threads_env) = threads_env {
        cmd.env("BLOOMERY_THREADS", threads_env);
    }
    let out = cmd.output().expect("re-exec of this test binary");
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "child at BLOOMERY_THREADS={threads_env:?} failed\n--- stdout ---\n{}\n--- stderr ---\n{err}",
        String::from_utf8_lossy(&out.stdout)
    );
    err
}

#[test]
#[ignore = "hw: needs the box; reads no model file; run via `just gate-mt`"]
fn hw_ccd_lanes_gate() {
    // The last child runs at the machine's own width and the poison lever's default.
    for (threads_env, poison) in [
        (Some("8"), true),
        (Some("30"), true),
        (None, true),
        (None, false),
    ] {
        let err = ccd_child(threads_env, poison);
        eprint!(
            "{}",
            err.lines()
                .filter(|l| l.starts_with("ccd lanes"))
                .collect::<Vec<_>>()
                .join("\n")
                + "\n"
        );
    }
}
