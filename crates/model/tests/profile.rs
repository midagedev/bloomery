//! Gate for the profiler: the instrument must not become a participant.
//!
//! Three assertions, in one test on purpose (see the `set_var` note below):
//!
//!   1. **Bit identity.** The same prompt through `forward`, once with profiling off
//!      and once at level 2 — the level that adds the most code to the hot loops —
//!      must produce identical logits, `max|diff| == 0.0`. No tolerance: a tolerance
//!      here would let every future hook edit hide a rounding it caused.
//!   2. **Coverage.** The instrumented sites must account for ≥ 98 % of one decode
//!      step's wall time. This is the gate's body — an unhooked hot loop shows up
//!      here as a hole, not as a wrong number.
//!   3. **Every site recorded.** Every hooked site — the three matmul-shaped ones
//!      and the coverage round's typeless sites — has `calls > 0` after a step, so
//!      a hook that silently stopped firing cannot pass on the other two.
//!
//! The level comes from `BLOOMERY_PROFILE`, read once per process into a `OnceLock` —
//! a process cannot be both profiled and unprofiled. Assertion 1 therefore runs the
//! two forwards in re-exec'd children of this very test binary (each child gets its
//! own `OnceLock` and its own level), dumping raw logits to a file the parent
//! compares. Assertions 2 and 3 need the in-process accumulators, so after the
//! children return the parent sets the variable and runs one step itself.
//!
//! `hw_` prefix: needs the box and the model file. It does NOT need the oracle to
//! hold anything specific — the oracle is used only as the shared prompt, the same
//! one every other gate runs, so the bit-identity claim is about the prompt that
//! matters.
#[path = "common/oracle.rs"]
mod oracle;

use model::forward::{argmax, forward, new_cache, step};
use model::profile;
use std::time::Instant;

fn model_path() -> String {
    std::env::var("BLOOMERY_MODEL")
        .unwrap_or_else(|_| "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf".into())
}

/// The re-exec entry point. Not a test of its own: when `BLOOMERY_PROFILE_CHILD_DUMP`
/// is absent (i.e. someone ran the file directly) it returns without doing anything —
/// all assertions live in [`hw_profile_gate`].
#[test]
#[ignore = "hw: re-exec child of hw_profile_gate; standalone it is a no-op"]
fn hw_profile_child_logits() {
    let Ok(dump) = std::env::var("BLOOMERY_PROFILE_CHILD_DUMP") else {
        eprintln!("child helper: no BLOOMERY_PROFILE_CHILD_DUMP, nothing to do");
        return;
    };
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(model_path()).unwrap();
    let tokens: Vec<u32> = o.tokens.iter().map(|&t| t as u32).collect();
    let logits = forward(&g, &tokens).unwrap();
    let bytes: Vec<u8> = logits.data.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&dump, bytes).unwrap();
    eprintln!(
        "child: {} logits (BLOOMERY_PROFILE={:?}) dumped to {dump}",
        logits.data.len(),
        std::env::var("BLOOMERY_PROFILE").ok()
    );
}

/// Run `forward` on the shared prompt in a child process at the given profile level
/// (absent = off) and return its logits. Each child is a fresh `OnceLock`, which is
/// the only way one process family can show both levels.
fn child_logits(profile_level: Option<&str>) -> Vec<f32> {
    let exe = std::env::current_exe().unwrap();
    let dump = std::env::temp_dir().join(format!(
        "bloomery-profile-child-{}-{}.f32",
        std::process::id(),
        profile_level.unwrap_or("off")
    ));
    let mut cmd = std::process::Command::new(exe);
    cmd.args([
        "--exact",
        "hw_profile_child_logits",
        "--ignored",
        "--nocapture",
    ])
    .env("BLOOMERY_PROFILE_CHILD_DUMP", &dump);
    match profile_level {
        Some(l) => {
            cmd.env("BLOOMERY_PROFILE", l);
        }
        None => {
            cmd.env_remove("BLOOMERY_PROFILE");
        }
    }
    let out = cmd.output().expect("re-exec of this test binary");
    assert!(
        out.status.success(),
        "child at BLOOMERY_PROFILE={profile_level:?} failed\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let bytes = std::fs::read(&dump).unwrap();
    let (words, _) = bytes.as_chunks::<4>();
    words
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The three assertions, one test so that the `set_var` below cannot race another
/// test's `env::var`. The recipe runs this file with `--test-threads=1`; the only
/// other test here is the child helper, which by then has long returned and touches
/// neither the environment nor the profiler.
#[test]
#[ignore = "hw: needs the box and the model file; run via `just gate-profile`"]
fn hw_profile_gate() {
    // 1. Bit identity: off vs level 2 (the level that adds Instant pairs inside the
    //    row loops, i.e. the maximum code a hook edit can put on the hot path).
    let off = child_logits(None);
    let on = child_logits(Some("2"));
    assert_eq!(
        off.len(),
        on.len(),
        "logit count {} vs {}",
        off.len(),
        on.len()
    );
    let worst = off
        .iter()
        .zip(&on)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        worst,
        0.0,
        "profiling changed the logits: max|diff| {worst:e} over {} values",
        off.len()
    );
    eprintln!("logits, off vs profiled level 2      max|diff| = 0   exact");

    // Assertions 2 and 3 need the in-process accumulators. The children were separate
    // processes, so this process has not called profile::level() yet and the OnceLock
    // still reads the environment — set here, before the first model code runs.
    //
    // SAFETY: this is the only active test in the binary at this point (the recipe
    // serializes with --test-threads=1 and the child helper has already returned),
    // so no thread can read the environment while this writes it.
    unsafe { std::env::set_var("BLOOMERY_PROFILE", "2") };

    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(model_path()).unwrap();
    let tokens: Vec<u32> = o.tokens.iter().map(|&t| t as u32).collect();

    // The prefill fills the cache; its profile lands in the accumulators too, so it
    // is dropped — the coverage claim is about one decode step, not a mixture.
    // `Derived::new` runs unprofiled (it has no hooks), before the first step.
    let mut cache = new_cache(&g).unwrap();
    let derived = model::derived::Derived::new(&g).unwrap();
    let logits = step(&g, &tokens, &mut cache, &derived).unwrap();
    profile::reset();

    // One decode step: one token in, wall-clock around the whole `step`.
    let next = argmax(&logits.data);
    let t0 = Instant::now();
    step(&g, &[next], &mut cache, &derived).unwrap();
    let wall_ns = t0.elapsed().as_nanos() as u64;

    // 2. Coverage. Integer arithmetic on purpose — no float equality near a gate.
    //
    // Raised 80 -> 98 on 2026-09-20 by the coverage round: after the thread pool
    // and `Derived`, measured coverage fell to 91.6% (level-1 table, decode, 2
    // steps, wall 475.0 ms) — the ~20 ms/step the table could not name was
    // already larger than ik's whole 12.9 ms step, so the gate was told to
    // demand that the round hook it. With the typeless sites in place the same
    // gate measures 98.1% on this tree (the first green run's output, not a
    // target chosen in advance). 98 leaves 0.1 points of headroom — thin on
    // purpose: the unhooked remainder this round leaves behind is ~4.4 ms of
    // step glue per 233 ms step, so a healthy tree cannot drift far below, and
    // a site the size of `flash_attn_latent` (9.6 ms, 4.1%) going dark lands
    // near 94% and fails loud, not marginal.
    //
    // Lead re-ran it five times on the merged tree before committing: 98.04,
    // 98.06, 98.08, 98.09, 98.11 % over walls of 189 to 237 ms. The ratio does
    // not move with the wall — the unhooked remainder is step glue that scales
    // with the step, not a fixed overhead — so the 0.04 points of headroom at
    // the worst run are not a noise band waiting to flip. Anything that does
    // push this below 98 is new unhooked work, which is the thing the gate is
    // for. Do not lower the threshold to make such a run pass; hook the work.
    let instrumented = profile::instrumented_ns();
    let pct = instrumented as f64 / wall_ns as f64 * 100.0;
    assert!(
        instrumented * 100 >= wall_ns * 98,
        "coverage {pct:.1}% < 98%: an unhooked hot loop is eating the step \
         ({:.1} ms instrumented of {:.1} ms wall)",
        instrumented as f64 / 1e6,
        wall_ns as f64 / 1e6
    );
    eprintln!(
        "coverage                             {pct:.1}% of one decode step ({:.1} / {:.1} ms)",
        instrumented as f64 / 1e6,
        wall_ns as f64 / 1e6
    );

    // 3. Every hooked site did something. The list is every site the crate
    // records; a new hook that forgets to fire shows up here by name.
    for site in [
        "matmul_q",
        "q_nope2_absorbed",
        "embed",
        "rms_norm",
        "f32_tensor",
        "residual_add",
        "gain",
        "is_moe",
        "ffn_weights",
        "attn_params",
        "attn_latent",
        "attn_rope",
        "attn_kvr",
        "flash_attn_latent",
        "wv_b_heads",
        "moe_setup",
        "moe_route",
        "swiglu",
        "moe_expert_io",
        "moe_trace",
        "head_params",
    ] {
        let calls: u64 = profile::entries()
            .iter()
            .filter(|e| e.site == site)
            .map(|e| e.calls)
            .sum();
        assert!(calls > 0, "site {site} recorded nothing — its hook is dead");
    }
    eprintln!("sites                                all 21 hooked sites live");

    // The table itself, printed whatever the verdict — a coverage number without the
    // rows behind it cannot be acted on.
    eprintln!("{}", profile::report(wall_ns, "decode"));
}
