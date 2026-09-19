//! The decision gate for round 1-4: does the assembled forward pick ik's token?
//!
//! The oracle is one 6-token sequence. That is enough to prove the arithmetic and not
//! enough to prove the decision — a chain whose logits drift by 6e-1 (measured in
//! `tests/forward.rs`) keeps the argmax on a confident prompt and can lose it on a
//! near-tie. So this runs thirty-two, at 2 to 11 tokens, and reports the tie margin
//! next to every verdict instead of only the verdict. Prompt 32 is 56 tokens on its own:
//! flash attention walks keys in blocks of 32, and only a row that sees more than 32
//! allowed keys reaches the M-bump rescale where ik uses glibc `expf` and we use
//! `f32::exp`. Nothing shorter can touch that branch.
//!
//! Both sides read the same token ids from `tools/ref/prompts.tsv`; neither tokenizes.
//! ik's answers come from `$BLOOMERY_DATA/argmax-ik.tsv`, written by
//! `tools/ref/argmax.sh` (`just argmax-ref`), which is a different binary from the
//! oracle dumper on purpose — see `tools/ref/argmax_ref.cpp`.
//!
//! `hw_` prefix: needs the box, the model file and ik's answers.
use model::forward::{argmax, forward_trace};

fn model_path() -> String {
    std::env::var("BLOOMERY_MODEL")
        .unwrap_or_else(|_| "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf".into())
}

struct Row {
    id: usize,
    text: String,
    tokens: Vec<u32>,
}

struct IkAnswer {
    n_tokens: usize,
    argmax: u32,
    top5: Vec<u32>,
    logits: Vec<f32>,
}

fn read_prompts() -> Vec<Row> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/ref/prompts.tsv")
        .canonicalize()
        .expect("tools/ref/prompts.tsv must be in the tree");
    let text = std::fs::read_to_string(&path).unwrap();
    text.lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            Row {
                id: f[0].parse().unwrap(),
                text: f[1].to_string(),
                tokens: f[2].split(',').map(|s| s.trim().parse().unwrap()).collect(),
            }
        })
        .collect()
}

fn read_ik() -> Vec<IkAnswer> {
    let base = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into());
    let path = std::path::PathBuf::from(base).join("argmax-ik.tsv");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "no ik answers at {} ({e}). The lead produces them: `just argmax-ref`. \
             Do not compute them yourself and do not skip this test.",
            path.display()
        )
    });
    text.lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            IkAnswer {
                n_tokens: f[1].parse().unwrap(),
                argmax: f[2].parse().unwrap(),
                top5: f[3].split(',').map(|s| s.parse().unwrap()).collect(),
                logits: f[4].split(',').map(|s| s.parse().unwrap()).collect(),
            }
        })
        .collect()
}

/// Our own top-5 by the same rule `argmax_ref` uses: logit descending, id ascending.
fn top5(logits: &[f32]) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_by(|&a, &b| {
        let (la, lb) = (logits[a as usize], logits[b as usize]);
        lb.partial_cmp(&la).unwrap().then(a.cmp(&b))
    });
    idx.truncate(5);
    idx
}

/// Thirty-two prompts, one row each: ik's pick, ours, ik's own top1-top2 margin, and
/// the worst disagreement on the five logits ik reported.
///
/// The margin column is the point. A match on a prompt whose top two are 5 logits apart
/// is weak evidence; a match on one that is 0.1 apart is strong, because our measured
/// logit error is 6e-1 and could have moved it. Both appear in this set by construction.
#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/argmax-ik.tsv"]
fn hw_argmax_matches_ik_on_the_prompt_set() {
    let rows = read_prompts();
    let ik = read_ik();
    assert_eq!(rows.len(), 33, "the prompt set is 33 rows");
    assert_eq!(
        ik.len(),
        rows.len(),
        "ik answered a different number of prompts"
    );
    let g = gguf::Gguf::open(model_path()).unwrap();

    eprintln!(
        "{:>3} {:>3} {:>8} {:>8} {:>7} {:>10} {:>6}  prompt",
        "id", "n", "ik", "ours", "margin", "max|dlogit|", "top5"
    );
    let mut mismatched = Vec::new();
    let mut top5_differs = Vec::new();
    let mut worst_dlogit = 0.0f32;
    let mut total = std::time::Duration::ZERO;

    for (r, a) in rows.iter().zip(&ik) {
        assert_eq!(
            r.tokens.len(),
            a.n_tokens,
            "prompt {}: ik saw {} tokens, the file has {}",
            r.id,
            a.n_tokens,
            r.tokens.len()
        );
        let t0 = std::time::Instant::now();
        let tr = forward_trace(&g, &r.tokens).unwrap();
        total += t0.elapsed();

        let ours = argmax(&tr.logits.data);
        let mine5 = top5(&tr.logits.data);
        // ik's own margin: how far its winner is from its runner-up.
        let margin = a.logits[0] - a.logits[1];
        // The worst we are off on the five logits ik printed. Not the full 102400 -- ik
        // reported five -- but it is the part of the vector the decision turns on.
        let dlogit = a
            .top5
            .iter()
            .zip(&a.logits)
            .map(|(&id, &l)| (tr.logits.data[id as usize] - l).abs())
            .fold(0.0f32, f32::max);
        worst_dlogit = worst_dlogit.max(dlogit);

        let same5 = mine5 == a.top5;
        eprintln!(
            "{:>3} {:>3} {:>8} {:>8} {:>7.3} {:>10.4} {:>6}  {}",
            r.id,
            r.tokens.len(),
            a.argmax,
            ours,
            margin,
            dlogit,
            if same5 { "same" } else { "DIFF" },
            r.text
        );
        if ours != a.argmax {
            mismatched.push((r.id, a.argmax, ours, margin));
        }
        if !same5 {
            top5_differs.push((r.id, a.top5.clone(), mine5));
        }
    }

    let n = rows.len() as u32;
    eprintln!(
        "\n{n} prompts in {:?} ({:?} each); worst |logit - ik| over ik's top-5 = {:.4}",
        total,
        total / n,
        worst_dlogit
    );
    eprintln!(
        "argmax: {}/{n} match; top-5 order: {}/{n} identical",
        n as usize - mismatched.len(),
        n as usize - top5_differs.len()
    );
    for (id, want, got, margin) in &mismatched {
        eprintln!("  prompt {id}: ik {want}, ours {got}, ik's margin was {margin:.3}");
    }
    for (id, want, got) in &top5_differs {
        eprintln!("  prompt {id} top-5: ik {want:?}, ours {got:?}");
    }

    // The contract: 31 of 33, and the ones that differ are named, not tolerated in bulk.
    //
    // Measured 2026-09-20, after the Q3_K fused wiring (MUL-21): `ops::matmul_q` routes
    // Q3_K through `crates/qdot`, which is MORE accurate than the f32 round trip it
    // replaced, so every Q3_K logit moves and the near-ties move with them. The
    // divergence set moved from {5} to {14, 24}:
    //   * prompt 5 (`def add(a, b):`) matches again (ik 185, ours 185). It was the
    //     2026-09-19 divergence, ik margin 0.262; the fused path moved our logit onto
    //     ik's pick.
    //   * prompt 14 (`A right triangle has`): ik 245, ours 9226, margin 0.108,
    //     |dlogit| 0.285 -- a straight 1st/2nd swap.
    //   * prompt 24 (`Machine learning models are trained on`): ik 245, ours 1191,
    //     margin 0.151, |dlogit| 0.339 -- same 1st/2nd swap.
    // Both flips sit far inside the drift this same run measured (worst |dlogit| 0.994),
    // the same class as the old prompt 5: near-ties the inherited error can move, not
    // different computations. The round spec's repin rule is met (≤ 2 mismatches, every
    // flipped prompt at ik margin < 1.0): two flips, margins 0.108 and 0.151.
    //
    // Where the drift is from was already documented and is unchanged in kind: `q_nope2`'s
    // activation-code tie flips in `attn.rs`'s `quantize_act` (its own gate measures with
    // a 3.8e-6 exact-input companion -- the flips come from the input, not from that
    // stage's arithmetic). The wiring relocated the flips, it did not remove them.
    //
    // The set is pinned rather than counted. A third prompt flipping fails; one of these
    // starting to match also fails, because that means something moved and the round that
    // moved it should say what. Do not widen this to `len() <= 2`.
    const KNOWN_DIVERGENCE: &[usize] = &[14, 24];
    let ids: Vec<usize> = mismatched.iter().map(|(id, ..)| *id).collect();
    assert_eq!(
        ids,
        KNOWN_DIVERGENCE,
        "the divergence set moved: {} of 32 prompts chose a different token",
        mismatched.len()
    );
    // The top-5 ORDER is reported but not gated. ik's 4th and 5th are routinely within
    // 2e-3 of each other (prompt 0: 25.076096 vs 25.074284) and our logits carry 6e-1 of
    // inherited drift, so gating the order would be gating noise. A reordering inside the
    // tail is not a decode difference; a changed argmax is, and that is what fails above.
}
