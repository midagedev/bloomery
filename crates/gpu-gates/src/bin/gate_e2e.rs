//! The end-to-end gate: the whole GPU chain — 27 layers, the lm_head and
//! the argmax — against ik CUDA's greedy answers on the 33-prompt set, and
//! the graph replay against the eager body it was captured from.
//!
//! Named `gate_e2e`, not `gate_pN`: the `P` numbers are the kernel packages
//! of `docs/gpu-design.md` and P9/P10 are taken by live gates. This one
//! gates an assembly, not a kernel.
//!
//! Each prompt runs on fresh caches (`GpuModel::reset`), its tokens fed one
//! position at a time through the decode path — there is no prefill kernel
//! and the reference was written the same way (`argmax.sh --step-prefill`).
//! The prompt's own `step` yields generated token 0; `GEN` - 1 feedback
//! steps follow.
//!
//! What is asserted:
//! - (0) reference sanity, STRUCTURAL only: the argmax file and the greedy
//!   file cover the same prompt ids with the same token counts as
//!   `prompts.tsv`. Their step-1 tokens are cross-checked and PRINTED, not
//!   asserted: the two files are two ik runs at different context sizes
//!   (`tools/ref/argmax.sh` sizes `-c` as `512 + GEN`), and they disagree on
//!   two prompts and on the top-5 logits of nearly every row. Which of the
//!   two is right is the reference tooling's question, not this gate's, so
//!   this gate names the disagreement and rests its verdict on the greedy
//!   file alone — the only one of the two that carries continuations.
//! - (1) no prompt is `diverged` at index 0: the chain produces ik's first
//!   token, or the reference's own top-2 margin there is under
//!   `MARGIN_FLOOR` and the flip is a near-tie re-lottery. Near ties are
//!   printed, not failed — the criterion `crates/model/tests/prompts.rs`
//!   holds for the CPU chain, carried to the GPU chain. Divergences past
//!   index 0 are printed and not failed: pinning the divergence set is A3's
//!   later round and needs three runs first.
//! - (2) graph mode reproduces the eager sequence token for token on every
//!   prompt — the eager-equals-replay arm every step gate has.
//! - (3) determinism: two eager runs give identical tables.
//!
//! The classifier, the reference reader and the three classes are
//! `gpu_gates::prompts`; this gate does not re-implement them.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_e2e: built without the `gpu` feature; see `just gate-gpu-e2e`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
use bloomery_gpu::GpuModel;
#[cfg(feature = "gpu")]
use bloomery_gpu::model::StepMode;
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::open_model;
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::prompts::{GreedyClass, GreedyRow, compare_greedy, read_greedy};

/// Generated tokens per prompt — the reference file's own width
/// (`greedy-ik-cuda-32.tsv`, written with `BLOOMERY_REF_GEN=32`).
#[cfg(feature = "gpu")]
const GEN: usize = 32;

/// Cache rows per prompt: the longest prompt of the set is 56 tokens and
/// `GEN` more follow, so this is room with margin and no re-allocation
/// between prompts.
#[cfg(feature = "gpu")]
const CTX_MAX: usize = 256;

/// PIN(2026-09-21): the reference top1-top2 margin below which a first
/// difference is a near-tie re-lottery rather than a fault. Derivation: the
/// A2t round measured the reference's own margin distribution over this set
/// — 216 of 1056 margins under 0.5, 43 under 0.1 — and ik's CPU and CUDA
/// backends disagree with each other on 2 of the 33 prompts (5 and 24),
/// both inside this floor. 0.5 is also the floor `prompts.rs`'s own
/// classifier tests are written around. It is a classification threshold,
/// not a band: raising it hides faults, and it is the one number this gate
/// owns.
#[cfg(feature = "gpu")]
const MARGIN_FLOOR: f32 = 0.5;

/// `$BLOOMERY_DATA/<name>` — the reference files the box holds.
#[cfg(feature = "gpu")]
fn data_file(name: &str) -> std::path::PathBuf {
    let data = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".to_string());
    std::path::PathBuf::from(data).join(name)
}

/// PIN(2026-09-21): the whole chain captures exactly this many graph nodes
/// at `CTX_MAX` — a cache tall enough that the flash is cut into segments,
/// so every layer carries its merge pass. Derivation, against prof2's own
/// pins (`gate_p8`: `NODES_BLOCK0` = 21, `NODES_LAYER1` = 26, each plus one
/// merge when the cache is split): block 0 dense (21 + 1) + 26 routed
/// layers (26 + 1) each + 27 residual copies (one per layer boundary, the
/// last into the head's input) + the head's 4 = 22 + 702 + 27 + 4 = 755.
/// Measured 728 at `--ctx 64`, where no merge runs: 21 + 676 + 27 + 4. Not
/// a band — the node count is deterministic, the margin is zero. A debug
/// copy left in the chain, or a fused pair that stopped fusing, keeps every
/// other arm of this gate green and fails only here.
#[cfg(feature = "gpu")]
const NODES_CHAIN: usize = 755;

/// The argmax-only reference (`argmax-ik-cuda.tsv`): each data row's id,
/// its argmax and its top-5 ids. `gpu_gates::prompts::read_greedy` rejects
/// this five-column format on purpose — an empty `gen_ids` would compare as
/// vacuously identical — so the cross-check between the two reference files
/// needs these fields read here. Deliberately not a second general reader.
#[cfg(feature = "gpu")]
fn read_argmax_ids(
    path: &std::path::Path,
) -> Result<Vec<(usize, u32, Vec<u32>)>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("read_argmax_ids: cannot read {}: {e}", path.display()))?;
    let mut rows = Vec::new();
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 5 {
            return Err(format!(
                "read_argmax_ids: {}: row has {} fields, want the 5-column argmax format: {line:?}",
                path.display(),
                f.len()
            )
            .into());
        }
        rows.push((
            f[0].trim().parse::<usize>()?,
            f[2].trim().parse::<u32>()?,
            f[3].split(',')
                .map(|t| t.trim().parse::<u32>())
                .collect::<Result<Vec<_>, _>>()?,
        ));
    }
    if rows.is_empty() {
        return Err(format!("read_argmax_ids: no rows in {}", path.display()).into());
    }
    Ok(rows)
}

/// One greedy continuation per reference row, in file order: fresh caches,
/// the prompt fed one token at a time, then `GEN` - 1 feedback steps. The
/// prompt token ids come from the reference rows' own set so both engines
/// answer the same question; `reference[i].n_tokens` is ik's count and is
/// cross-checked against the file's.
#[cfg(feature = "gpu")]
fn run_set(
    model: &mut GpuModel,
    prompts: &[bloomery_gpu_gates::prompts::PromptRow],
    reference: &[GreedyRow],
) -> Result<Vec<Vec<u32>>, Box<dyn std::error::Error>> {
    let mut out = Vec::with_capacity(reference.len());
    for (r, p) in reference.iter().zip(prompts) {
        if r.id != p.id || r.n_tokens != p.tokens.len() {
            return Err(format!(
                "run_set: prompt {} has {} tokens, the reference row {} says {}",
                p.id,
                p.tokens.len(),
                r.id,
                r.n_tokens
            )
            .into());
        }
        model.reset()?;
        let mut next = model.step(&p.tokens)?;
        let mut seq = Vec::with_capacity(GEN);
        seq.push(next);
        for _ in 1..GEN {
            next = model.step(&[next])?;
            seq.push(next);
        }
        out.push(seq);
    }
    Ok(out)
}

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut ok = true;
    let prompts_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/ref/prompts.tsv");
    let prompts = bloomery_gpu_gates::prompts::read_prompts(&prompts_path)?;
    let reference = read_greedy(&data_file("greedy-ik-cuda-32.tsv"))?;
    let argmax_only = read_argmax_ids(&data_file("argmax-ik-cuda.tsv"))?;

    // (0) Structural sanity, asserted: the three files describe the same
    // prompt set. Anything below this rests on the rows lining up.
    if argmax_only.len() != reference.len() || reference.len() != prompts.len() {
        return Err(format!(
            "reference sanity: {} argmax rows, {} greedy rows, {} prompts — all three must \
             be the same set",
            argmax_only.len(),
            reference.len(),
            prompts.len()
        )
        .into());
    }
    for ((id, _, _), r) in argmax_only.iter().zip(&reference) {
        if *id != r.id {
            return Err(format!(
                "reference sanity: argmax row {id} against greedy row {}",
                r.id
            )
            .into());
        }
    }

    // Cross-check, PRINTED not asserted. The two files are two ik runs at
    // different context sizes — `argmax.sh` sizes `-c` as `512 + GEN`, so
    // the argmax file ran at 512 and the greedy file at 544 — and they are
    // not the same quantity. Which one is right is the reference tooling's
    // question; this gate records the distance and judges against the
    // greedy file, the only one carrying continuations.
    let tok_diff: Vec<_> = argmax_only
        .iter()
        .zip(&reference)
        .filter(|((_, a, _), r)| *a != r.gen_ids[0])
        .map(|((id, a, _), r)| (*id, *a, r.gen_ids[0]))
        .collect();
    let top5_diff = argmax_only
        .iter()
        .zip(&reference)
        .filter(|((_, _, t5), r)| *t5 != r.top5)
        .count();
    println!(
        "sanity rows={} argmax_vs_greedy_step1_token_diff={} top5_set_diff={top5_diff} \
         (printed, not asserted — the two files are ik runs at -c 512 and -c 544)",
        reference.len(),
        tok_diff.len()
    );
    for (id, a, g) in &tok_diff {
        println!("  sanity prompt {id}: argmax file {a}, greedy file step 1 {g}");
    }

    let gguf = open_model()?;
    let mut model = GpuModel::load_full(&gguf, CTX_MAX)?;
    println!(
        "resident bytes={} ctx_max={CTX_MAX} layers=0..{} gen={GEN}",
        model.resident_bytes(),
        model.stages()[0].layers().end
    );

    // Eager first: the correctness path the graph is checked against.
    model.set_mode(StepMode::Eager);
    let eager = run_set(&mut model, &prompts, &reference)?;
    let report = compare_greedy(&eager, &reference, MARGIN_FLOOR);
    println!(
        "{:>3} {:>10} {:>11} {:>10}  prompt",
        "id", "class", "first_diff", "ref_margin"
    );
    for row in &report.rows {
        let class = match row.class {
            GreedyClass::Identical => "identical",
            GreedyClass::NearTie => "near_tie",
            GreedyClass::Diverged => "diverged",
        };
        let diff = row
            .first_diff
            .map_or_else(|| "-".to_string(), |i| i.to_string());
        let margin = row
            .ref_margin
            .map_or_else(|| "-".to_string(), |m| format!("{m:.3}"));
        println!(
            "{:>3} {class:>10} {diff:>11} {margin:>10}  {}",
            row.id,
            prompts
                .iter()
                .find(|p| p.id == row.id)
                .map_or("", |p| p.text.as_str())
        );
    }
    println!(
        "classes identical={} near_tie={} diverged={}",
        report.n_identical, report.n_near_tie, report.n_diverged
    );

    // (1) The decision criterion: the chain must produce ik's FIRST token,
    // or miss it only where the reference itself was inside the floor. A
    // divergence later in the continuation is a different question — the two
    // sides are conditioned on different text from there — and pinning that
    // set is a later round.
    let bad_at_zero: Vec<&_> = report
        .rows
        .iter()
        .filter(|r| r.class == GreedyClass::Diverged && r.first_diff == Some(0))
        .collect();
    println!(
        "first_token diverged_at_index0={} {}",
        bad_at_zero.len(),
        if bad_at_zero.is_empty() { "ok" } else { "FAIL" }
    );
    for r in &bad_at_zero {
        println!(
            "  prompt {}: our first token differs where ik's margin was {:?}",
            r.id, r.ref_margin
        );
        ok = false;
    }

    // (3) Determinism: the same binary, the same inputs, twice.
    let eager2 = run_set(&mut model, &prompts, &reference)?;
    let deterministic = eager2 == eager;
    println!(
        "determinism two_eager_runs_identical={deterministic} {}",
        if deterministic { "ok" } else { "FAIL" }
    );
    if !deterministic {
        ok = false;
    }

    // (2) eager == replay. The capture happens on the first `step` in graph
    // mode; the node count is a deterministic property of the chain and is
    // printed beside it.
    model.set_mode(StepMode::Graph);
    let nodes = model.capture_step()?;
    let graph = run_set(&mut model, &prompts, &reference)?;
    let same = graph == eager;
    let nodes_pinned = nodes == NODES_CHAIN;
    println!(
        "graph graph_nodes={nodes} want={NODES_CHAIN} eager_vs_graph_identical={same} {}",
        if same && nodes_pinned { "ok" } else { "FAIL" }
    );
    if !nodes_pinned {
        println!("  the chain captured {nodes} nodes, the pin is {NODES_CHAIN}");
        ok = false;
    }
    if !same {
        ok = false;
        for (i, (g, e)) in graph.iter().zip(&eager).enumerate() {
            if g != e {
                let at = (0..g.len().min(e.len())).find(|&k| g[k] != e[k]);
                println!("  prompt {}: first difference at {at:?}", reference[i].id);
            }
        }
    }

    if ok {
        println!(
            "gate_e2e: PASS — the whole chain picks the greedy reference's first token on \
             every prompt or misses it only inside MARGIN_FLOOR; graph replay equals eager \
             token for token at the pinned node count; two eager runs are identical. The \
             two reference files' own disagreement is printed above, not gated."
        );
        Ok(())
    } else {
        Err("gate_e2e: FAIL — see the lines marked FAIL above".into())
    }
}
