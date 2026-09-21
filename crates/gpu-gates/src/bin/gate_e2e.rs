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
//! - (4) every flash stage-doubling arm (`StepProbe::keyaxis_arms`) writes
//!   the base path's tokens, on this prompt set and on a deeper prompt, at
//!   the same node count. Those arms are the ruler the attention rounds
//!   read their stage prices off, and the reading is only a stage's price
//!   while nothing else moves: identical tokens mean identical routing, so
//!   the rest of the step is the same work. This is where the probes'
//!   `fma(jig, second, first)` fold is pinned as the no-op it claims to be.
//! - (5) a cache prepared by `GpuModel::seed_depth` leaves the next step at
//!   the same position and live key count a decoded prompt of the same
//!   depth does, at the same captured node count — the contract
//!   `generate --seed-depth` rests on. State, not time.
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
use bloomery_gpu::model::{StepMode, StepProbe};
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

/// The keyaxis arms' deeper case: a prompt whose live keys reach past one
/// flash segment, so the merge folds more than one partial and a probe's
/// second pass wraps inside a whole segment instead of a short first one.
/// `DEEP_PROMPT + DEEP_GEN` stays inside `CTX_MAX`.
#[cfg(feature = "gpu")]
const DEEP_PROMPT: usize = 200;
#[cfg(feature = "gpu")]
const DEEP_GEN: usize = 16;

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
///
/// PIN(2026-09-21, lfold round): 755 → 728. The attention half's two
/// `gemv_q3k_heads` launches became one `gemv_q3k_heads_pair` over all
/// sixteen heads, one launch fewer in every layer. Derivation on the new
/// pins (`NODES_BLOCK0` = 20, `NODES_LAYER1` = 25, each plus one merge on a
/// split cache): (20 + 1) + 26 × (25 + 1) + 27 residual copies + the head's
/// 4 = 21 + 676 + 27 + 4 = 728. At `--ctx 64`, where no merge runs:
/// 20 + 650 + 27 + 4 = 701.
///
/// PIN(2026-09-21, lfold round): 728 → 701. The attention half's two
/// `quantize_q8_1(kqvc_*)` launches became one, again one launch fewer in
/// every layer. Derivation on the new pins (`NODES_BLOCK0` = 19,
/// `NODES_LAYER1` = 24, each plus one merge on a split cache):
/// (19 + 1) + 26 × (24 + 1) + 27 + 4 = 20 + 650 + 27 + 4 = 701. Both merges
/// carry a value-neutral rollback lever (`StepProbe::split_heads`,
/// `split_kqvc`). Since the fmerge round the levers nest, so restoring the
/// original 755 takes all four of them; measured at `--ctx 256` (lead,
/// 2026-09-22): none 648, `split_heads`+`split_flash_quant` **702**,
/// plus `split_kqvc` 729, plus `split_moe_quant` **755**. The tokens are
/// identical to the default path's at every one of those counts.
/// `split_kqvc` alone does nothing — there is no pair launch to split until
/// `split_flash_quant` restores one — so `generate` refuses that
/// combination rather than reading as a broken lever.
///
/// PIN(2026-09-22, fmerge round): 701 → 648. Two launches left every
/// routed layer and one left block 0: the `kqvc` q8_1 quantization became a
/// side output of the attention launch (every layer), and the MoE half's
/// two quantize launches became one `moe_quantize_pair` (routed layers
/// only). Derivation on the new pins (`NODES_BLOCK0` = 18,
/// `NODES_LAYER1` = 22, each plus one merge on a split cache — `CTX_MAX`
/// 256 is split): (18 + 1) + 26 × (22 + 1) + 27 residual copies + the
/// head's 4 = 19 + 598 + 27 + 4 = 648. FAIL-first held: the same source
/// with this constant still 701 printed `graph graph_nodes=648 want=701
/// eager_vs_graph_identical=true FAIL`, with the token classes and the
/// determinism arm unchanged. Both new merges carry a value-neutral
/// rollback lever (`StepProbe::split_flash_quant`, `split_moe_quant`).
#[cfg(feature = "gpu")]
const NODES_CHAIN: usize = 648;

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

/// A fixed pseudo-random prompt of `n` ids: BOS, then an LCG walk over the
/// id range the depth runners feed. Random ids are less kind to the caches
/// than real text, which is what the deep arm wants of them.
#[cfg(feature = "gpu")]
fn lcg_prompt(n: usize) -> Vec<u32> {
    let mut s: u64 = 12345;
    let mut out = vec![100000u32];
    while out.len() < n {
        s = s.wrapping_mul(1103515245).wrapping_add(12345) % 2147483648;
        out.push(1000 + (s % 90000) as u32);
    }
    out
}

/// One continuation of `prompt`: fresh caches, the prompt fed one token at
/// a time, then `DEEP_GEN` - 1 feedback steps.
#[cfg(feature = "gpu")]
fn run_deep(model: &mut GpuModel, prompt: &[u32]) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    model.reset()?;
    let mut next = model.step(prompt)?;
    let mut seq = Vec::with_capacity(DEEP_GEN);
    seq.push(next);
    for _ in 1..DEEP_GEN {
        next = model.step(&[next])?;
        seq.push(next);
    }
    Ok(seq)
}

/// The first place two token tables differ, as `(prompt index, token
/// index)` — what a failing arm needs to say about itself.
#[cfg(feature = "gpu")]
fn first_diff(a: &[Vec<u32>], b: &[Vec<u32>]) -> Option<(usize, usize)> {
    a.iter().zip(b).enumerate().find_map(|(i, (x, y))| {
        (0..x.len().min(y.len()))
            .find(|&k| x[k] != y[k])
            .map(|k| (i, k))
    })
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

    // Cross-check, ASSERTED since the ikclear round (2026-09-21). It used to
    // be printed only, on the premise that the two files were ik runs at
    // different context sizes (`argmax.sh` sized `-c` as `512 + GEN`) and so
    // were not the same quantity. That premise was wrong twice over: the
    // context size makes no difference here (regenerating the greedy file at
    // -c 544 is byte-identical to -c 512), and the real cause of the skew was
    // ik widening a sequence's first graph to all 64 experts whenever it
    // matched its warmup predicate — which the argmax file hit on every row
    // and the greedy file only on row 0. `argmax_ref` now primes the context
    // so neither does, and the two files agree by construction: same model,
    // same prompts, same first token, same top-5. Any future regeneration
    // that reintroduces the skew has hit that class of bug again, and this
    // gate is where it should stop. Derivation of the threshold: zero, not a
    // margin — both files are the same greedy argmax of the same logits, so
    // any difference at all is a defect in how they were produced.
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
    let refs_agree = tok_diff.is_empty() && top5_diff == 0;
    println!(
        "sanity rows={} argmax_vs_greedy_step1_token_diff={} top5_set_diff={top5_diff} {}",
        reference.len(),
        tok_diff.len(),
        if refs_agree { "ok" } else { "FAIL" }
    );
    for (id, a, g) in &tok_diff {
        println!("  sanity prompt {id}: argmax file {a}, greedy file step 1 {g}");
    }
    if !refs_agree {
        println!(
            "  the two reference files disagree — they are the same quantity, so one of them \
             was produced under a different graph shape (see tools/ref/argmax_ref.cpp's \
             priming block). Regenerate both with `just argmax-ref` before trusting either."
        );
        ok = false;
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

    // (4) The flash stage-doubling arms against the base path they are read
    // against. Every arm runs the whole set and the deeper prompt in graph
    // mode; `base` runs too, which also says the graph arm above reproduces
    // itself after a probe rebuild.
    let deep_prompt = lcg_prompt(DEEP_PROMPT);
    let mut deep_base: Vec<u32> = Vec::new();
    for (name, probe) in StepProbe::keyaxis_arms() {
        model.set_probe(probe)?;
        let nodes = model.capture_step()?;
        let set = run_set(&mut model, &prompts, &reference)?;
        let deep = run_deep(&mut model, &deep_prompt)?;
        if name == "base" {
            deep_base = deep.clone();
        }
        let set_same = set == graph;
        let deep_same = deep == deep_base;
        let pass = set_same && deep_same && nodes == NODES_CHAIN;
        println!(
            "keyaxis arm={name} graph_nodes={nodes} set_identical={set_same} \
             deep_identical={deep_same} {}",
            if pass { "ok" } else { "FAIL" }
        );
        if !pass {
            ok = false;
            if let Some((i, k)) = first_diff(&set, &graph) {
                println!(
                    "  prompt {} token {k}: arm {} vs base {}",
                    reference[i].id, set[i][k], graph[i][k]
                );
            }
            if let Some(k) = (0..deep.len().min(deep_base.len())).find(|&k| deep[k] != deep_base[k])
            {
                println!(
                    "  deep prompt token {k}: arm {} vs base {}",
                    deep[k], deep_base[k]
                );
            }
            if nodes != NODES_CHAIN {
                println!("  the arm captured {nodes} nodes, the pin is {NODES_CHAIN}");
            }
        }
    }
    model.set_probe(StepProbe::default())?;
    // (5) A prepared cache stands where a decoded prompt stands.
    // `GpuModel::seed_depth` fills KV rows directly so a deep step can be
    // timed without decoding a prompt into them; that is only an instrument
    // if the step which follows has the same shape. Asserted as state, not
    // as time: the captured node count, and the position and live key count
    // the launches actually read. The seeded rows are a deterministic
    // pattern and not the model's own keys, so the two arms' VALUES are not
    // comparable and are not compared — this arm pins the shape alone.
    //
    // The twin is `SEED_D` seeded rows plus one token against `SEED_D + 1`
    // decoded tokens: both leave the last `refresh_params` at position
    // `SEED_D`, so all three numbers must read `SEED_D + 1`, `SEED_D`,
    // `SEED_D + 1`. The node count is a property of `CTX_MAX` rather than of
    // the cache contents, so it is the weaker half of the claim; it is here
    // because a seeded run that captured a different chain would not be the
    // chain the timing arm means to measure.
    const SEED_D: usize = 128;
    let seed_token = prompts[0].tokens[0];
    model.reset()?;
    model.seed_depth(SEED_D)?;
    let seed_nodes = model.capture_step()?;
    model.step(&[seed_token])?;
    let (seed_pos_buf, seed_keys) = model.device_step_params()?;
    let seeded = (model.pos(), seed_pos_buf, seed_keys);

    model.reset()?;
    let twin = vec![seed_token; SEED_D + 1];
    model.step(&twin)?;
    let (twin_pos_buf, twin_keys) = model.device_step_params()?;
    let decoded = (model.pos(), twin_pos_buf, twin_keys);

    let want = (SEED_D as u32 + 1, SEED_D as u32, SEED_D as u32 + 1);
    let seed_ok = seeded == want && decoded == want && seed_nodes == NODES_CHAIN;
    println!(
        "seed depth={SEED_D} graph_nodes={seed_nodes} want_nodes={NODES_CHAIN} \
         seeded(pos,pos_buf,n_keys)={seeded:?} decoded={decoded:?} want={want:?} {}",
        if seed_ok { "ok" } else { "FAIL" }
    );
    if !seed_ok {
        ok = false;
    }

    if ok {
        println!(
            "gate_e2e: PASS — the whole chain picks the greedy reference's first token on \
             every prompt or misses it only inside MARGIN_FLOOR; graph replay equals eager \
             token for token at the pinned node count; two eager runs are identical; every \
             flash stage-doubling arm writes the base path's tokens at that same node count; a \
             seeded cache leaves the step at the same position and key count a decoded \
             prompt does; and the two reference files agree with each other."
        );
        Ok(())
    } else {
        Err("gate_e2e: FAIL — see the lines marked FAIL above".into())
    }
}
