//! The GLM-5.3-Flash NextN draft's gate: the target opened by its NextN plan
//! on the gate card with the next-token layer beside it
//! (`Body::open_placed_nextn`, `place::PlanInputs::plan_nextn`), the NextN
//! program's walks (`bloomery_gpu_glm5next::nextn_walk`, `nextn_chain`)
//! held against ik's MTP draft set (refset family `mtp-glm5next`) and the
//! drafted windows (`app::mtp::MtpDraft<Body>`) against the plain run.
//!
//! What is asserted:
//! - (m) the set's MTP input: its warmup graph's `enorm` reads `inp_embd`
//!   itself, with no position-mask node between, and no row of `inp_embd`
//!   is zeros, position 0 included — the rule the program's embedding rows
//!   follow.
//! - (o) teacher-forced, every graph of the set replayed in ik's order — the
//!   warmup, then each block's graphs — each row's token, position and
//!   target hidden row (ik's `inp_tokens`, `inp_pos`, `inp_mtp_states`) fed
//!   from the host in runs of up to [`WALK_ROWS`] rows, the runs before the
//!   last store-only walks and the last one a chain: the chain's id is the
//!   argmax of our full head's logits, first index on a tie; on the graph's
//!   last row that argmax is ik's `result_output` argmax wherever ik's
//!   top-2 margin clears [`margin_cap`] at [`logits_band`], and ik's
//!   runner-up otherwise. Each block's proposal — the argmax of the graph
//!   before its update graph — is the set's draft token for the block, or
//!   that graph's tie. Every graph with ik's head on its last row holds our
//!   logits within [`logits_band`] of ik's (`‖ours − ik‖ / ‖ik‖`), the worst
//!   printed over the band. At least one graph is held off a tie.
//! - (p) the pairing: the hidden rows a walk fed the target's arenas reads
//!   (`bloomery_gpu_glm5next::nextn_hidden`, the walk's own gather): the
//!   set's first `CHUNK` prompt ids in a batch and by steps leave the same
//!   rows in the prompt-batch and step arenas, bit for bit (a batch of at
//!   most a chunk runs the step's gemvs; the whole prompt's distance between
//!   the two printed, a longer batch running the GEMM), and a prompt of
//!   [`GROUPED`] of its ids cycled (two batches) at groups of two leaves the
//!   rows it leaves at groups of one, bit for bit, every unit's rows read
//!   after the group; each value of every row of the whole prompt's
//!   within its derived bound of the f64 replica of the gather — the four
//!   streams' mean, then `output_norm`'s RMS at the file's eps — on the
//!   target streams the row was made from (`nextn_target_streams`; the
//!   bound is `HiddenRef`'s); the row at position `q` closer to ik's warmup
//!   row at `q + 1` than to its rows `q` and `q + 2`, each row's three
//!   distances printed, not held (no band in the tree derives a row past a
//!   flip, and the set carries no routing to tell the rows off every flip's
//!   path); a verify of two
//!   rows leaves in the pair arena the rows two plain steps leave, bit for
//!   bit, and keeps them past its commit and the next step; a verify whose
//!   commit keeps row 0 alone leaves that row readable and row 1 refused by
//!   name (the arena holds the kept position alone).
//! - (t) the target is the NextN load's own: the plain run of the set's
//!   prompt and [`N`] greedy ids on a load of the NextN plan's target plan
//!   without the layer (two KDA lanes, as the plan counts them), and the
//!   same on the NextN load, give the same ids and every step's logits bit
//!   for bit. ik's committed stream beside them is printed, not held.
//! - (w) the windows end to end, the prompt fed in batches and by steps:
//!   under `Speculative<MtpDraft<Body>, 2>` the drafted greedy ids are the
//!   plain run's on the same path for [`N`] tokens, with a rejected row and an accepted
//!   proposal among the windows — else either path never ran and the clause
//!   is red.
//! - (f) the walk's refusals by name: a walk on a load without the layer,
//!   and the MTP window opened over it (`MtpBody::head`), a captured walk, no
//!   rows, [`WALK_ROWS`] + 1 rows, a walk from past the positions the store
//!   holds, a token past the vocabulary, a host hidden slice of the wrong
//!   length, target rows past the arena's, and a chain with own walks, in
//!   the store walk's mode or into no place; each leaves the store's
//!   positions where they were. Then the target's rows by position
//!   (`nextn_hidden`, the walk's own gather): the pair arena before any
//!   verify, the step arena for a walk at position 0, and after a prompt of
//!   [`FED`] ids in one batch the prompt-batch arena's row past the batch's
//!   last and its rows read as other positions, each refused by name.
//! - (h) a fault in a chain poisons the model at once: the layer's
//!   `shared_head_norm` gain's first value set to NaN, a chain from a reset
//!   is a fault by name and leaves the model poisoned by that fault before
//!   any target call, and a walk after it is refused as poisoned; the gain
//!   put back and a reset, the same chain gives the clean chain's id.
//! - (s) a sequence state (`bloomery_gpu_glm5next::seq_save`) of the load
//!   without the layer, taken after its plain run, put back on the NextN
//!   load (`seq_resume`): the store layouts differ (the NextN layer's store
//!   rows), so it is refused by name before any copy, and the model stays at
//!   position 0, unpoisoned. The same file, card and context: the refusal is
//!   the layout's, not the identity's.

#[cfg(not(feature = "glm5next"))]
fn main() {
    eprintln!(
        "gate_glm5next_mtp: built without the `glm5next` feature; see `just gate-gpu-glm5next-mtp`."
    );
    std::process::exit(2);
}

#[cfg(feature = "glm5next")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_glm5next_mtp", gate::run())
}

#[cfg(feature = "glm5next")]
mod gate {
    use std::time::Instant;

    use app::Session;
    use app::mtp::MtpDraft;
    use bloomery_gpu::GpuError;
    use bloomery_gpu::host::PassKind;
    use bloomery_gpu::host::swap::Residency;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::weights::DevWeight;
    use bloomery_gpu_gates::rounding::q8_32_rel;
    use bloomery_gpu_gates::{Fnv1a64, GateError, checks_failed, patch_bytes, verdict};
    use bloomery_gpu_glm5next::{
        Body, CHUNK, GEMM_FROM, Glm5nextModel, GlmArena, GlmPromptSink, GlmSeq, NextnFeed,
        NextnHead, NextnHidden, NextnMode, PrefillMode, WALK_ROWS, feed, nextn_chain, nextn_hidden,
        nextn_logits, nextn_target_streams, nextn_walk, prompt_with, seq_resume, seq_save,
        set_prefill, set_prefill_group,
    };
    use gguf::Split;
    use model::arch::glm5next::names;
    use model::arch::glm5next::place::{KdaLanes, NextnInputs, PlanInputs};
    use model::placement::{PlanLevers, workstation};
    use refset::arch::glm5next::{MODEL, MTP, MTP_SET};
    use refset::ik::Layout;
    use refset::mtpref::{Graph, MtpSet};
    use runtime::{Advance as _, Committed, PassSink};

    /// Cache rows: the e2e gate's main load.
    const CTX: usize = 3136;
    /// (p)'s grouped prompt: two batches, the second short.
    const GROUPED: usize = 600;
    /// (f)'s prompt in one batch: its rows are the prompt-batch arena's.
    const FED: usize = 12;
    /// Generated ids a plain or drafted run holds: the set's.
    const N: usize = 64;
    /// The verify's rows: a proposal of one id and the token before it.
    const M: usize = 2;

    /// PIN(2026-10-01): the band of the NextN head's logits against ik's that
    /// (o) holds and its tie cap reads: on ik's own inputs (the token and the hidden row
    /// fed from the set), the projections in series where ik's CPU side reads
    /// 32-value q8 activations and ours f32 — `eh_proj`, the joined latent
    /// projection, the query's up projection, the key and value absorbs, the
    /// attention's output projection, the routed experts' gate·up and down,
    /// the shared expert's gate·up and down, and the head's projection:
    /// eleven terms of [`q8_32_rel`], √11 · 1.2858e-2 = 4.26e-2.
    fn logits_band() -> f64 {
        11f64.sqrt() * q8_32_rel()
    }

    /// The deviation of `v` about its mean: the spread a ranking reads.
    fn spread(v: &[f32]) -> f64 {
        let n = v.len().max(1) as f64;
        let mean = v.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
        (v.iter()
            .map(|&x| (f64::from(x) - mean).powi(2))
            .sum::<f64>()
            / n)
            .sqrt()
    }

    /// The widest gap between two of a row's values the errors can cross:
    /// `band` times the row's spread (a common offset moves no rank), three
    /// deviations each — the e2e gates' `margin_cap`.
    fn margin_cap(band: f64, v: &[f32]) -> f64 {
        6.0 * band * spread(v)
    }

    /// `‖a − b‖ / ‖b‖` in f64; infinite on a NaN or a length mismatch.
    fn rel(a: &[f32], b: &[f32]) -> f64 {
        if a.len() != b.len() {
            return f64::INFINITY;
        }
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            num += (f64::from(x) - f64::from(y)).powi(2);
            den += f64::from(y).powi(2);
        }
        let r = (num / den.max(f64::MIN_POSITIVE)).sqrt();
        if r.is_nan() { f64::INFINITY } else { r }
    }

    /// The first index of the largest value, and the largest of the rest:
    /// the head's tie rule.
    fn top2(v: &[f32]) -> (usize, usize) {
        let mut best = 0;
        for (i, &x) in v.iter().enumerate() {
            if x > v[best] {
                best = i;
            }
        }
        let mut second = usize::from(best == 0);
        for (i, &x) in v.iter().enumerate() {
            if i != best && x > v[second] {
                second = i;
            }
        }
        (best, second)
    }

    /// One graph of the set: its block, label and rows, ik's side.
    struct IkGraph {
        block: i32,
        graph: Graph,
        tokens: Vec<u32>,
        pos: Vec<u32>,
        states: Vec<f32>,
        /// ik's head rows: `inp_out_ids`, or the one row of a one-row graph.
        out_ids: Vec<usize>,
        logits: Vec<f32>,
    }

    impl IkGraph {
        fn read(
            set: &MtpSet,
            block: i32,
            graph: Graph,
            hidden: usize,
        ) -> Result<IkGraph, GateError> {
            let find = |name: &str| set.find(block, graph, name, 0);
            let unsigned = |name: &str| -> Result<Vec<u32>, GateError> {
                set.i32s(find(name)?, Layout::Flat)?
                    .into_iter()
                    .map(|v| u32::try_from(v).map_err(|_| format!("{name} holds {v}").into()))
                    .collect()
            };
            let tokens = unsigned("inp_tokens")?;
            let n = tokens.len();
            // GLM's MTP layer has no rope, so ik's graph has no `inp_pos`: a
            // row's position is its mask row's count of visible keys, less one.
            // The mask is `ne[0]` keys by its rows padded past `n`.
            let mask_row = find("KQ_mask")?;
            let kv = usize::try_from(mask_row.row.ne[0])?;
            let mask = set.logical_f32s(mask_row)?;
            if n == 0 || kv == 0 || mask.len() < n * kv {
                return Err(
                    format!("KQ_mask holds {} values, {n} rows of {kv} keys", mask.len()).into(),
                );
            }
            let pos: Vec<u32> = (0..n)
                .map(|r| {
                    let seen = mask[r * kv..(r + 1) * kv]
                        .iter()
                        .filter(|x| x.is_finite())
                        .count();
                    u32::try_from(seen)
                        .ok()
                        .and_then(|c| c.checked_sub(1))
                        .ok_or_else(|| GateError::from(format!("KQ_mask row {r} shows no key")))
                })
                .collect::<Result<_, _>>()?;
            // ik builds `inp_out_ids` only for a graph of more than one row.
            let out_ids = if n > 1 {
                unsigned("inp_out_ids")?
                    .into_iter()
                    .map(|v| v as usize)
                    .collect()
            } else {
                vec![0]
            };
            let g = IkGraph {
                block,
                graph,
                states: set.logical_f32s(find("inp_mtp_states")?)?,
                logits: set.logical_f32s(find("result_output")?)?,
                tokens,
                pos,
                out_ids,
            };
            let label = g.label();
            if n == 0 || g.pos.len() != n || g.states.len() != n * hidden {
                return Err(format!(
                    "{label}: {n} tokens, {} positions, {} hidden values; {n} rows of {hidden} take \
                     {}",
                    g.pos.len(),
                    g.states.len(),
                    n * hidden
                )
                .into());
            }
            if g.pos.windows(2).any(|w| w[1] != w[0] + 1) || g.out_ids.iter().any(|&o| o >= n) {
                return Err(format!(
                    "{label}: positions not consecutive, or an output row past {n}"
                )
                .into());
            }
            Ok(g)
        }

        fn label(&self) -> String {
            format!("block {} {}", self.block, self.graph.as_str())
        }
    }

    /// The set's graphs in the order ik computed them: the warmup, then each
    /// verified block's graphs in file order.
    fn graphs_of(set: &MtpSet) -> Vec<(i32, Graph)> {
        let mut order: Vec<(i32, Graph)> = Vec::new();
        for r in &set.rows {
            if !order.contains(&(r.block, r.graph)) {
                order.push((r.block, r.graph));
            }
        }
        order
    }

    /// How a graph's last row's argmax stood against ik's.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Head {
        Same,
        /// ik's runner-up, ik's margin under the cap.
        Tie,
        Bad,
        /// ik computed no head on the last row: not held.
        Unheld,
    }

    /// One graph's replay: our id, ik's, how they stood, and our logits'
    /// distance from ik's where ik computed the head.
    struct Replayed {
        ours: u32,
        ik: Option<u32>,
        head: Head,
        rel: Option<f64>,
    }

    /// Replay `g` from the host: the runs before the last store-only walks,
    /// the last a chain; our logits against ik's on the last row.
    fn replay(m: &mut Glm5nextModel, g: &IkGraph, hidden: usize) -> Result<Replayed, GateError> {
        let n = g.tokens.len();
        let mut c0 = 0;
        let mut ours = None;
        while c0 < n {
            let rows = WALK_ROWS.min(n - c0);
            let f = NextnFeed {
                tokens: &g.tokens[c0..c0 + rows],
                pos0: g.pos[c0],
                hidden: NextnHidden::Host(&g.states[c0 * hidden..(c0 + rows) * hidden]),
            };
            if c0 + rows < n {
                nextn_walk(m, f, NextnHead::Full, NextnMode::Store)?;
            } else {
                let mut out = [0u32; 1];
                let k = nextn_chain(m, f, 0, NextnHead::Full, NextnMode::Eager, &mut out)?;
                if k != 1 {
                    return Err(format!("{}: a chain of {k} ids, want 1", g.label()).into());
                }
                ours = Some(out[0]);
            }
            c0 += rows;
        }
        let ours = ours.ok_or_else(|| format!("{}: no chain ran", g.label()))?;
        let logits = nextn_logits(m)?;
        let vocab = logits.len();
        let (top, _) = top2(&logits);
        if top != ours as usize {
            println!(
                "(o) {}: the chain names {ours}, our logits' argmax {top} FAIL",
                g.label()
            );
            return Ok(Replayed {
                ours,
                ik: None,
                head: Head::Bad,
                rel: None,
            });
        }
        if g.logits.len() != g.out_ids.len() * vocab {
            return Err(format!(
                "{}: result_output holds {} values, {} output rows of {vocab}",
                g.label(),
                g.logits.len(),
                g.out_ids.len()
            )
            .into());
        }
        let Some(k) = g.out_ids.iter().position(|&o| o == n - 1) else {
            println!(
                "(o) {}: ik computed no head on row {}, not held",
                g.label(),
                n - 1
            );
            return Ok(Replayed {
                ours,
                ik: None,
                head: Head::Unheld,
                rel: None,
            });
        };
        let ik = &g.logits[k * vocab..(k + 1) * vocab];
        let (ik_top, ik_2) = top2(ik);
        let margin = f64::from(ik[ik_top]) - f64::from(ik[ik_2]);
        let cap = margin_cap(logits_band(), ik);
        let head = if top == ik_top {
            Head::Same
        } else if margin <= cap && top == ik_2 {
            Head::Tie
        } else {
            Head::Bad
        };
        let band = logits_band();
        let r = rel(&logits, ik);
        let in_band = r <= band;
        println!(
            "(o) {}: {n} rows from {}, argmax {top} ik {ik_top} (runner-up {ik_2}, margin \
             {margin:.3e}, cap {cap:.3e}), logits rel {r:.3e} (held <= {band:.3e}) {}",
            g.label(),
            g.pos[0],
            match (head, in_band) {
                (Head::Same, true) => "PASS",
                (Head::Tie, true) => "PASS (a tie: ik's runner-up)",
                (Head::Same | Head::Tie, false) => "FAIL (past the band)",
                _ => "FAIL",
            }
        );
        Ok(Replayed {
            ours,
            ik: Some(ik_top as u32),
            head,
            rel: Some(r),
        })
    }

    /// (m): the warmup graph's embedding rows, unmasked at every position.
    fn pos_mask(set: &MtpSet, hidden: usize) -> Result<bool, GateError> {
        let masked = set
            .find(-1, Graph::Warmup, "mtp_token_embd_pos_masked", 0)
            .is_ok();
        let e = set.logical_f32s(set.find(-1, Graph::Warmup, "inp_embd", 0)?)?;
        if e.is_empty() || e.len() % hidden != 0 {
            return Err(format!("inp_embd holds {} values, rows of {hidden}", e.len()).into());
        }
        let rows = e.len() / hidden;
        // The warmup is the prompt from position 0, so its row 0 is position 0.
        let ok =
            !masked && (0..rows).all(|r| e[r * hidden..(r + 1) * hidden].iter().any(|&x| x != 0.0));
        println!(
            "(m) the warmup's {rows} embedding rows (the prompt from position 0): no mask node, none zeros {}",
            verdict(ok)
        );
        Ok(ok)
    }

    /// (o): every graph replayed, each block's proposal against the set's.
    fn oracle(m: &mut Glm5nextModel, set: &MtpSet, hidden: usize) -> Result<bool, GateError> {
        m.reset()?;
        let mut ok = true;
        let mut last: Option<Replayed> = None;
        let (mut same, mut ties, mut bad, mut unheld) = (0usize, 0usize, 0usize, 0usize);
        let (mut blocks_ok, mut blocks_tie, mut blocks_bad) = (0usize, 0usize, 0usize);
        let band = logits_band();
        let (mut worst, mut past) = (0.0f64, 0usize);
        for (b, g) in graphs_of(set) {
            if g == Graph::Update {
                let draft = set.drafts_of(b);
                let (&[want], Some(prev)) = (draft.as_slice(), last.as_ref()) else {
                    println!(
                        "(o) block {b}: {} draft rows and {} graph before its update FAIL",
                        draft.len(),
                        if last.is_some() { "a" } else { "no" }
                    );
                    blocks_bad += 1;
                    continue;
                };
                if prev.ik.is_some_and(|ik| ik != want) {
                    println!(
                        "(o) block {b}: the set's draft {want} is not the argmax {:?} of the graph \
                         before its update FAIL",
                        prev.ik
                    );
                    blocks_bad += 1;
                } else if prev.ours == want {
                    blocks_ok += 1;
                } else if prev.head == Head::Tie {
                    blocks_tie += 1;
                    println!(
                        "(o) block {b}: our proposal {} against the set's {want}, a tie",
                        prev.ours
                    );
                } else {
                    blocks_bad += 1;
                    println!(
                        "(o) block {b}: our proposal {} against the set's {want} FAIL",
                        prev.ours
                    );
                }
            }
            let ig = IkGraph::read(set, b, g, hidden)?;
            let r = replay(m, &ig, hidden)?;
            match r.head {
                Head::Same => same += 1,
                Head::Tie => ties += 1,
                Head::Bad => bad += 1,
                Head::Unheld => unheld += 1,
            }
            if let Some(d) = r.rel {
                worst = worst.max(d);
                // `rel` reads a NaN as infinite: past the band.
                if d > band {
                    past += 1;
                }
            }
            last = Some(r);
        }
        let blocks = set.blocks().len();
        let graphs_ok = bad == 0 && same > 0 && past == 0;
        let props_ok = blocks_bad == 0 && blocks_ok + blocks_tie == blocks && blocks_ok > 0;
        ok &= graphs_ok && props_ok;
        println!(
            "(o) graphs: {same} ik's argmax, {ties} ik's runner-up at a tie, {bad} other, {unheld} \
             without ik's head; logits rel worst {worst:.3e}, {:.3} of the band {band:.3e}, {past} \
             past it {}",
            worst / band,
            verdict(graphs_ok)
        );
        println!(
            "(o) blocks: {blocks_ok} of {blocks} proposals the set's draft, {blocks_tie} a tie, \
             {blocks_bad} other {}",
            verdict(props_ok)
        );
        Ok(ok)
    }

    /// Every hidden row a prompt call's units leave, in position order, read
    /// by [`nextn_hidden`] from the unit's arena in runs of [`WALK_ROWS`],
    /// and the target streams each run was made from
    /// ([`nextn_target_streams`]).
    fn unit_rows(
        m: &mut Glm5nextModel,
        prompt: &[u32],
        path: PrefillMode,
    ) -> Result<(Vec<f32>, Vec<f32>), GateError> {
        m.reset()?;
        set_prefill(m, path)?;
        let mut got: Vec<f32> = Vec::new();
        let mut streams: Vec<f32> = Vec::new();
        let mut units = |m: &mut Glm5nextModel, arena: GlmArena, first: u32, rows: usize| {
            let mut c0 = 0;
            while c0 < rows {
                let r = WALK_ROWS.min(rows - c0);
                // The walk from the position after the unit's row c0 reads it.
                let pos0 = first + c0 as u32 + 1;
                let target = NextnHidden::Target {
                    walk: arena,
                    first: c0,
                };
                got.extend(nextn_hidden(m, pos0, target, r)?);
                streams.extend(nextn_target_streams(m, pos0, target, r)?);
                c0 += r;
            }
            Ok(())
        };
        let sink: &mut GlmPromptSink<'_> = &mut units;
        prompt_with(m, prompt, path, Some(sink))?;
        Ok((got, streams))
    }

    /// The unit roundoff of f32.
    const U: f64 = 1.0 / (1u64 << 24) as f64;

    /// PIN(2026-10-01): the bound on one hidden value the walk's gather
    /// writes — `ds41_hc_mean` over the four streams, then `rms_norm` over
    /// `n` = 4,096 values with `output_norm`'s gain — against its f64
    /// replica on the same f32 streams, first order in `u` = 2^-24, derived
    /// from the kernels' own order:
    /// - the mean `(((a + b) + c) + d) · 0.25`: three additions, the product
    ///   exact, so `|δm_i| <= 3u · S_i`, `S_i` the mean of the four `|s|`
    ///   (absolute: the streams may cancel);
    /// - the sum of squares of the device's own means: each of 256 threads
    ///   sums its 16 squares in order, then a five-level warp butterfly and
    ///   the three-level warp tree — every term through at most 23 additions
    ///   and its square's rounding, all terms positive: relative `24u`; the
    ///   means' error moves it by at most `6u · ρ` relative, `ρ = Σ|m_i|S_i /
    ///   Σm_i²`;
    /// - `/ n` exact (a power of two), `+ eps` `u`, the square root halves its
    ///   argument's error and adds its own, and the reciprocal its own, each
    ///   2u (the approximate forms' bound): `ε_scale = (25u + 6u·ρ) / 2 +
    ///   4u`;
    /// - `(scale · g_i) · m_i`: two roundings, `2u`.
    ///
    /// So `|y_i − ŷ_i| <= b_i`, with `b_i = |g_i| · scale · (3u · S_i +
    /// |m_i| · (ε_scale + 2u))`. The clause holds every value within `2 ·
    /// b_i` (the second order and the replica's own f64 error are below
    /// 10^-6 of `b_i`) plus `2^-126 · (1 + |g_i| · scale)`, a flushed
    /// subnormal. A value is bounded, not a row's norm: the bound is
    /// componentwise.
    struct HiddenRef {
        /// The replica's values and each value's bound, `2 · b_i` plus the
        /// subnormal floor.
        y: Vec<f64>,
        bound: Vec<f64>,
        /// The row's mean square before the norm.
        ms: f64,
    }

    /// Row `streams` (`4 · n`, stream-major) through the walk's gather in
    /// f64, with the bound on each value ([`HiddenRef`]).
    fn hidden_ref(streams: &[f32], gain: &[f32], eps: f32) -> HiddenRef {
        let n = gain.len();
        let at = |j: usize, i: usize| f64::from(streams[j * n + i]);
        let m: Vec<f64> = (0..n)
            .map(|i| (at(0, i) + at(1, i) + at(2, i) + at(3, i)) * 0.25)
            .collect();
        let s: Vec<f64> = (0..n)
            .map(|i| (0..4).map(|j| at(j, i).abs()).sum::<f64>() * 0.25)
            .collect();
        let sq: f64 = m.iter().map(|v| v * v).sum();
        let ms = sq / n as f64;
        let scale = 1.0 / (ms + f64::from(eps)).sqrt();
        let rho = if sq > 0.0 {
            m.iter().zip(&s).map(|(a, b)| a.abs() * b).sum::<f64>() / sq
        } else {
            0.0
        };
        let e_scale = (25.0 * U + 6.0 * U * rho) / 2.0 + 4.0 * U;
        let floor = f64::from(f32::MIN_POSITIVE);
        let mut y = Vec::with_capacity(n);
        let mut bound = Vec::with_capacity(n);
        for i in 0..n {
            let g = f64::from(gain[i]).abs() * scale;
            y.push(f64::from(gain[i]) * scale * m[i]);
            bound.push(
                2.0 * g * (3.0 * U * s[i] + m[i].abs() * (e_scale + 2.0 * U)) + floor * (1.0 + g),
            );
        }
        HiddenRef { y, bound, ms }
    }

    /// (p) the value held: every hidden row a prompt call's units leave, as
    /// the walk's gather wrote it, within its derived bound of the f64
    /// replica on the target streams it read ([`HiddenRef`]); the worst
    /// value's distance over its bound, the rows' rel-L2, their mean squares
    /// and the bit-equal count printed.
    fn hidden_held(rows: &[f32], streams: &[f32], gain: &[f32], eps: f32, path: &str) -> bool {
        let n = gain.len();
        let count = rows.len() / n.max(1);
        let shape_ok = n > 0 && rows.len() == count * n && streams.len() == count * 4 * n;
        let (mut worst, mut num, mut den) = (0.0f64, 0.0f64, 0.0f64);
        let (mut out, mut bits, mut ms_lo, mut ms_hi) = (0usize, 0usize, f64::MAX, 0.0f64);
        if shape_ok {
            for r in 0..count {
                let h = hidden_ref(&streams[r * 4 * n..(r + 1) * 4 * n], gain, eps);
                ms_lo = ms_lo.min(h.ms);
                ms_hi = ms_hi.max(h.ms);
                for i in 0..n {
                    let got = f64::from(rows[r * n + i]);
                    let d = (got - h.y[i]).abs();
                    worst = worst.max(d / h.bound[i]);
                    if d.is_nan() || d > h.bound[i] {
                        out += 1;
                    }
                    if got.to_bits() == (h.y[i] as f32 as f64).to_bits() {
                        bits += 1;
                    }
                    num += d * d;
                    den += h.y[i] * h.y[i];
                }
            }
        }
        let ok = shape_ok && count > 0 && out == 0;
        println!(
            "(p) the gather's value ({path}): {count} rows of {n} against the f64 replica of \
             mean-then-norm on their target streams: {out} values past their bound, the worst at \
             {worst:.3e} of it (held <= 1), rel-L2 {:.3e}, mean square {ms_lo:.3e}..{ms_hi:.3e}, \
             {bits} values the replica's f32 bits (printed) {}",
            (num / den.max(f64::MIN_POSITIVE)).sqrt(),
            verdict(ok)
        );
        ok
    }

    /// (p) the pairing: the hidden rows a walk fed the target's arenas reads.
    /// A prompt call's first chunk of units in a batch (the prompt-batch
    /// arena) and by steps (the step arena) leave the same rows bit for bit,
    /// the whole prompt's two printed, each value of either within
    /// its bound of the gather's f64 replica on the rows' target streams
    /// ([`hidden_held`], at the file's `eps`); the row at position
    /// `q` lies closer to ik's warmup row at `q + 1` — ik's MTP row `p` reads
    /// the target's hidden of `p − 1` — than to its rows `q` and `q + 2`, the
    /// distances printed. Then a verify of two rows after the prompt leaves in the
    /// pair arena the rows two plain steps leave in the step arena, bit for
    /// bit, and after its commit and one more step still does, the step arena
    /// holding the plain run's next row: the verify's row 0 kept past the
    /// step that writes its buffers.
    fn pairing(
        m: &mut Glm5nextModel,
        set: &MtpSet,
        prompt: &[u32],
        hidden: usize,
        eps: f32,
    ) -> Result<bool, GateError> {
        let (batch, batch_streams) = unit_rows(m, prompt, PrefillMode::Batch)?;
        let (steps, steps_streams) = unit_rows(m, prompt, PrefillMode::Steps)?;
        let gain = {
            let (gpu, w, _) = m.body_parts("pairing")?;
            let name = names::output_norm();
            let Some(DevWeight::F32 { w: g, .. }) = w.get(&name) else {
                return Err(format!("{name} is not resident as F32").into());
            };
            g.buf().to_host_vec(gpu.stream())?
        };
        let held_ok = hidden_held(&batch, &batch_streams, &gain, eps, "batches")
            & hidden_held(&steps, &steps_streams, &gain, eps, "steps");
        let n = prompt.len();
        let bits = |a: &[f32], b: &[f32]| {
            a.len() == b.len()
                && a.iter()
                    .map(|x| x.to_bits())
                    .eq(b.iter().map(|x| x.to_bits()))
        };
        // A batch of at most a chunk runs the step's gemvs, so its rows are
        // the steps' bits; a longer one runs the GEMM, whose rows the gather's
        // bound holds on their own target streams above.
        let short = &prompt[..n.min(CHUNK)];
        let (batch_short, _) = unit_rows(m, short, PrefillMode::Batch)?;
        let (steps_short, _) = unit_rows(m, short, PrefillMode::Steps)?;
        let same = batch_short.len() == short.len() * hidden && bits(&batch_short, &steps_short);
        let mut ok = same && held_ok && batch.len() == n * hidden;
        println!(
            "(p) the prompt's first {} units' hidden rows, in a batch = by steps, bit for bit {}; \
             all {n} units' in batches against by steps {:.3e} (printed: from {GEMM_FROM} \
             positions a batch runs the GEMM)",
            short.len(),
            verdict(same),
            rel(&batch, &steps)
        );
        // Two batches in one group: the sink reads each unit's own final
        // streams after the group; the same batches at groups of one.
        let two: Vec<u32> = prompt.iter().copied().cycle().take(GROUPED).collect();
        set_prefill_group(m, 2)?;
        let grouped = unit_rows(m, &two, PrefillMode::Batch);
        set_prefill_group(m, 1)?;
        let (grouped, _) = grouped?;
        let (ones, _) = unit_rows(m, &two, PrefillMode::Batch)?;
        let same2 = grouped.len() == GROUPED * hidden && bits(&grouped, &ones);
        ok &= same2;
        println!(
            "(p) {GROUPED} ids (two batches) at groups of two: every unit's hidden rows = at \
             groups of one, bit for bit {}",
            verdict(same2)
        );
        let warm = IkGraph::read(set, -1, Graph::Warmup, hidden)?;
        let pos_ok = warm.tokens == prompt && warm.pos.iter().copied().eq(0..n as u32);
        if !pos_ok {
            println!("(p) the warmup's rows are not the prompt's at positions 0.. FAIL");
            return Ok(false);
        }
        let ik_row = |p: usize| &warm.states[p * hidden..(p + 1) * hidden];
        let (mut worst, mut bad_shift) = (0.0f64, 0usize);
        for q in 0..n - 1 {
            let ours = &batch[q * hidden..(q + 1) * hidden];
            let at = rel(ours, ik_row(q + 1));
            worst = worst.max(at);
            let before = rel(ours, ik_row(q));
            let after = (q + 2 < n).then(|| rel(ours, ik_row(q + 2)));
            let shifted = before <= at || after.is_some_and(|a| a <= at);
            if shifted {
                bad_shift += 1;
            }
            println!(
                "(p) row {q}: against ik's row {} {at:.3e}, row {q} {before:.3e}, row {} {}{}",
                q + 1,
                q + 2,
                after.map_or_else(|| "-".to_string(), |a| format!("{a:.3e}")),
                if shifted { " closer off its pair" } else { "" }
            );
        }
        let ik_ok = bad_shift == 0;
        ok &= ik_ok;
        println!(
            "(p) our row at q against ik's warmup row q + 1, {} rows: worst {worst:.3e} (printed), \
             {bad_shift} closer at q or q + 2 {}",
            n - 1,
            verdict(ik_ok)
        );

        m.reset()?;
        set_prefill(m, PrefillMode::Batch)?;
        let step_row = |m: &mut Glm5nextModel| -> Result<Vec<f32>, GpuError> {
            let pos0 = m.pos();
            nextn_hidden(
                m,
                pos0,
                NextnHidden::Target {
                    walk: GlmArena::Step,
                    first: 0,
                },
                1,
            )
        };
        let t0 = feed(m, prompt)?;
        let p = m.pos();
        let t1 = m.step(&[t0])?;
        let s0 = step_row(m)?;
        let t2 = m.step(&[t1])?;
        let s1 = step_row(m)?;
        m.step(&[t2])?;
        let s2 = step_row(m)?;
        let plain: Vec<u32> = s0.iter().chain(&s1).map(|x| x.to_bits()).collect();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        m.reset()?;
        feed(m, prompt)?;
        let rows = m.step_rows::<2>([t0, t1])?;
        let pair_row = |m: &mut Glm5nextModel| {
            nextn_hidden(
                m,
                p + 1,
                NextnHidden::Target {
                    walk: GlmArena::Pair,
                    first: 0,
                },
                2,
            )
        };
        let v = pair_row(m)?;
        let verify_ok = rows == [t1, t2] && bits(&v) == plain;
        m.rollback(p + 2)?;
        m.keep_rows(2, PassKind::Pair)?;
        m.step(&[t2])?;
        let kept_ok = bits(&pair_row(m)?) == plain && bits(&step_row(m)?) == bits(&s2);
        ok &= verify_ok && kept_ok;
        println!(
            "(p) a verify of two rows at {p}: tokens {rows:?} (want {:?}), the pair arena's rows = \
             two plain steps' {}; after its commit and a step, the pair arena's rows unchanged and \
             the step arena's the plain run's next {}",
            [t1, t2],
            verdict(verify_ok),
            verdict(kept_ok)
        );
        // A verify whose row 1 its commit takes back: the pair arena then
        // holds the kept row's position alone.
        let q = m.pos();
        m.step_rows::<2>([t2, t2])?;
        m.rollback(q + 1)?;
        m.keep_rows(1, PassKind::Pair)?;
        let pair = |first: usize| NextnHidden::Target {
            walk: GlmArena::Pair,
            first,
        };
        let one = nextn_hidden(m, q + 1, pair(0), 1);
        let both = nextn_hidden(m, q + 1, pair(0), 2);
        let held = format!("the arena holds positions {q}..{}", q + 1);
        let cut_ok = one.is_ok() && matches!(&both, Err(e) if e.to_string().contains(&held));
        ok &= cut_ok;
        println!(
            "(p) a verify at {q} with row 0 kept alone: the kept row {}, both rows {} (want \
             refused: {held}) {}",
            match &one {
                Ok(v) => format!("{} values", v.len()),
                Err(e) => format!("error \"{e}\""),
            },
            match &both {
                Ok(_) => "accepted".to_string(),
                Err(e) => format!("error \"{e}\""),
            },
            verdict(cut_ok)
        );
        m.reset()?;
        Ok(ok)
    }

    /// A plain run's ids and each step's logits digest.
    struct Plain {
        ids: Vec<u32>,
        steps: Vec<u64>,
    }

    /// The set's prompt fed by `path` from a reset, then [`N`] − 1 steps.
    fn plain(m: &mut Glm5nextModel, prompt: &[u32], path: PrefillMode) -> Result<Plain, GateError> {
        m.reset()?;
        set_prefill(m, path)?;
        let mut ids = vec![feed(m, prompt)?];
        let mut steps = Vec::with_capacity(N);
        for _ in 1..N {
            let t = m.step(&[ids[ids.len() - 1]])?;
            ids.push(t);
            steps.push(Fnv1a64::default().f32s(&m.logits()?).value());
        }
        Ok(Plain { ids, steps })
    }

    /// ik's committed stream after the prompt: each round's target tokens,
    /// then its plain steps' tokens.
    fn ik_stream(set: &MtpSet) -> Vec<u32> {
        let mut s: Vec<u32> = set
            .verify
            .iter()
            .flat_map(|v| v.target.iter().copied())
            .collect();
        let mut plain = set.plain.clone();
        plain.sort_by_key(|p| p.pos);
        s.extend(plain.iter().map(|p| p.token));
        s
    }

    /// The generation's passes, kept.
    #[derive(Default)]
    struct Kept {
        rows: Vec<usize>,
        proposed: Vec<bool>,
    }

    impl PassSink<Session<Body>> for Kept {
        type Error = GateError;

        fn begin(&mut self, _: &Session<Body>) -> Result<(), GateError> {
            Ok(())
        }

        fn pass(
            &mut self,
            _: &Session<Body>,
            c: &Committed,
            _: &[u32],
            _: std::time::Duration,
        ) -> Result<(), GateError> {
            self.rows.push(c.kept);
            self.proposed.push(c.proposed);
            Ok(())
        }
    }

    /// The rows-log that hears nothing.
    struct Quiet;

    impl app::RowsLog for Quiet {
        fn capture_rows(&mut self, _: usize, _: usize) -> Result<(), app::SessionError> {
            Ok(())
        }
    }

    /// (w) one drafted run: the set's prompt fed by `path` from a reset, the
    /// windows for [`N`] ids, against `plain`.
    fn drafted(
        m: Glm5nextModel,
        prompt: &[u32],
        path: PrefillMode,
        plain: &Plain,
        set: &MtpSet,
    ) -> Result<(Glm5nextModel, bool), GateError> {
        let mut m = m;
        m.reset()?;
        set_prefill(&mut m, path)?;
        let ctx = u32::try_from(CTX)?;
        let mut s = Session::from_model(m, ctx);
        let draft = MtpDraft::open(s.model(), path, StepMode::Eager)?;
        let mut spec = s.with_draft::<MtpDraft<Body>, M>(draft, &mut Quiet)?;
        let first = spec.prompt(&mut s, prompt)?;
        let mut k = Kept::default();
        let out = runtime::generate(
            &mut s,
            &mut spec,
            prompt,
            first,
            &runtime::Stop::new(N, ctx)?,
            &mut k,
        )?;
        let m = s.into_model();
        let ids_ok = out.tokens.len() >= N && out.tokens[..N] == plain.ids[..N];
        let windows = k.proposed.iter().filter(|&&p| p).count();
        let accepted = k
            .rows
            .iter()
            .zip(&k.proposed)
            .filter(|&(&r, &p)| p && r == M)
            .count();
        let rejected = windows - accepted;
        let rows_ok = rejected > 0 && accepted > 0;
        println!(
            "(w) {path:?}: the drafted run's {N} ids = the plain run's {}",
            verdict(ids_ok)
        );
        println!(
            "(w) {path:?}: {windows} windows, {accepted} accepted, {rejected} rejected a row (ik: {} \
             blocks, {} accepted) {}",
            set.verify.len(),
            set.verify.iter().map(|v| v.accepted).sum::<usize>(),
            verdict(rows_ok)
        );
        Ok((m, ids_ok && rows_ok))
    }

    /// Whether `r` is refused by name, `want` in its message.
    fn refused<T>(what: &str, r: Result<T, GpuError>, want: &str) -> bool {
        let ok = matches!(&r, Err(e) if e.to_string().contains(want));
        let got = match &r {
            Ok(_) => "accepted".to_string(),
            Err(e) => e.to_string(),
        };
        println!("(f) {what} -> {got} {}", verdict(ok));
        ok
    }

    /// (s) `without`, a state of the load without the layer, put back on the
    /// empty NextN load `m`: refused by name, and `m` still at position 0,
    /// unpoisoned. Mutant: the body's NextN-side check taken out (the shared
    /// layout check refuses it under its own name).
    fn layout_refused(m: &mut Glm5nextModel, without: &GlmSeq) -> bool {
        let r = seq_resume(m, without);
        let want = "a state without the NextN rows put back on a load with the layer";
        let named = matches!(&r, Err(e) if e.to_string().contains(want));
        let empty = m.pos() == 0 && m.poisoned().is_none();
        println!(
            "(s) a state of the load without the layer put back on the NextN load -> {} {}; the \
             model after it at position {}, poisoned {:?} {}",
            match &r {
                Ok(()) => "accepted".to_string(),
                Err(e) => e.to_string(),
            },
            verdict(named),
            m.pos(),
            m.poisoned(),
            verdict(empty)
        );
        named && empty
    }

    /// A walk's feed.
    fn f<'a>(tokens: &'a [u32], pos0: u32, hidden: NextnHidden<'a>) -> NextnFeed<'a> {
        NextnFeed {
            tokens,
            pos0,
            hidden,
        }
    }

    /// (f) on the NextN load, from a reset: every refusal leaves the store's
    /// positions at 0.
    fn refusals(m: &mut Glm5nextModel, hidden: usize) -> Result<bool, GateError> {
        m.reset()?;
        let zeros = vec![0.0f32; (WALK_ROWS + 1) * hidden];
        let host = |rows: usize| NextnHidden::Host(&zeros[..rows * hidden]);
        let toks = vec![1u32; WALK_ROWS + 1];
        let mut ok = true;
        ok &= refused(
            "a captured walk",
            nextn_walk(
                m,
                f(&toks[..1], 0, host(1)),
                NextnHead::Full,
                NextnMode::Graph,
            ),
            "a captured NextN walk",
        );
        ok &= refused(
            "no rows",
            nextn_walk(
                m,
                f(&toks[..0], 0, host(0)),
                NextnHead::Full,
                NextnMode::Eager,
            ),
            "a walk of 0 rows",
        );
        ok &= refused(
            "one row past the walk's width",
            nextn_walk(
                m,
                f(&toks, 0, host(WALK_ROWS + 1)),
                NextnHead::Full,
                NextnMode::Eager,
            ),
            &format!("a walk of {} rows", WALK_ROWS + 1),
        );
        ok &= refused(
            "a walk from past the store's positions",
            nextn_walk(
                m,
                f(&toks[..1], 1, host(1)),
                NextnHead::Full,
                NextnMode::Eager,
            ),
            "a walk from position 1",
        );
        ok &= refused(
            "a token past the vocabulary",
            nextn_walk(
                m,
                f(&[u32::MAX], 0, host(1)),
                NextnHead::Full,
                NextnMode::Eager,
            ),
            "past the",
        );
        ok &= refused(
            "a host hidden slice of the wrong length",
            nextn_walk(
                m,
                f(&toks[..1], 0, NextnHidden::Host(&zeros[..hidden - 1])),
                NextnHead::Full,
                NextnMode::Eager,
            ),
            "hidden values for 1 rows",
        );
        ok &= refused(
            "target rows past the step arena's one",
            nextn_walk(
                m,
                f(
                    &toks[..1],
                    0,
                    NextnHidden::Target {
                        walk: GlmArena::Step,
                        first: 1,
                    },
                ),
                NextnHead::Full,
                NextnMode::Eager,
            ),
            "of the Step arena's 1",
        );
        let mut out = [0u32; 1];
        ok &= refused(
            "a chain with an own walk",
            nextn_chain(
                m,
                f(&toks[..1], 0, host(1)),
                1,
                NextnHead::Full,
                NextnMode::Eager,
                &mut out,
            ),
            "a chain of 1 own walks",
        );
        ok &= refused(
            "a chain in the store walk's mode",
            nextn_chain(
                m,
                f(&toks[..1], 0, host(1)),
                0,
                NextnHead::Full,
                NextnMode::Store,
                &mut out,
            ),
            "a chain in the store walk's mode",
        );
        ok &= refused(
            "a chain into no place",
            nextn_chain(
                m,
                f(&toks[..1], 0, host(1)),
                0,
                NextnHead::Full,
                NextnMode::Eager,
                &mut [],
            ),
            "into 0 places",
        );
        let held = m
            .body("refusals")?
            .nextn()
            .map_or(usize::MAX, bloomery_gpu_glm5next::Nextn::held);
        let held_ok = held == 0;
        ok &= held_ok;
        println!(
            "(f) the store's positions after the refusals: {held} {}",
            verdict(held_ok)
        );
        ok &= by_position(m, &toks)?;
        Ok(ok)
    }

    /// (f) the target's rows by position, through the walk's own gather:
    /// the pair arena from a reset (no verify has written it), the step arena
    /// for a walk at position 0, then, after [`FED`] ids in one batch, the
    /// prompt-batch arena's row [`FED`] (past the batch's rows, inside its
    /// buffers) for the walk that follows, and its rows 0 and 1 for a walk
    /// from 5, which reads positions 4 and 5: each refused by name. The model
    /// left reset.
    fn by_position(m: &mut Glm5nextModel, toks: &[u32]) -> Result<bool, GateError> {
        let target = |walk: GlmArena, first: usize| NextnHidden::Target { walk, first };
        m.reset()?;
        let mut ok = refused(
            "the pair arena before any verify",
            nextn_hidden(m, 1, target(GlmArena::Pair, 0), 1),
            "of the Pair arena as positions 0..1 for a walk from 1: the arena holds no position",
        );
        ok &= refused(
            "the step arena for a walk at position 0",
            nextn_hidden(m, 0, target(GlmArena::Step, 0), 1),
            "for a walk at position 0",
        );
        set_prefill(m, PrefillMode::Batch)?;
        let ids: Vec<u32> = toks.iter().copied().cycle().take(FED).collect();
        feed(m, &ids)?;
        let fed = FED as u32;
        ok &= refused(
            "a prompt-batch row past the batch's last",
            nextn_hidden(m, fed + 1, target(GlmArena::Prefill, FED), 1),
            &format!(
                "for a walk from {}: the arena holds positions 0..{FED}",
                FED + 1
            ),
        );
        ok &= refused(
            "prompt-batch rows read as other positions",
            nextn_hidden(m, 5, target(GlmArena::Prefill, 0), 2),
            "as positions 4..6",
        );
        let good = nextn_hidden(m, fed, target(GlmArena::Prefill, FED - 1), 1);
        let good_ok = good.is_ok();
        ok &= good_ok;
        println!(
            "(f) the batch's last row for the walk after it: {} {}",
            match &good {
                Ok(v) => format!("{} values", v.len()),
                Err(e) => format!("error \"{e}\""),
            },
            verdict(good_ok)
        );
        m.reset()?;
        Ok(ok)
    }

    /// Write `bytes` over the first value of the NextN layer's
    /// `shared_head_norm` gain and return the bytes it replaced.
    fn patch_head_norm(
        m: &mut Glm5nextModel,
        index: usize,
        bytes: [u8; 4],
    ) -> Result<[u8; 4], GateError> {
        let name = names::nextn_shared_head_norm(index);
        let nx = m
            .body("patch_head_norm")?
            .nextn()
            .ok_or("the NextN load holds no NextN layer")?;
        let Some(DevWeight::F32 { w: gain, .. }) = nx.weights().get(&name) else {
            return Err(format!("{name} is not resident as F32").into());
        };
        patch_bytes(m.gpu().stream(), gain.buf(), 0, bytes)
    }

    /// (h): a NaN gain in the head's norm makes the chain's logits NaN; the
    /// chain is a fault by name and the model poisoned by it at once, a walk
    /// after it refused as poisoned; the gain put back and a reset, the
    /// chain's id is the clean one.
    fn chain_fault(m: &mut Glm5nextModel, index: usize, hidden: usize) -> Result<bool, GateError> {
        let zeros = vec![0.0f32; hidden];
        let tok = [1u32];
        let feed = || f(&tok, 0, NextnHidden::Host(&zeros));
        let mut out = [0u32; 1];
        m.reset()?;
        nextn_chain(m, feed(), 0, NextnHead::Full, NextnMode::Eager, &mut out)?;
        let clean = out[0];
        m.reset()?;
        let old = patch_head_norm(m, index, f32::NAN.to_le_bytes())?;
        let first =
            nextn_chain(m, feed(), 0, NextnHead::Full, NextnMode::Eager, &mut out).map(|_| out[0]);
        let poisoned = m.poisoned();
        let walk = nextn_walk(m, feed(), NextnHead::Full, NextnMode::Store);
        patch_head_norm(m, index, old)?;
        m.reset()?;
        let again =
            nextn_chain(m, feed(), 0, NextnHead::Full, NextnMode::Eager, &mut out).map(|_| out[0]);
        m.reset()?;
        let named = match &first {
            Err(GpuError::Fault { fault, .. }) => poisoned == Some(*fault),
            _ => false,
        };
        let walk_refused = matches!(walk, Err(GpuError::Poisoned { .. }));
        let ok = named && walk_refused && matches!(again, Ok(t) if t == clean);
        let shown = |r: &Result<u32, GpuError>| match r {
            Ok(t) => format!("id {t}"),
            Err(e) => format!("error \"{e}\""),
        };
        println!(
            "(h) NaN in the NextN head norm's gain, a chain: {} (want a fault), the model poisoned \
             by {} (want the chain's fault), a walk after it: {}; the gain put back and a reset: \
             {} (want the clean {clean}) {}",
            shown(&first),
            poisoned.map_or_else(|| "none".to_string(), |f| f.to_string()),
            match &walk {
                Ok(()) => "accepted".to_string(),
                Err(e) => format!("error \"{e}\""),
            },
            shown(&again),
            verdict(ok)
        );
        Ok(ok)
    }

    pub(super) fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[])?;
        let dir = MTP.path(MTP_SET);
        let set = MtpSet::open(&dir, &MTP)?;
        let prompt = set
            .tokens
            .clone()
            .ok_or_else(|| format!("{}: no # tokens line", dir.display()))?;
        let order = graphs_of(&set);
        let accepted: usize = set.verify.iter().map(|v| v.accepted).sum();
        println!(
            "{}: {} graphs over {} blocks ({accepted} accepted), prompt {} ids, family {}",
            dir.display(),
            order.len(),
            set.blocks().len(),
            prompt.len(),
            MTP.name
        );
        println!(
            "bands: the logits' (held, and the argmax cap's) {:.4e}",
            logits_band()
        );
        let open = |p: &str| Split::open(p).map_err(|e| format!("open {p}: {e}"));
        let file = open(MODEL)?;
        let inputs = PlanInputs::read(&file)?;
        let nextn = NextnInputs::read(&inputs)?;
        let machine = workstation::plan_gate(inputs.model.layers);
        let place = PlanLevers::from_levers(&levers)?;
        let ctx = u64::try_from(CTX)?;
        let base = inputs.plan(&machine, ctx, &place)?;
        let np = inputs.plan_nextn(&machine, ctx, &place, &nextn)?;
        let card = |n_l: &[u64]| n_l.iter().sum::<u64>();
        println!(
            "plan: layer {} beside the target, its card bytes {} and arena {}; card experts {} \
             without it, {} with it",
            nextn.index,
            np.nextn_card_bytes(),
            np.arena_bytes,
            card(&base.n_l),
            card(&np.plan.n_l)
        );
        drop(file);

        // (t) the target's plan loaded without the layer: the reference run,
        // at the two KDA lanes the plan counts.
        let t = Instant::now();
        let mut m = Body::open_placed_lanes(
            open(MODEL)?,
            &np.plan,
            &inputs,
            0,
            levers.host(),
            Residency::Off,
            KdaLanes::Two,
        )?;
        m.set_mode(StepMode::Graph);
        println!(
            "load without the layer: {:.1} s, {} resident bytes",
            t.elapsed().as_secs_f64(),
            m.resident_bytes()
        );
        let hidden = m.body("run")?.nextn_hidden_width();
        let mut ok = refused(
            "a walk on a load without the layer",
            nextn_walk(
                &mut m,
                NextnFeed {
                    tokens: &[1],
                    pos0: 0,
                    hidden: NextnHidden::Host(&vec![0.0; hidden]),
                },
                NextnHead::Full,
                NextnMode::Eager,
            ),
            "without the NextN layer",
        );
        ok &= refused(
            "the MTP window over a load without the layer",
            MtpDraft::<Body>::open(&m, PrefillMode::Batch, StepMode::Eager).map_err(|e| {
                GpuError::Shape {
                    what: "MtpDraft::open",
                    detail: e.to_string(),
                }
            }),
            "an MTP draft on a load without the NextN layer",
        );
        let reference = plain(&mut m, &prompt, PrefillMode::Batch)?;
        // (s) this load's sequence state, for the NextN load to refuse.
        let without = seq_save(&mut m)?;
        drop(m);

        // The NextN load.
        let t = Instant::now();
        let mut m = Body::open_placed_nextn(open(MODEL)?, &np, &inputs, &nextn, 0, levers.host())?;
        m.set_mode(StepMode::Graph);
        let (res, arena) = m
            .body("run")?
            .nextn()
            .map(|n| (n.resident_bytes(), n.arena_bytes()))
            .ok_or("the NextN load holds no NextN layer")?;
        println!(
            "load with layer {}: {:.1} s, {} resident bytes, the layer's {res} and its arena \
             {arena}",
            nextn.index,
            t.elapsed().as_secs_f64(),
            m.resident_bytes()
        );

        ok &= layout_refused(&mut m, &without);
        drop(without);
        ok &= pos_mask(&set, hidden)?;
        ok &= refusals(&mut m, hidden)?;
        ok &= chain_fault(&mut m, nextn.index, hidden)?;
        ok &= pairing(&mut m, &set, &prompt, hidden, inputs.hp.rms_eps)?;
        ok &= oracle(&mut m, &set, hidden)?;

        let with = plain(&mut m, &prompt, PrefillMode::Batch)?;
        let ids_same = with.ids == reference.ids;
        let steps_same = with.steps == reference.steps;
        ok &= ids_same && steps_same;
        println!(
            "(t) the NextN load's plain run = the load without the layer: {N} ids {}, {} steps' \
             logits bit for bit {}",
            verdict(ids_same),
            with.steps.len(),
            verdict(steps_same)
        );
        let ik = ik_stream(&set);
        let agree = ik.iter().zip(&with.ids).take_while(|(a, b)| a == b).count();
        println!(
            "(t) ik's committed stream: {} ids, the plain run's first {agree} of them (printed, \
             not held)",
            ik.len()
        );

        // Each path against its own plain run: a prompt batch past a chunk
        // runs the GEMM, so the two paths' plain runs are not one run's bits.
        let by_steps = plain(&mut m, &prompt, PrefillMode::Steps)?;
        for (path, plain) in [(PrefillMode::Batch, &with), (PrefillMode::Steps, &by_steps)] {
            let (model, w_ok) = drafted(m, &prompt, path, plain, &set)?;
            m = model;
            ok &= w_ok;
        }
        m.reset()?;
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
