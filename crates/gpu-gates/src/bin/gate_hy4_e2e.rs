//! The Hy4-flash end-to-end gate: the whole decode step on one card (the host embedding into the
//! four streams, the engram sites, the identity hyper-connections around every sub-layer, the
//! gated latent attention over a sparse layer's k-pool list or a window layer's ring, each with a
//! sink a head, the dense and the latent routed blocks with the gate-first clamp, the head's fold
//! and the output head) over the fixture file (five layers, the cut [0, 1, 6, 11, 41] of the
//! released file: sparse + dense, sparse + engram + routed, window, window + engram, sparse; window
//! 64, top-k 64), one prompt id a step, against llama.cpp's CPU node dumps of that file (PR
//! #32517, refset family `fx-lcpp-hy4`). The model opens through `Body::open` of `bloomery-gpu-hy4`.
//!
//! The fixture tier only: the released file (124 GB) does not fit one card and this gate opens
//! the whole-card body, so the real tier ends by name before anything loads.
//!
//! For each prompt set of [`SETS`] (set `e`, 93 ids, reaches the selector's pick past the kept
//! pools, the window's truncation and the open pool's forced tail), against the vendor's nodes at
//! the prompt's last position:
//!
//! - `layer_outputs_are_vendors`: the free chain. Layer 0's output (`l_out-0`) within
//!   [`FREE_BAND`] and the logits row (`result_output`) within [`LOGITS_BAND`] by relative L2;
//!   every stage of every layer ([`TAP_STAGES`]), the head's fold and each routed layer's picks
//!   print beside them, so a miss names the first stage that left the band.
//! - `forced_mixes_are_vendors`: each sub-layer's identity mix alone on the vendor's own streams
//!   (the pre, the norm its sub-layer reads, and the post of the vendor's sub-layer output) within
//!   [`MIX_BAND`] of `attn_norm`/`ffn_norm` and `hc_attn_out`/`l_out`; each engram site alone on
//!   the vendor's input streams within [`ENGRAM_BAND`] of `engram_out`; the head's fold of the
//!   vendor's last streams within [`MIX_BAND`] of `hc_head`.
//! - `forced_attention_is_vendors`: each layer's attention alone over the vendor's normed rows of
//!   every position (`attn_norm`, from a fresh walk of the layer's store), its output at the last
//!   position within [`ATTN_BAND`] of `attn_out`. On the sets that dump a sparse layer's selection
//!   (`a` and `e`: `indexer_score`, `indexer_sel`) the selector's pool scores lie within
//!   [`SCORE_BAND`] of the vendor's and its rows are the vendor's unless the vendor's own cut is a
//!   tie at our rounding, and the flash at the last position reads the vendor's rows.
//! - `forced_ffn_is_vendors`: each layer's feed-forward block alone on the vendor's normed row
//!   (`ffn_norm`) within [`FFN_BAND`] of `ffn_out`, and a routed block's picks the vendor's
//!   `ffn_moe_topk` as a set, their weights within [`WEIGHTS_BAND`] of `ffn_moe_weights_scaled`.
//! - `forced_top1_is_vendors`: the vendor's own greedy ids fed back one a step (teacher-forced),
//!   our argmax at each of the greedy table's positions against its id: the disagreements at a
//!   vendor margin of [`MARGIN_FLOOR`] or more are at most the pin the Gaussian error model gives
//!   ([`forced_pin`]).
//!
//! and the clauses that need no vendor:
//!
//! - `opens_the_fixture`: the model opens and carries the cut's five layers of their kinds.
//! - `tier_act_is_gate_first`: the tier's activation of every routed layer of the file's
//!   description is the card leg's gate-first clamp at the layer's limit, and a plain SwiGLU block
//!   is refused by name.
//! - `graph_is_eager`: set `e`'s prompt and 8 greedy steps in graph mode (the chain captured once
//!   and replayed at every position) give the eager chain's ids and last logits row bit for bit.
//!
//! Before the card loads, a set absent from the box, dumped from another file or build, or
//! unfinished ends the gate by name. A step that raises the fault word ends the gate by name.
//!
//! What the free chain can hold. The two engines read the same weights and differ by how they
//! round: the vendor's CPU path quantizes each projection's input row (q8_0 in blocks of 32 for a
//! Q8_0 weight, q8_K in blocks of 256 for a K-quant), ours reads it in f32 or in blocks of 128; a
//! block's step is `amax / 127`, so an input's relative error is about 0.7 %, and a sub-layer
//! chains two to five such sites. The forced clauses measure each block's own share on the
//! vendor's inputs. The free chain compounds it, and three discrete choices turn a rounding gap
//! into a different computation once a margin is narrower than it: the router's 16 of 512 picks
//! (the fixture's random codes leave the 16th and 17th scores close), the selector's kept pools
//! and the argmax. So the free chain holds layer 0 and the logits row loosely, and the forced
//! clauses hold every block.

#[cfg(feature = "hy4")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_hy4_e2e", gate::run())
}

#[cfg(feature = "hy4")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "hy4")]
mod gate {
    use crate::e2e::{rel, tap};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{Gpu, GpuError, GpuModel};
    use bloomery_gpu_gates::prompts::{GreedyRow, MARGIN_FLOOR, compare_forced, std_normal_cdf};
    use bloomery_gpu_gates::tier::{self, Tag, Tier};
    use bloomery_gpu_gates::{
        GateError, RefManifest, RefRow, bits_equal, checks_failed, mask_bits_in, open_split,
        ref_tensor_logical_in, topk_ids_logical_within, verdict,
    };
    use bloomery_gpu_hy4::{Body, Picks, TAP_STAGES, tier as hy4_tier};
    use model::arch::Arch;
    use model::arch::hy4::names::Sub;
    use model::arch::models::{Act as ModelAct, Ffn};
    use refset::arch::hy4::fixture::{self, A, B, C, D, E};
    use refset::arch::prompt_tables::{self, Greedy, Top};
    use refset::ik::plain_is_logical;

    /// The gate's name: the recipe's and the refusals'.
    const NAME: &str = "gate_hy4_e2e";
    /// The recipe that runs it.
    const RECIPE: &str = "gate-gpu-hy4-e2e";

    /// The sets the step path is held to, in the order they run.
    const SETS: &[&str] = &[A, B, C, D, E];

    /// Positions of every store: the longest prompt (set `e`, 93 ids) and its 32 greedy ids fit.
    const CTX: usize = 128;

    /// The experts of the fixture's routed layers: the released file's.
    const N_EXPERT: u32 = 512;

    /// The cut's layers and their kinds, (sparse, engram site), in layer order: the fixture plan of
    /// the released file's layers 0, 1, 6, 11 and 41.
    const KINDS: [(bool, bool); 5] = [
        (true, false),
        (true, true),
        (false, false),
        (false, true),
        (true, false),
    ];
    const LAYERS: usize = KINDS.len();

    /// The free chain's layer 0 output, relative L2 from the vendor's `l_out-0`.
    /// PIN(2026-11-18): 1.6 times the measured maximum 2.18e-2 (five sets, 1.67e-2 to 2.18e-2).
    const FREE_BAND: f64 = 3.5e-2;

    /// The free chain's logits row, relative L2 from the vendor's `result_output`.
    /// PIN(2026-11-18): 1.5 times the measured maximum 0.421 (set `e`; 0.157 to 0.421).
    const LOGITS_BAND: f64 = 6.5e-1;

    /// An identity mix or the head's fold alone (f32 sums of both engines), relative L2.
    /// PIN(2026-11-18): 2.4 times the measured maximum 4.1e-7, under the f32 order bound
    /// `ε·√n` = 3.3e-6 at n = 3072.
    const MIX_BAND: f64 = 1.0e-6;

    /// An engram site alone (one projection of the gathered rows), relative L2 of the streams.
    /// PIN(2026-11-18): 1.5 times the measured maximum 2.67e-3.
    const ENGRAM_BAND: f64 = 4.0e-3;

    /// A layer's attention alone, relative L2 of its output.
    /// PIN(2026-11-18): 1.55 times the measured maximum 1.41e-2 (1.02e-2 to 1.41e-2).
    const ATTN_BAND: f64 = 2.2e-2;

    /// A sparse layer's pool scores alone, relative L2 over the scored pools.
    /// PIN(2026-11-18): 1.55 times the measured maximum 1.62e-2 (set `e`, three sparse layers).
    const SCORE_BAND: f64 = 2.5e-2;

    /// The pools the fixture's selector keeps beside the open one: top-k 64 rows over pools of 4.
    const KEPT: usize = 15;

    /// A layer's feed-forward block alone, relative L2 of its output.
    /// PIN(2026-11-18): 1.5 times the measured maximum 2.12e-2 (1.14e-2 to 2.12e-2).
    const FFN_BAND: f64 = 3.2e-2;

    /// A routed block's pick weights against the vendor's, relative L2 over the picks by id: the
    /// F32 router on the vendor's own row in both engines.
    /// PIN(2026-11-18): 8.8 times the measured maximum 1.14e-7, the f32 order of a 3072-term dot.
    const WEIGHTS_BAND: f64 = 1.0e-6;

    /// One implementation's logit deviation from the other's, in logit units: the σ of the
    /// teacher-forced argmax pin ([`forced_pin`]).
    /// PIN(2026-11-18): above the measured σ̂ of the five sets' last logits rows (0.157 to 0.421).
    const SIGMA_LOGIT: f64 = 0.5;

    /// The vendor clauses a run declares: five a set.
    const FX_ORACLE: usize = 5 * SETS.len();

    /// The greedy steps `graph_is_eager` takes after the prompt.
    const GRAPH_STEPS: usize = 8;

    /// One set's prompt, its tables and the vendor's nodes.
    struct Set {
        name: &'static str,
        prompt: Vec<u32>,
        top: Vec<Top>,
        greedy: Greedy,
        /// The logits row at the prompt's last position (`result_output`).
        logits: Vec<f32>,
        /// The head's fold at the last position (`hc_head`).
        head: Vec<f32>,
        /// The streams the first layer reads at the last position (`hc_init`).
        init: Vec<f32>,
        /// Each stage of each layer ([`TAP_STAGES`]) at the last position; `None` where the
        /// layer's kind has no such node.
        stages: Vec<Vec<Option<Vec<f32>>>>,
        /// Each layer's normed attention rows of every position (`attn_norm-L`).
        attn_rows: Vec<Vec<Vec<f32>>>,
        /// Each routed layer's picks and their weights at the last position (`ffn_moe_topk-L`,
        /// `ffn_moe_weights_scaled-L`); `None` on a dense layer.
        picks: Vec<Option<Picks>>,
        /// Each sparse layer's selection at the last position where the set dumps it: the cache
        /// rows its mask leaves visible (`indexer_sel-L`) and the pool scores
        /// (`indexer_score-L`, the unscored pools `-inf`).
        sel: Vec<Option<(Vec<u32>, Vec<f32>)>>,
    }

    impl Set {
        /// Stage `stage` of layer `l` at the last position.
        fn at(&self, l: usize, stage: &str) -> Result<&[f32], GateError> {
            let i = TAP_STAGES
                .iter()
                .position(|s| *s == stage)
                .ok_or_else(|| format!("{stage} is not a tapped stage"))?;
            self.stages[l][i]
                .as_deref()
                .ok_or_else(|| format!("{}: {stage}-{l} absent", self.name).into())
        }

        /// The streams layer `l`'s engram site reads: the previous layer's output, or the first
        /// layer's input.
        fn site_input(&self, l: usize) -> Result<&[f32], GateError> {
            if l == 0 {
                Ok(&self.init)
            } else {
                self.at(l - 1, "l_out")
            }
        }

        /// The streams layer `l`'s attention mix reads: the site's output on a site layer.
        fn attn_input(&self, l: usize) -> Result<&[f32], GateError> {
            if KINDS[l].1 {
                self.at(l, "engram_out")
            } else {
                self.site_input(l)
            }
        }
    }

    /// `all` cut into its `tokens` rows.
    fn rows_of(all: &[f32], tokens: usize, what: &str) -> Result<Vec<Vec<f32>>, GateError> {
        if tokens == 0 || !all.len().is_multiple_of(tokens) {
            return Err(format!("{what}: {} values for {tokens} positions", all.len()).into());
        }
        Ok(all
            .chunks(all.len() / tokens)
            .map(<[f32]>::to_vec)
            .collect())
    }

    /// The last `ne[0]` values of node `n`: a node after the dump's crop to the output rows.
    fn last_output(man: &RefManifest, n: &str, set: &str) -> Result<Vec<f32>, GateError> {
        let row = man.tensor(n, 0)?;
        let all = ref_tensor_logical_in(&man.dir, row)?;
        let w = usize::try_from(row.ne[0])?;
        if w == 0 || !all.len().is_multiple_of(w) {
            return Err(format!("{set}: {n} is {:?}", row.ne).into());
        }
        Ok(all[all.len() - w..].to_vec())
    }

    /// The last of `tokens` rows of the selector's scores node `row`: an f32 row whose file is its
    /// logical order and holds `-inf` past the scored pools, so it is read whole rather than
    /// through the finite reader.
    fn scores_last(man: &RefManifest, row: &RefRow, tokens: usize) -> Result<Vec<f32>, GateError> {
        if row.ty != "f32" || !plain_is_logical(row) {
            return Err(format!("{}: a {} view, want a plain f32 row", row.name, row.ty).into());
        }
        let raw = std::fs::read(man.dir.join(row.file_name()))?;
        if Some(raw.len() as u64) != row.count().checked_mul(4) {
            return Err(format!(
                "{}: {} bytes for {} values",
                row.name,
                raw.len(),
                row.count()
            )
            .into());
        }
        let all: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        Ok(rows_of(&all, tokens, &row.name)?.pop().unwrap_or_default())
    }

    fn open_set(clause: &str, name: &'static str) -> Result<Set, GateError> {
        let path = tier::fixture_set(clause, &fixture::LCPP, name)?;
        let man = RefManifest::open(&path, &fixture::LCPP)?;
        let prompt = prompt_tables::tokens(&man)?.to_vec();
        let tokens = prompt.len();
        let last = |n: &str| -> Result<Option<Vec<f32>>, GateError> {
            if man.tensor(n, 0).is_err() {
                return Ok(None);
            }
            Ok(rows_of(&tap(&man, n)?, tokens, n)?.pop())
        };
        let stages = (0..LAYERS)
            .map(|l| {
                TAP_STAGES
                    .iter()
                    .map(|stage| last(&format!("{stage}-{l}")))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<_, _>>()?;
        let attn_rows = (0..LAYERS)
            .map(|l| {
                let n = format!("attn_norm-{l}");
                rows_of(&tap(&man, &n)?, tokens, &n)
            })
            .collect::<Result<_, _>>()?;
        let picks = (0..LAYERS)
            .map(|l| -> Result<_, GateError> {
                let Ok(row) = man.tensor(&format!("ffn_moe_topk-{l}"), 0) else {
                    return Ok(None);
                };
                let ids = topk_ids_logical_within(&man, row, N_EXPERT)?
                    .into_iter()
                    .map(|i| u32::try_from(i).map_err(|e| GateError::from(e.to_string())))
                    .collect::<Result<Vec<_>, _>>()?;
                let wname = format!("ffn_moe_weights_scaled-{l}");
                let ws = last(&wname)?.ok_or_else(|| format!("{name}: {wname} absent"))?;
                let ids = ids[ids.len() - ws.len()..].to_vec();
                Ok(Some((ids, ws)))
            })
            .collect::<Result<_, _>>()?;
        let sel = (0..LAYERS)
            .map(|l| -> Result<_, GateError> {
                let (m_name, s_name) = (format!("indexer_sel-{l}"), format!("indexer_score-{l}"));
                let (Ok(mask), Ok(score_row)) = (man.tensor(&m_name, 0), man.tensor(&s_name, 0))
                else {
                    return Ok(None);
                };
                let scores = scores_last(&man, score_row, tokens)?;
                let bits = mask_bits_in(&man.dir, mask)?;
                let width = usize::try_from(mask.ne[0])?;
                let start = bits.len().checked_sub(width).ok_or_else(|| {
                    format!(
                        "{name}: {m_name} holds {} cells, a row is {width}",
                        bits.len()
                    )
                })?;
                let rows = (0u32..)
                    .zip(&bits[start..])
                    .filter(|(_, b)| **b == 0)
                    .map(|(i, _)| i)
                    .collect();
                Ok(Some((rows, scores)))
            })
            .collect::<Result<_, _>>()?;
        let set = Set {
            name,
            prompt,
            top: prompt_tables::topk(&man)?,
            greedy: prompt_tables::greedy(&man)?,
            logits: last_output(&man, "result_output", name)?,
            head: last_output(&man, "hc_head", name)?,
            init: last("hc_init")?.ok_or_else(|| format!("{name}: hc_init absent"))?,
            stages,
            attn_rows,
            picks,
            sel,
        };
        if set.greedy.ids.is_empty() || set.greedy.margins.len() != set.greedy.ids.len() {
            return Err(format!(
                "{name}: a greedy table of {} ids and {} margins",
                set.greedy.ids.len(),
                set.greedy.margins.len()
            )
            .into());
        }
        let need = set.prompt.len() + set.greedy.ids.len();
        if need > CTX {
            return Err(format!("{name}: {need} positions, the gate's stores hold {CTX}").into());
        }
        Ok(set)
    }

    /// Picks as `(id, weight)` sorted by id.
    fn by_id((ids, ws): (&[u32], &[f32])) -> Vec<(u32, f32)> {
        let mut v: Vec<(u32, f32)> = ids.iter().copied().zip(ws.iter().copied()).collect();
        v.sort_unstable_by_key(|p| p.0);
        v
    }

    /// Whether `ours` picked the vendor's experts, and their weights' distance by id (infinite
    /// when the sets differ).
    fn picks_match(ours: (&[u32], &[f32]), theirs: &Picks) -> (bool, f64, String) {
        let (a, b) = (by_id(ours), by_id((&theirs.0, &theirs.1)));
        let same = a.iter().map(|p| p.0).eq(b.iter().map(|p| p.0));
        let wa: Vec<f32> = a.iter().map(|p| p.1).collect();
        let wb: Vec<f32> = b.iter().map(|p| p.1).collect();
        let wr = if same { rel(&wa, &wb) } else { f64::INFINITY };
        let diff = if same {
            String::new()
        } else {
            let only = |x: &[(u32, f32)], y: &[(u32, f32)]| -> Vec<u32> {
                x.iter()
                    .map(|p| p.0)
                    .filter(|i| !y.iter().any(|q| q.0 == *i))
                    .collect()
            };
            format!(
                " ours_only={:?} vendor_only={:?}",
                only(&a, &b),
                only(&b, &a)
            )
        };
        (same, wr, diff)
    }

    /// The root mean square of the logits' difference: one logit's deviation, σ̂.
    fn sigma_hat(ours: &[f32], theirs: &[f32]) -> f64 {
        let s: f64 = ours
            .iter()
            .zip(theirs)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
            .sum();
        (s / ours.len().max(1) as f64).sqrt()
    }

    /// Clause `layer_outputs_are_vendors` of set `s` from a reset model, taps armed: the prompt,
    /// then every stage printed and layer 0's output and the logits row held. Returns the
    /// clause's verdict and the prompt's argmax.
    fn layer_outputs_are_vendors(
        m: &mut GpuModel<Body>,
        s: &Set,
    ) -> Result<(bool, u32), GateError> {
        m.reset()?;
        let first = m.step(&s.prompt)?;
        let row = m.logits()?;
        if !tier::run_clause(
            &format!("layer_outputs_are_vendors set={}", s.name),
            Tag::FixtureOracle,
        )? {
            return Ok((true, first));
        }
        let ours = m.body(NAME)?.taps(m.gpu().stream())?;
        let stages = TAP_STAGES.len();
        for (l, theirs) in s.stages.iter().enumerate() {
            let mut cells = Vec::new();
            for (i, node) in theirs.iter().enumerate() {
                let Some(node) = node else {
                    cells.push(format!("{}=-", TAP_STAGES[i]));
                    continue;
                };
                let have = &ours.stages[l * stages + i];
                let r = if node.len() <= have.len() {
                    rel(&have[..node.len()], node)
                } else {
                    f64::INFINITY
                };
                cells.push(format!("{}={r:.2e}", TAP_STAGES[i]));
            }
            let picks = match &s.picks[l] {
                Some(p) => {
                    let (ids, ws) = &ours.picks[l];
                    let (same, wr, diff) = picks_match((ids, ws), p);
                    format!(" picks_same={same} weights_rel={wr:.2e}{diff}")
                }
                None => String::new(),
            };
            println!(
                "  free set={} layer={l}: {}{picks}",
                s.name,
                cells.join(" ")
            );
        }
        let l0 = s.at(0, "l_out")?;
        let l0_rel = rel(&ours.stages[TAP_STAGES.len() - 1][..l0.len()], l0);
        let head_rel = rel(&ours.head, &s.head);
        let logits_rel = rel(&row, &s.logits);
        let theirs = s.top.first().map_or(u32::MAX, |t| t.id);
        let pass = l0_rel <= FREE_BAND && logits_rel <= LOGITS_BAND;
        println!(
            "layer_outputs_are_vendors set={} l_out0_rel={l0_rel:.3e} band={FREE_BAND:.1e} \
             hc_head_rel={head_rel:.3e} logits_rel={logits_rel:.3e} band={LOGITS_BAND:.1e} \
             sigma_hat={:.3} prompt_top1_ours={first} vendor={theirs} {}",
            s.name,
            sigma_hat(&row, &s.logits),
            verdict(pass)
        );
        Ok((pass, first))
    }

    /// Clause `forced_mixes_are_vendors` of set `s`, right after its prompt (the engram sites read
    /// the rows the last refresh gathered): every identity mix, engram site and the head's fold
    /// alone on the vendor's streams.
    fn forced_mixes_are_vendors(m: &mut GpuModel<Body>, s: &Set) -> Result<bool, GateError> {
        if !tier::run_clause(
            &format!("forced_mixes_are_vendors set={}", s.name),
            Tag::FixtureOracle,
        )? {
            return Ok(true);
        }
        let (gpu, w, body) = m.body_parts(NAME)?;
        let mut mix_worst = 0.0f64;
        let mut engram_worst = 0.0f64;
        for (l, &(_, site)) in KINDS.iter().enumerate() {
            let mut cells = Vec::new();
            if site {
                let out = body.engram_alone(gpu, w, l, s.site_input(l)?)?;
                let r = rel(&out, s.at(l, "engram_out")?);
                engram_worst = engram_worst.max(r);
                cells.push(format!("engram_out={r:.2e}"));
            }
            let subs = [
                (
                    Sub::Attn,
                    s.attn_input(l)?,
                    "attn_out",
                    "attn_norm",
                    "hc_attn_out",
                ),
                (
                    Sub::Ffn,
                    s.at(l, "hc_attn_out")?,
                    "ffn_out",
                    "ffn_norm",
                    "l_out",
                ),
            ];
            for (sub, streams, out, normed, post) in subs {
                let got = body.mix_alone(gpu, w, (l, sub), streams, s.at(l, out)?)?;
                let (rn, rp) = (
                    rel(&got.normed, s.at(l, normed)?),
                    rel(&got.streams, s.at(l, post)?),
                );
                mix_worst = mix_worst.max(rn).max(rp);
                cells.push(format!("{normed}={rn:.2e} {post}={rp:.2e}"));
            }
            println!("  mixes set={} layer={l}: {}", s.name, cells.join(" "));
        }
        let fold = body.head_fold_alone(gpu, w, s.at(LAYERS - 1, "l_out")?)?;
        let head_rel = rel(&fold, &s.head);
        mix_worst = mix_worst.max(head_rel);
        let pass = mix_worst <= MIX_BAND && engram_worst <= ENGRAM_BAND;
        println!(
            "forced_mixes_are_vendors set={} hc_head={head_rel:.2e} mix_worst={mix_worst:.3e} \
             band={MIX_BAND:.1e} engram_worst={engram_worst:.3e} band={ENGRAM_BAND:.1e} {}",
            s.name,
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `forced_attention_is_vendors` of set `s`: every layer's attention alone over the
    /// vendor's normed rows of every position. Where the set dumps a sparse layer's selection,
    /// the flash at the last position reads the vendor's rows, and the selector's own pool scores
    /// are held within [`SCORE_BAND`] and its rows to the vendor's, unless the vendor's own cut
    /// (its last kept pool's score over the first dropped one's) is within twice our scores'
    /// widest distance from its: a tie the two engines' rounding may split either way.
    fn forced_attention_is_vendors(m: &mut GpuModel<Body>, s: &Set) -> Result<bool, GateError> {
        if !tier::run_clause(
            &format!("forced_attention_is_vendors set={}", s.name),
            Tag::FixtureOracle,
        )? {
            return Ok(true);
        }
        let (gpu, w, body) = m.body_parts(NAME)?;
        let mut cells = Vec::new();
        let (mut worst, mut score_worst, mut ok) = (0.0f64, 0.0f64, true);
        for l in 0..LAYERS {
            let theirs = s.sel[l].as_ref();
            let (out, ours) =
                body.attention_alone(gpu, w, l, &s.attn_rows[l], theirs.map(|t| t.0.as_slice()))?;
            let r = rel(&out, s.at(l, "attn_out")?);
            worst = worst.max(r);
            cells.push(format!("{l}={r:.2e}"));
            let (Some((rows, scores)), Some((mut picked, our_scores))) = (theirs, ours) else {
                continue;
            };
            picked.sort_unstable();
            let scored = scores.iter().take_while(|v| v.is_finite()).count();
            let (mut dev, mut margin, mut sr) = (0.0f32, f32::INFINITY, 0.0f64);
            if scored > KEPT {
                let (a, b) = (&our_scores[..scored], &scores[..scored]);
                sr = rel(a, b);
                dev = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0, f32::max);
                let mut sorted = b.to_vec();
                sorted.sort_unstable_by(|x, y| y.total_cmp(x));
                margin = sorted[KEPT - 1] - sorted[KEPT];
            }
            score_worst = score_worst.max(sr);
            let same = picked == *rows;
            let tie = !same && margin <= 2.0 * dev;
            ok &= same || tie;
            println!(
                "  selection set={} layer={l} rows ours={} vendor={} same={same} tie={tie} \
                 scored_pools={scored} scores_rel={sr:.2e} widest={dev:.3e} vendor_cut={margin:.3e}",
                s.name,
                picked.len(),
                rows.len()
            );
        }
        let pass = ok && worst <= ATTN_BAND && score_worst <= SCORE_BAND;
        println!(
            "forced_attention_is_vendors set={} attn_out_rel=[{}] worst={worst:.3e} band={ATTN_BAND:.1e} \
             scores_worst={score_worst:.3e} band={SCORE_BAND:.1e} {}",
            s.name,
            cells.join(" "),
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `forced_ffn_is_vendors` of set `s`: every layer's feed-forward block alone on the
    /// vendor's normed row.
    fn forced_ffn_is_vendors(m: &mut GpuModel<Body>, s: &Set) -> Result<bool, GateError> {
        if !tier::run_clause(
            &format!("forced_ffn_is_vendors set={}", s.name),
            Tag::FixtureOracle,
        )? {
            return Ok(true);
        }
        let (gpu, w, body) = m.body_parts(NAME)?;
        let mut ok = true;
        let mut worst = 0.0f64;
        let mut cells = Vec::new();
        for l in 0..LAYERS {
            let (out, picks) = body.ffn_alone(gpu, w, l, s.at(l, "ffn_norm")?)?;
            let r = rel(&out, s.at(l, "ffn_out")?);
            worst = worst.max(r);
            match (&picks, &s.picks[l]) {
                (Some((ids, ws)), Some(theirs)) => {
                    let (same, wr, diff) = picks_match((ids, ws), theirs);
                    ok &= same && wr <= WEIGHTS_BAND;
                    cells.push(format!(
                        "{l}={r:.2e} picks_same={same} weights_rel={wr:.2e}{diff}"
                    ));
                }
                (None, None) => cells.push(format!("{l}={r:.2e}")),
                _ => {
                    return Err(format!(
                        "{}: layer {l} routes on one side only (ours {}, the vendor's {})",
                        s.name,
                        picks.is_some(),
                        s.picks[l].is_some()
                    )
                    .into());
                }
            }
        }
        let pass = ok && worst <= FFN_BAND;
        println!(
            "forced_ffn_is_vendors set={} [{}] worst={worst:.3e} band={FFN_BAND:.1e} weights_band={WEIGHTS_BAND:.1e} {}",
            s.name,
            cells.join("; "),
            verdict(pass)
        );
        Ok(pass)
    }

    /// The disagreements the Gaussian error of two implementations predicts at the clear
    /// positions of `margins` (those at or past [`MARGIN_FLOOR`]), and the pin that follows: the
    /// expected count plus three standard deviations of it, rounded up.
    ///
    /// [derived]: at the vendor's margin `m` the argmax flips with probability
    /// Φ(−m / (√2 σ)), σ being [`SIGMA_LOGIT`].
    fn forced_pin(margins: &[f32]) -> (f64, usize) {
        let sigma_pair = SIGMA_LOGIT * std::f64::consts::SQRT_2;
        let expected: f64 = margins
            .iter()
            .filter(|&&m| m >= MARGIN_FLOOR)
            .map(|&m| std_normal_cdf(-f64::from(m) / sigma_pair))
            .sum();
        (expected, (expected + 3.0 * expected.sqrt()).ceil() as usize)
    }

    /// Clause `forced_top1_is_vendors` of set `s` from a reset model: the prompt, then the
    /// vendor's greedy ids fed back one a step.
    fn forced_top1_is_vendors(m: &mut GpuModel<Body>, s: &Set) -> Result<bool, GateError> {
        if !tier::run_clause(
            &format!("forced_top1_is_vendors set={}", s.name),
            Tag::FixtureOracle,
        )? {
            return Ok(true);
        }
        m.reset()?;
        let mut table = vec![m.step(&s.prompt)?];
        for &id in &s.greedy.ids[..s.greedy.ids.len() - 1] {
            table.push(m.step(&[id])?);
        }
        let theirs = s.top.first().map_or(u32::MAX, |t| t.id);
        let vendor = GreedyRow {
            id: 0,
            n_tokens: s.prompt.len(),
            argmax: theirs,
            top5: s.top.iter().take(5).map(|t| t.id).collect(),
            gen_ids: s.greedy.ids.clone(),
            gen_margins: s.greedy.margins.clone(),
        };
        let report = compare_forced(&[table], &[vendor], MARGIN_FLOOR);
        let (expected, pin) = forced_pin(&s.greedy.margins);
        let pass = report.disagree_ge_floor <= pin;
        println!(
            "forced_top1_is_vendors set={} positions={} disagree={} buckets={:?} ge_floor={} \
             expected {expected:.2} pin {pin} {}",
            s.name,
            report.positions,
            report.disagree.len(),
            report.buckets,
            report.disagree_ge_floor,
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `graph_is_eager`: set `s`'s prompt and [`GRAPH_STEPS`] greedy steps in graph mode
    /// give the eager chain's ids and last logits row bit for bit.
    fn graph_is_eager(m: &mut GpuModel<Body>, s: &Set) -> Result<bool, GateError> {
        let run =
            |m: &mut GpuModel<Body>, mode: StepMode| -> Result<(Vec<u32>, Vec<f32>), GateError> {
                m.set_mode(mode);
                m.reset()?;
                let mut ids = vec![m.step(&s.prompt)?];
                while ids.len() <= GRAPH_STEPS {
                    let last = ids[ids.len() - 1];
                    ids.push(m.step(&[last])?);
                }
                Ok((ids, m.logits()?))
            };
        let eager = run(m, StepMode::Eager)?;
        let graph = run(m, StepMode::Graph)?;
        m.set_mode(StepMode::Eager);
        let same_bits = bits_equal(&eager.1, &graph.1);
        let pass = eager.0 == graph.0 && same_bits;
        println!(
            "graph_is_eager set={} eager={:?} graph={:?} logits_same_bits={same_bits} {}",
            s.name,
            eager.0,
            graph.0,
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `tier_act_is_gate_first` over the file's description `spec`: every routed layer's
    /// tier activation is the gate-first clamp at its limit, and a plain SwiGLU block is refused
    /// by name.
    fn tier_act_is_gate_first(spec: &model::arch::models::ModelSpec) -> Result<bool, GateError> {
        let mut ok = true;
        let mut routed = 0usize;
        let mut plain = None;
        for (l, layer) in spec.layers.iter().enumerate() {
            let Ffn::Moe(moe) = &layer.ffn else {
                continue;
            };
            routed += 1;
            let ModelAct::SwiGluGateFirst { limit } = moe.act else {
                println!(
                    "tier_act_is_gate_first layer={l} act={} FAIL",
                    moe.act.what()
                );
                ok = false;
                continue;
            };
            let act = hy4_tier::tier_act(moe)?;
            let pass = act == bloomery_gpu::kquant::Act::SwigluGateClamp { limit };
            println!("  tier_act layer={l} {act:?} {}", verdict(pass));
            ok &= pass;
            plain.get_or_insert_with(|| {
                let mut m = moe.clone();
                m.act = ModelAct::SwiGlu { limit: Some(limit) };
                m
            });
        }
        let refused = match plain.as_ref().map(hy4_tier::tier_act) {
            Some(Err(GpuError::Shape { what, detail })) => {
                what == "hy4 tier" && detail.contains("gate-first")
            }
            _ => false,
        };
        let pass = ok && routed > 0 && refused;
        println!(
            "tier_act_is_gate_first routed_layers={routed} plain_swiglu_refused={refused} {}",
            verdict(pass)
        );
        Ok(pass)
    }

    /// The vendor clauses of set `s`, in the order their state needs: the free prompt (its last
    /// refresh is the engram rows the forced mixes read), the blocks alone, then the forced
    /// argmax from a reset.
    fn vendor_clauses(m: &mut GpuModel<Body>, s: &Set) -> Result<bool, GateError> {
        let (mut ok, _) = layer_outputs_are_vendors(m, s)?;
        ok &= forced_mixes_are_vendors(m, s)?;
        ok &= forced_attention_is_vendors(m, s)?;
        ok &= forced_ffn_is_vendors(m, s)?;
        ok &= forced_top1_is_vendors(m, s)?;
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        if Tier::from_env()? != Tier::Fixture {
            return Err(format!(
                "{NAME}: the released file (124 GB) does not fit one card and this gate opens the \
                 whole-card body, so the gate runs the fixture tier: BLOOMERY_TIER=fixture just \
                 {RECIPE}"
            )
            .into());
        }
        let sets: Vec<Set> = SETS
            .iter()
            .map(|&n| open_set("the vendor clauses", n))
            .collect::<Result<_, _>>()?;
        let spec = model::arch::hy4::spec::read(&open_split(Arch::Hy4, RECIPE)?)
            .map_err(|e| format!("{NAME}: the fixture's description: {e}"))?
            .spec;
        let file = open_split(Arch::Hy4, RECIPE)?;
        let t = std::time::Instant::now();
        let mut m = match Body::open(Gpu::new()?, file, CTX) {
            Ok(m) => m,
            Err(e) => {
                println!("opens_the_fixture {e} {}", verdict(false));
                return Err(e.into());
            }
        };
        let kinds = m.body(NAME)?.kinds();
        let opened = kinds == KINDS;
        println!(
            "opens_the_fixture resident_bytes={} store_bytes={} ctx={CTX} kinds={kinds:?} in {:.1} s {}",
            m.resident_bytes(),
            m.body(NAME)?.store_bytes(),
            t.elapsed().as_secs_f64(),
            verdict(opened)
        );
        if !opened {
            return Err(checks_failed());
        }
        let mut ok = tier_act_is_gate_first(&spec)?;
        {
            let (gpu, _, body) = m.body_parts(NAME)?;
            body.arm_taps(gpu)?;
        }
        for s in &sets {
            ok &= vendor_clauses(&mut m, s)?;
        }
        tier::expect_fixture_oracle(NAME, FX_ORACLE)?;
        m.body_parts(NAME)?.2.disarm_taps();
        ok &= graph_is_eager(&mut m, &sets[4])?;
        println!("{}", tier::tally_line());
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} on every set of {SETS:?}: layer 0 within {FREE_BAND:.1e} and the \
             logits within {LOGITS_BAND:.1e}; every mix within {MIX_BAND:.1e}, every engram site \
             within {ENGRAM_BAND:.1e}, every attention within {ATTN_BAND:.1e}, every feed-forward \
             block within {FFN_BAND:.1e} with the vendor's picks; the teacher-forced argmax \
             disagreements within the pin; the graph is the eager chain; the tier runs the \
             gate-first clamp"
        );
        Ok(())
    }
}
