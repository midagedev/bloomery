//! The prompt-set side of the GPU engine's end-to-end gate: ik's greedy
//! answers on disk, and the comparer that judges our continuation against
//! them. Host-only — no device code, no `gpu` feature.
//!
//! The reference files come from `tools/ref/argmax.sh`: the plain argmax set
//! (`just argmax-ref-cuda`, five columns) and the greedy set (`--gen N`, seven
//! columns, `just greedy-ref-cuda`). Both engines read token ids from the same
//! `tools/ref/prompts.tsv`; neither tokenizes — a tokenizer difference would
//! surface as a wrong token id and read as a kernel bug.
//!
//! There are three comparers, and they answer different questions.
//!
//! `compare_greedy` judges a FREE-RUNNING continuation, and only its FIRST
//! difference. After that difference the two continuations are conditioned on
//! different text, so every later token answers a different question and
//! carries no evidence. It reports the first differing index, the reference's
//! own top1-top2 margin there, and one of three classes; what happens past
//! that index never enters the verdict. The cost is its sample size: one
//! event per prompt.
//!
//! `compare_forced` judges a TEACHER-FORCED table, where our argmax at every
//! step was taken with the reference's own tokens fed as input. Both sides
//! then stand at the same position on the same text at every step, so every
//! step is evidence and `gen_margins[s]` is the margin at the place it is
//! read. The unit is the position, not the prompt.
//!
//! `compare_forced_exact` is the same teacher-forced table judged against a
//! different answer: the f64 truth `exact_ref` computes on the reference's
//! path (`just build-exact-forced`, read by [`read_exact`]). The input path
//! is still the reference's tokens — both engines stand on the same text —
//! but the right answer and its margin at each position are the model's own
//! exact ones, so a position the reference itself gets wrong is not counted
//! against us, and one where we agree with the reference's wrong answer is.
//!
//! `sigma_forced` reads the same table and truth as a continuous quantity:
//! the RMS distance of our margin from the exact margin over every position,
//! and the clear-position disagreement count that width predicts.

use std::path::Path;

use crate::GateError;

/// One row of `tools/ref/prompts.tsv`: `id <TAB> text <TAB> comma-separated
/// token ids` (comments start with `#`).
#[derive(Debug, Clone)]
pub struct PromptRow {
    pub id: usize,
    pub text: String,
    pub tokens: Vec<u32>,
}

/// One data row of a greedy reference file (`argmax_ref --gen N` output).
///
/// `gen_ids` is the greedy continuation starting at the prompt's argmax
/// (`gen_ids[0] == argmax`), one entry per step. A continuation that stopped
/// on EOS is shorter than N: the EOS token itself is recorded and the shorter
/// length is the record of the stop. `gen_margins` is each step's top1-top2
/// logit margin under the writer's order (logit descending, id ascending),
/// one entry per generated token.
#[derive(Debug, Clone)]
pub struct GreedyRow {
    pub id: usize,
    pub n_tokens: usize,
    pub argmax: u32,
    pub top5: Vec<u32>,
    pub gen_ids: Vec<u32>,
    pub gen_margins: Vec<f32>,
}

/// `tools/ref/prompts.tsv` rows in file order. Comment and empty lines are
/// skipped; any other line must be the three tab-separated fields.
pub fn read_prompts(path: &Path) -> Result<Vec<PromptRow>, GateError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("read_prompts: cannot read {}: {e}", path.display()))?;
    let mut rows = Vec::new();
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 3 {
            return Err(format!(
                "read_prompts: {}: row has {} fields, want id<tab>text<tab>ids: {line:?}",
                path.display(),
                f.len()
            )
            .into());
        }
        let bad = |e: std::num::ParseIntError| {
            format!("read_prompts: {}: {line:?}: {e}", path.display()).into()
        };
        let tokens = f[2]
            .split(',')
            .map(|s| s.trim().parse::<u32>().map_err(bad))
            .collect::<Result<Vec<_>, GateError>>()?;
        if tokens.is_empty() {
            return Err(format!(
                "read_prompts: {}: empty token list: {line:?}",
                path.display()
            )
            .into());
        }
        rows.push(PromptRow {
            id: f[0].trim().parse::<usize>().map_err(bad)?,
            text: f[1].to_string(),
            tokens,
        });
    }
    if rows.is_empty() {
        return Err(format!("read_prompts: no rows in {}", path.display()).into());
    }
    Ok(rows)
}

/// Greedy reference rows (`argmax_ref --gen N` output) in file order. Only
/// the seven-column `--gen` format parses: a five-column argmax file is a
/// different artifact and is rejected, not read as zero generated tokens —
/// an empty `gen_ids` would compare as vacuously identical.
///
/// Enforces the writer's own duplication contract, `gen_ids[0] == argmax`,
/// so a truncated or hand-edited file fails here instead of at the verdict.
pub fn read_greedy(path: &Path) -> Result<Vec<GreedyRow>, GateError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("read_greedy: cannot read {}: {e}", path.display()))?;
    let mut rows = Vec::new();
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 7 {
            return Err(format!(
                "read_greedy: {}: row has {} fields, want the 7-column --gen format \
                 (id n_tokens argmax top5_ids top5_logits gen_ids gen_margins): {line:?}",
                path.display(),
                f.len()
            )
            .into());
        }
        let bad_i = |e: std::num::ParseIntError| {
            format!("read_greedy: {}: {line:?}: {e}", path.display()).into()
        };
        let bad_f = |e: std::num::ParseFloatError| {
            format!("read_greedy: {}: {line:?}: {e}", path.display()).into()
        };
        let ids = |s: &str| {
            s.split(',')
                .map(|t| t.trim().parse::<u32>().map_err(bad_i))
                .collect::<Result<Vec<u32>, GateError>>()
        };
        let floats = |s: &str| {
            s.split(',')
                .map(|t| t.trim().parse::<f32>().map_err(bad_f))
                .collect::<Result<Vec<f32>, GateError>>()
        };
        let top5 = ids(f[3])?;
        let logits = floats(f[4])?;
        let gen_ids = ids(f[5])?;
        let gen_margins = floats(f[6])?;
        if top5.len() != 5 || logits.len() != 5 {
            return Err(format!(
                "read_greedy: {}: top5 columns hold {} and {} values, want 5: {line:?}",
                path.display(),
                top5.len(),
                logits.len()
            )
            .into());
        }
        if gen_ids.is_empty() || gen_ids.len() != gen_margins.len() {
            return Err(format!(
                "read_greedy: {}: gen_ids ({}) and gen_margins ({}) must be non-empty \
                 and the same length: {line:?}",
                path.display(),
                gen_ids.len(),
                gen_margins.len()
            )
            .into());
        }
        let argmax = f[2].trim().parse::<u32>().map_err(bad_i)?;
        if gen_ids[0] != argmax {
            return Err(format!(
                "read_greedy: {}: gen_ids[0] {} != argmax {argmax} — the file is not \
                 what argmax_ref --gen writes: {line:?}",
                path.display(),
                gen_ids[0]
            )
            .into());
        }
        rows.push(GreedyRow {
            id: f[0].trim().parse::<usize>().map_err(bad_i)?,
            n_tokens: f[1].trim().parse::<usize>().map_err(bad_i)?,
            argmax,
            top5,
            gen_ids,
            gen_margins,
        });
    }
    if rows.is_empty() {
        return Err(format!("read_greedy: no rows in {}", path.display()).into());
    }
    Ok(rows)
}

/// The verdict for one prompt's continuation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GreedyClass {
    /// No difference over the compared span (the shorter of the two
    /// continuations). Lengths still ride along in the row: a span that
    /// matches while one side ran past the other's EOS is Identical here and
    /// is the consumer's stopping contract to judge, not this comparer's.
    Identical,
    /// The first difference sits where the reference's own top1-top2 margin
    /// is under `margin_floor` — a legitimate near-tie flip, not a fault.
    NearTie,
    /// The first difference sits where the reference was clear by at least
    /// `margin_floor`.
    Diverged,
}

/// `compare_greedy`'s per-prompt row.
#[derive(Debug, Clone)]
pub struct GreedyPromptResult {
    pub id: usize,
    /// First index where the tokens differ, `None` when they never do over
    /// the compared span.
    pub first_diff: Option<usize>,
    /// The reference's top1-top2 margin at `first_diff`, `None` when
    /// `first_diff` is `None`.
    pub ref_margin: Option<f32>,
    pub class: GreedyClass,
    /// The continuation's token count and the reference's `gen_ids` length.
    pub n_ours: usize,
    pub n_ref: usize,
}

/// The whole set: one row per prompt plus the class totals.
#[derive(Debug, Clone)]
pub struct GreedyReport {
    pub rows: Vec<GreedyPromptResult>,
    pub n_identical: usize,
    pub n_near_tie: usize,
    pub n_diverged: usize,
}

/// Judge our greedy continuations against the reference, prompt by prompt.
///
/// `ours[i]` is our continuation for `reference[i]`, indexed like
/// `gen_ids` (element 0 is the prompt's next token). Only the FIRST
/// difference is judged — see the module docs. A reference row that stopped
/// early on EOS is compared over its own, shorter length.
///
/// # Panics
///
/// Panics when `ours.len() != reference.len()`: every reference row must be
/// answered, and silently skipping some would under-report divergences.
pub fn compare_greedy(
    ours: &[Vec<u32>],
    reference: &[GreedyRow],
    margin_floor: f32,
) -> GreedyReport {
    assert_eq!(
        ours.len(),
        reference.len(),
        "compare_greedy: {} continuations for {} reference rows",
        ours.len(),
        reference.len()
    );
    let mut rows = Vec::with_capacity(reference.len());
    for (cont, r) in ours.iter().zip(reference) {
        let span = cont.len().min(r.gen_ids.len());
        let first_diff = (0..span).find(|&i| cont[i] != r.gen_ids[i]);
        let (first_diff, ref_margin, class) = match first_diff {
            None => (None, None, GreedyClass::Identical),
            Some(i) => {
                let m = r.gen_margins[i];
                let class = if m < margin_floor {
                    GreedyClass::NearTie
                } else {
                    GreedyClass::Diverged
                };
                (Some(i), Some(m), class)
            }
        };
        rows.push(GreedyPromptResult {
            id: r.id,
            first_diff,
            ref_margin,
            class,
            n_ours: cont.len(),
            n_ref: r.gen_ids.len(),
        });
    }
    let n_identical = rows
        .iter()
        .filter(|r| r.class == GreedyClass::Identical)
        .count();
    let n_near_tie = rows
        .iter()
        .filter(|r| r.class == GreedyClass::NearTie)
        .count();
    let n_diverged = rows
        .iter()
        .filter(|r| r.class == GreedyClass::Diverged)
        .count();
    GreedyReport {
        rows,
        n_identical,
        n_near_tie,
        n_diverged,
    }
}

/// One teacher-forced position: our argmax where the reference's own path
/// was fed, against the ruler's answer there and its margin. The ruler is
/// the reference itself under [`compare_forced`] and the exact truth under
/// [`compare_forced_exact`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForcedPos {
    /// The prompt's id, from the ruler's row.
    pub id: usize,
    /// The step inside the continuation, indexed like `gen_ids`.
    pub step: usize,
    /// Our argmax at that step.
    pub ours: u32,
    /// The ruler's token there: the reference's `gen_ids[step]`, or the
    /// exact top1.
    pub theirs: u32,
    /// The ruler's top1-top2 margin at that step: the reference's
    /// `gen_margins[step]`, or the exact margin.
    pub ref_margin: f32,
}

/// Margin bucket edges of [`ForcedReport::buckets`], as half-open lower
/// bounds: `[0, 0.1)`, `[0.1, 0.5)`, `[0.5, 1.0)`, `[1.0, inf)`. Fixed, not
/// derived from `margin_floor`: the shape of the disagreement distribution is
/// what the reader wants, and it must not move when a threshold does.
pub const FORCED_BUCKETS: [f32; 3] = [0.1, 0.5, 1.0];

/// The whole-set result of [`compare_forced`] and [`compare_forced_exact`].
/// "The ruler" below is the reference under the first and the exact truth
/// under the second; the fields mean the same thing against either.
#[derive(Debug, Clone)]
pub struct ForcedReport {
    /// Positions compared: the sum over rows of the shorter of the two
    /// lengths. The denominator of every count below.
    pub positions: usize,
    /// Every position where our argmax is not the ruler's token, in row
    /// then step order.
    pub disagree: Vec<ForcedPos>,
    /// Disagreements by the ruler's margin, cut at [`FORCED_BUCKETS`]. A
    /// margin exactly on an edge falls in the upper bucket.
    pub buckets: [usize; 4],
    /// Disagreements whose ruler margin is at least `margin_floor` — the
    /// ones the ruler was clear about, and the decision quantity.
    pub disagree_ge_floor: usize,
    /// The largest ruler margin at any disagreement, `None` when there are
    /// none.
    pub max_margin: Option<f32>,
}

/// One prompt of the exact truth file (`just build-exact-forced`,
/// `exact_ref --emit`): at each step of the reference's forced path, the
/// model's exact top1 and top2 and the margin between them. Indexed like
/// `gen_ids`.
#[derive(Debug, Clone)]
pub struct ExactRow {
    pub id: usize,
    pub top1: Vec<u32>,
    pub top2: Vec<u32>,
    pub margins: Vec<f32>,
}

/// The exact truth file: its first comment line (what the rows were
/// computed from, as `key=value` words) and one [`ExactRow`] per prompt in
/// file order.
#[derive(Debug, Clone)]
pub struct ExactFile {
    pub header: String,
    pub rows: Vec<ExactRow>,
}

impl ExactFile {
    /// The value of `key=value` in the header, `None` when the header does
    /// not carry the key.
    #[must_use]
    pub fn header_value(&self, key: &str) -> Option<&str> {
        self.header
            .split_whitespace()
            .find_map(|w| w.strip_prefix(key)?.strip_prefix('='))
    }
}

/// The exact truth file written by `just build-exact-forced`: a header
/// comment, a column comment, then `id step exact_top1 exact_top2
/// exact_margin` lines. A prompt's lines are contiguous and its steps run
/// 0, 1, 2, … without a gap — a missing step would leave a forced position
/// with no answer, and the comparer must never skip one silently.
pub fn read_exact(path: &Path) -> Result<ExactFile, GateError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("read_exact: cannot read {}: {e}", path.display()))?;
    let mut header = None;
    let mut rows: Vec<ExactRow> = Vec::new();
    for line in text.lines() {
        if let Some(c) = line.strip_prefix('#') {
            if header.is_none() {
                header = Some(c.trim().to_string());
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }
        let bad = |why: &str| -> GateError {
            format!("read_exact: {}: {why}: {line:?}", path.display()).into()
        };
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 5 {
            return Err(bad("want id<tab>step<tab>top1<tab>top2<tab>margin"));
        }
        let int = |s: &str| s.trim().parse::<usize>().map_err(|e| bad(&e.to_string()));
        let tok = |s: &str| s.trim().parse::<u32>().map_err(|e| bad(&e.to_string()));
        let (id, step) = (int(f[0])?, int(f[1])?);
        let margin = f[4]
            .trim()
            .parse::<f32>()
            .map_err(|e| bad(&e.to_string()))?;
        if rows.last().is_none_or(|r| r.id != id) {
            if rows.iter().any(|r| r.id == id) {
                return Err(bad("the prompt's rows are not contiguous"));
            }
            rows.push(ExactRow {
                id,
                top1: Vec::new(),
                top2: Vec::new(),
                margins: Vec::new(),
            });
        }
        let r = rows.last_mut().expect("a row was pushed above");
        if step != r.top1.len() {
            return Err(bad(&format!("step {step} where {} was next", r.top1.len())));
        }
        r.top1.push(tok(f[2])?);
        r.top2.push(tok(f[3])?);
        r.margins.push(margin);
    }
    if rows.is_empty() {
        return Err(format!("read_exact: no rows in {}", path.display()).into());
    }
    Ok(ExactFile {
        header: header.unwrap_or_default(),
        rows,
    })
}

/// Whether `exact` answers every forced position of `reference`: the same
/// prompts in the same order, each with at least the reference row's own
/// length of steps. `Err` says which position has no answer.
pub fn exact_covers(exact: &[ExactRow], reference: &[GreedyRow]) -> Result<(), String> {
    if exact.len() != reference.len() {
        return Err(format!(
            "{} truth rows for {} reference rows",
            exact.len(),
            reference.len()
        ));
    }
    for (e, r) in exact.iter().zip(reference) {
        if e.id != r.id {
            return Err(format!("truth row {} against reference row {}", e.id, r.id));
        }
        if e.top1.len() < r.gen_ids.len() {
            return Err(format!(
                "prompt {}: the truth has {} steps, the reference {}",
                r.id,
                e.top1.len(),
                r.gen_ids.len()
            ));
        }
    }
    Ok(())
}

/// Judge our teacher-forced argmaxes against the reference, position by
/// position.
///
/// `ours[i][s]` must be our argmax for `reference[i]` at step `s` taken with
/// `reference[i].gen_ids[..s]` fed as input — the reference's path, not our
/// own. Fed our own path instead, this reads as a free-running comparison
/// with the wrong margins attached and is not what it claims to measure.
/// A row is compared over the shorter of the two lengths, the same
/// convention [`compare_greedy`] holds for a reference row that stopped
/// early on EOS.
///
/// # Panics
///
/// Panics when `ours.len() != reference.len()`: every reference row must be
/// answered, and silently skipping some would under-report disagreements.
#[must_use]
pub fn compare_forced(
    ours: &[Vec<u32>],
    reference: &[GreedyRow],
    margin_floor: f32,
) -> ForcedReport {
    assert_eq!(
        ours.len(),
        reference.len(),
        "compare_forced: {} tables for {} reference rows",
        ours.len(),
        reference.len()
    );
    tally_forced(
        ours.iter().zip(reference).map(|(t, r)| {
            (
                r.id,
                t.as_slice(),
                r.gen_ids.as_slice(),
                r.gen_margins.as_slice(),
            )
        }),
        margin_floor,
    )
}

/// Judge our teacher-forced argmaxes against the exact truth, position by
/// position: [`compare_forced`] with the answer and margin at every position
/// taken from `exact` instead of from the reference.
///
/// `ours[i]` is the same forced table [`compare_forced`] takes — the
/// reference's path fed as input — and `exact[i]` is the truth computed on
/// that same path for the same prompt (check with [`exact_covers`] first).
/// A row is compared over the shorter of the two lengths.
///
/// # Panics
///
/// Panics when `ours.len() != exact.len()`, for the reason
/// [`compare_forced`] does.
#[must_use]
pub fn compare_forced_exact(
    ours: &[Vec<u32>],
    exact: &[ExactRow],
    margin_floor: f32,
) -> ForcedReport {
    assert_eq!(
        ours.len(),
        exact.len(),
        "compare_forced_exact: {} tables for {} truth rows",
        ours.len(),
        exact.len()
    );
    tally_forced(
        ours.iter()
            .zip(exact)
            .map(|(t, e)| (e.id, t.as_slice(), e.top1.as_slice(), e.margins.as_slice())),
        margin_floor,
    )
}

/// The two forced comparers' shared count: per row `(id, ours, the ruler's
/// tokens, the ruler's margins)`, compared over the shorter of `ours` and
/// the ruler's tokens.
fn tally_forced<'a>(
    rows: impl Iterator<Item = (usize, &'a [u32], &'a [u32], &'a [f32])>,
    margin_floor: f32,
) -> ForcedReport {
    let mut positions = 0usize;
    let mut disagree = Vec::new();
    let mut buckets = [0usize; 4];
    let mut disagree_ge_floor = 0usize;
    let mut max_margin: Option<f32> = None;
    for (id, table, truth, margins) in rows {
        let span = table.len().min(truth.len());
        positions += span;
        for step in 0..span {
            if table[step] == truth[step] {
                continue;
            }
            let m = margins[step];
            let b = FORCED_BUCKETS.iter().filter(|&&e| m >= e).count();
            buckets[b] += 1;
            if m >= margin_floor {
                disagree_ge_floor += 1;
            }
            max_margin = Some(max_margin.map_or(m, |x: f32| x.max(m)));
            disagree.push(ForcedPos {
                id,
                step,
                ours: table[step],
                theirs: truth[step],
                ref_margin: m,
            });
        }
    }
    ForcedReport {
        positions,
        disagree,
        buckets,
        disagree_ge_floor,
        max_margin,
    }
}

/// The continuous measure of a teacher-forced table against the exact truth:
/// how far our top1-top2 margin sits from the model's exact one, over every
/// judged position. A count of clear disagreements moves in steps of one on a
/// sample of a handful; this moves with every position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SigmaReport {
    /// Positions judged: the sum over rows of the shortest of our tokens, our
    /// margins and the truth's steps.
    pub positions: usize,
    /// RMS of `s_i - m_i`, where `m_i` is the exact margin and `s_i` our
    /// margin signed toward the exact top1: `+margin` where our top1 is the
    /// exact top1, `-margin` where it is not (the exact top1 is then at best
    /// our runner-up).
    pub sigma: f64,
    /// Mean of `s_i - m_i`.
    pub bias: f64,
    /// `Σ Φ(-m_i / σ)` over the positions with `m_i >= margin_floor` — the
    /// disagreements a Gaussian error of width σ predicts there. Only the
    /// clear positions: near ties the prediction overshoots, because the
    /// quantization error is bounded and not Gaussian-tailed.
    pub expected_ge_floor: f64,
    /// Positions with `m_i >= margin_floor` whose top1 is not the exact top1.
    pub actual_ge_floor: usize,
}

/// [`SigmaReport`] of a forced table against the exact truth. `top1[i][s]`
/// is our argmax at the same position [`compare_forced_exact`] judges, and
/// `margins[i][s]` our top1-top2 logit margin there (non-negative when `top1`
/// is the logits' own argmax). A row is judged over the shortest of the three
/// lengths. Returns `None` when no position is judged.
///
/// # Panics
///
/// Panics when the three tables do not have the same number of rows, for
/// the reason [`compare_forced`] does.
#[must_use]
pub fn sigma_forced(
    top1: &[Vec<u32>],
    margins: &[Vec<f32>],
    exact: &[ExactRow],
    margin_floor: f32,
) -> Option<SigmaReport> {
    assert!(
        top1.len() == exact.len() && margins.len() == exact.len(),
        "sigma_forced: {} tables and {} margin tables for {} truth rows",
        top1.len(),
        margins.len(),
        exact.len()
    );
    // (s_i - m_i, m_i, top1 matches) per judged position.
    let mut d = Vec::new();
    for ((t, g), e) in top1.iter().zip(margins).zip(exact) {
        let span = t.len().min(g.len()).min(e.top1.len());
        for s in 0..span {
            let hit = t[s] == e.top1[s];
            let ours = f64::from(g[s]);
            let signed = if hit { ours } else { -ours };
            let m = f64::from(e.margins[s]);
            d.push((signed - m, m, hit));
        }
    }
    if d.is_empty() {
        return None;
    }
    let n = d.len() as f64;
    let sigma = (d.iter().map(|&(x, _, _)| x * x).sum::<f64>() / n).sqrt();
    let bias = d.iter().map(|&(x, _, _)| x).sum::<f64>() / n;
    let floor = f64::from(margin_floor);
    let clear = d.iter().filter(|&&(_, m, _)| m >= floor);
    let expected_ge_floor = clear
        .clone()
        .map(|&(_, m, _)| std_normal_cdf(-m / sigma))
        .sum();
    let actual_ge_floor = clear.filter(|&&(_, _, hit)| !hit).count();
    Some(SigmaReport {
        positions: d.len(),
        sigma,
        bias,
        expected_ge_floor,
        actual_ge_floor,
    })
}

/// Φ, the standard normal CDF, through Abramowitz & Stegun 7.1.26 for erfc
/// (absolute error under 1.5e-7) — std has no `erf`, and two decimals of an
/// expected count need no more.
#[must_use]
pub fn std_normal_cdf(z: f64) -> f64 {
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let tail = 0.5 * poly * (-x * x).exp();
    if z >= 0.0 { 1.0 - tail } else { tail }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A well-formed reference row: `gen_ids[0]` doubles as `argmax`.
    fn row(id: usize, gen_ids: &[u32], gen_margins: &[f32]) -> GreedyRow {
        GreedyRow {
            id,
            n_tokens: 4,
            argmax: gen_ids[0],
            top5: vec![gen_ids[0], 7, 9, 11, 13],
            gen_ids: gen_ids.to_vec(),
            gen_margins: gen_margins.to_vec(),
        }
    }

    /// Classification looks only at the FIRST difference and its reference
    /// margin: no difference is Identical, a first difference inside the
    /// floor is NearTie, outside it Diverged, and a wilder difference later
    /// never upgrades the class.
    #[test]
    fn compare_greedy_classifies_the_first_difference_only() {
        let floor = 0.5f32;
        let reference = vec![
            row(0, &[10, 11, 12], &[2.0, 1.5, 0.9]),
            // first diff at 1, ref margin 0.2 < floor
            row(1, &[20, 21, 22], &[2.0, 0.2, 0.05]),
            // first diff at 0, ref margin 2.0 >= floor
            row(2, &[30, 31, 32], &[2.0, 0.2, 0.05]),
        ];
        let ours = vec![vec![10, 11, 12], vec![20, 99, 22], vec![77, 31, 32]];
        let r = compare_greedy(&ours, &reference, floor);
        assert_eq!(r.rows[0].class, GreedyClass::Identical);
        assert_eq!(r.rows[0].first_diff, None);
        assert_eq!(r.rows[0].ref_margin, None);
        assert_eq!(r.rows[1].class, GreedyClass::NearTie);
        assert_eq!(r.rows[1].first_diff, Some(1));
        assert_eq!(r.rows[1].ref_margin, Some(0.2));
        assert_eq!(r.rows[2].class, GreedyClass::Diverged);
        assert_eq!(r.rows[2].first_diff, Some(0));
        assert_eq!(r.rows[2].ref_margin, Some(2.0));
        assert_eq!((r.n_identical, r.n_near_tie, r.n_diverged), (1, 1, 1));
    }

    /// A reference row that stopped on EOS is shorter; it is judged over its
    /// own length, and the stop itself is not a difference. Both lengths ride
    /// along, so the consumer can hold its own EOS contract.
    #[test]
    fn compare_greedy_judges_a_short_eos_row_over_its_own_length() {
        let floor = 0.5f32;
        let reference = vec![row(5, &[10, 11, 100_001], &[1.0, 0.8, 3.0])];
        let r = compare_greedy(&[vec![10, 11, 100_001]], &reference, floor);
        assert_eq!(r.rows[0].class, GreedyClass::Identical);
        assert_eq!((r.rows[0].n_ours, r.rows[0].n_ref), (3, 3));

        let r = compare_greedy(&[vec![10, 11, 100_001, 40, 41]], &reference, floor);
        assert_eq!(r.rows[0].class, GreedyClass::Identical);
        assert_eq!((r.rows[0].n_ours, r.rows[0].n_ref), (5, 3));
    }

    /// The teacher-forced ruler counts EVERY disagreeing position, not the
    /// first one per row — the mirror of
    /// `compare_greedy_classifies_the_first_difference_only`, on rows built
    /// so the two comparers must give different answers. Bucket edges are
    /// checked on the edge itself: a margin of exactly 0.1, 0.5 or 1.0 falls
    /// in the upper bucket, and the floor is `>=`, not `>`.
    #[test]
    fn compare_forced_counts_every_position_not_only_the_first() {
        let floor = 0.5f32;
        let reference = vec![
            row(0, &[10, 11, 12], &[2.0, 1.5, 0.9]),
            row(1, &[20, 21, 22], &[2.0, 0.2, 0.05]),
            row(2, &[30, 31, 32], &[1.0, 0.5, 0.1]),
        ];
        let ours = vec![vec![10, 11, 12], vec![20, 99, 98], vec![77, 88, 99]];
        let f = compare_forced(&ours, &reference, floor);
        assert_eq!(f.positions, 9);
        assert_eq!(f.disagree.len(), 5);
        // [0,0.1): 0.05. [0.1,0.5): 0.2 and 0.1. [0.5,1.0): 0.5. [1.0,inf): 1.0.
        assert_eq!(f.buckets, [1, 2, 1, 1]);
        assert_eq!(f.disagree_ge_floor, 2);
        assert_eq!(f.max_margin, Some(1.0));
        // The second row's LATER difference is a position of its own here;
        // the first-difference comparer never sees it.
        assert_eq!(
            f.disagree[1],
            ForcedPos {
                id: 1,
                step: 2,
                ours: 98,
                theirs: 22,
                ref_margin: 0.05,
            }
        );
        let g = compare_greedy(&ours, &reference, floor);
        assert_eq!((g.n_identical, g.n_near_tie, g.n_diverged), (1, 1, 1));
    }

    /// A row is compared over the shorter of the two lengths in either
    /// direction — a reference row that stopped on EOS, and a table of ours
    /// that is short — and a set with no disagreement carries no
    /// `max_margin`.
    #[test]
    fn compare_forced_spans_the_shorter_of_the_two_lengths() {
        let floor = 0.5f32;
        let reference = vec![row(5, &[10, 11, 100_001], &[1.0, 0.8, 3.0])];

        let f = compare_forced(&[vec![10, 99, 100_001, 40, 41]], &reference, floor);
        assert_eq!(f.positions, 3);
        assert_eq!(f.buckets, [0, 0, 1, 0]);
        assert_eq!(f.disagree_ge_floor, 1);
        assert_eq!(f.max_margin, Some(0.8));

        let f = compare_forced(&[vec![10]], &reference, floor);
        assert_eq!(f.positions, 1);
        assert!(f.disagree.is_empty());
        assert_eq!(f.buckets, [0, 0, 0, 0]);
        assert_eq!(f.max_margin, None);
    }

    /// The exact ruler judges against the truth's answer and margin, not the
    /// reference's: a position where we side with a wrong reference counts,
    /// one where we side with the truth against a confident reference does
    /// not, and the buckets and the floor read the truth's margin. Rows
    /// built so the two forced comparers must give different answers.
    #[test]
    fn compare_forced_exact_judges_against_the_truth_not_the_reference() {
        let floor = 0.5f32;
        let reference = vec![
            row(0, &[10, 11, 12], &[2.0, 0.9, 0.3]),
            row(1, &[20, 21, 22], &[1.0, 0.1, 0.2]),
        ];
        let exact = vec![
            // Step 1: the reference's 11 is wrong, the truth is 15 by 1.4.
            // Step 2: the reference is right but the truth's margin is 0.05.
            ExactRow {
                id: 0,
                top1: vec![10, 15, 12],
                top2: vec![7, 11, 13],
                margins: vec![2.1, 1.4, 0.05],
            },
            // Step 0: the reference's 20 is wrong by a near tie, truth 23 by
            // 0.2; step 1 right, margin exactly on the floor.
            ExactRow {
                id: 1,
                top1: vec![23, 21, 22],
                top2: vec![20, 9, 8],
                margins: vec![0.2, 0.5, 1.0],
            },
        ];
        // Row 0: we side with the reference at 1 (wrong), differ from both at 2.
        // Row 1: we side with the truth at 0, differ from both at 1.
        let ours = vec![vec![10, 11, 99], vec![23, 98, 22]];
        let f = compare_forced_exact(&ours, &exact, floor);
        assert_eq!(f.positions, 6);
        assert_eq!(f.disagree.len(), 3);
        // [0,0.1): 0.05. [0.1,0.5): none. [0.5,1.0): 0.5. [1.0,inf): 1.4.
        assert_eq!(f.buckets, [1, 0, 1, 1]);
        assert_eq!(f.disagree_ge_floor, 2);
        assert_eq!(f.max_margin, Some(1.4));
        assert_eq!(
            f.disagree[0],
            ForcedPos {
                id: 0,
                step: 1,
                ours: 11,
                theirs: 15,
                ref_margin: 1.4,
            }
        );
        // The reference ruler sees the same table differently: it counts row
        // 1 step 0 (reference margin 1.0) and never sees row 0 step 1.
        let g = compare_forced(&ours, &reference, floor);
        assert_eq!(g.disagree.len(), 3);
        assert_eq!(g.disagree_ge_floor, 1);
        assert_eq!((g.disagree[1].id, g.disagree[1].step), (1, 0));
    }

    /// σ signs our margin toward the exact top1, so a disagreement adds its
    /// margin to the error instead of cancelling it; the expected and actual
    /// counts read only the positions at or above the floor, the floor
    /// itself included; a row is judged over the shortest of its three
    /// lengths.
    #[test]
    fn sigma_forced_signs_toward_the_truth_and_counts_only_clear_positions() {
        let exact = vec![
            ExactRow {
                id: 0,
                top1: vec![10, 15, 12],
                top2: vec![7, 11, 13],
                margins: vec![1.5, 1.0, 9.0],
            },
            ExactRow {
                id: 1,
                top1: vec![20],
                top2: vec![21],
                margins: vec![0.5],
            },
        ];
        // Row 0: agree by 2.0 (d +0.5), disagree by 0.5 (d -1.5); step 2 has
        // no margin of ours and is not judged. Row 1: agree by 0.3 (d -0.2),
        // exact margin exactly on the floor.
        let top1 = vec![vec![10, 11, 12], vec![20]];
        let margins = vec![vec![2.0, 0.5], vec![0.3]];
        let r = sigma_forced(&top1, &margins, &exact, 0.5).unwrap();
        assert_eq!(r.positions, 3);
        let sigma = (2.54f64 / 3.0).sqrt();
        assert!((r.sigma - sigma).abs() < 1e-6, "{}", r.sigma);
        assert!((r.bias - (-0.4)).abs() < 1e-6, "{}", r.bias);
        let want = std_normal_cdf(-1.5 / sigma)
            + std_normal_cdf(-1.0 / sigma)
            + std_normal_cdf(-0.5 / sigma);
        assert!((r.expected_ge_floor - want).abs() < 1e-9);
        assert_eq!(r.actual_ge_floor, 1);

        assert_eq!(sigma_forced(&[vec![]], &[vec![]], &exact[..1], 0.5), None);
    }

    /// Φ against tabulated values, both tails.
    #[test]
    fn std_normal_cdf_matches_the_table() {
        for (z, want) in [
            (0.0, 0.5),
            (-1.0, 0.158_655_25),
            (-2.0, 0.022_750_13),
            (-3.0, 0.001_349_90),
            (1.0, 0.841_344_75),
        ] {
            assert!((std_normal_cdf(z) - want).abs() < 2e-7, "Φ({z})");
        }
    }

    /// `read_exact` groups contiguous steps per prompt and reads the header;
    /// a gap in the steps, a prompt split in two, or a short line is an
    /// error, and `exact_covers` names a prompt the truth does not answer
    /// all the way.
    #[test]
    fn read_exact_takes_contiguous_steps_and_checks_coverage() -> std::io::Result<()> {
        let uniq = std::process::id();
        let dir = std::env::temp_dir();
        let head = "# exact_ref model=/m.gguf kv=f64 forced_sha256=abc\n\
                    #id\tstep\texact_top1\texact_top2\texact_margin\n";
        let p = dir.join(format!("bloomery-read-exact-{uniq}.tsv"));
        std::fs::write(
            &p,
            format!(
                "{head}0\t0\t10\t7\t2.000000\n0\t1\t11\t9\t0.250000\n\
                 1\t0\t20\t3\t1.500000\n"
            ),
        )?;
        let f = read_exact(&p).unwrap();
        assert_eq!(f.header_value("kv"), Some("f64"));
        assert_eq!(f.header_value("forced_sha256"), Some("abc"));
        assert_eq!(f.header_value("sha256"), None);
        assert_eq!(f.rows.len(), 2);
        assert_eq!(f.rows[0].top1, vec![10, 11]);
        assert_eq!(f.rows[0].top2, vec![7, 9]);
        assert_eq!(f.rows[0].margins, vec![2.0, 0.25]);
        assert_eq!((f.rows[1].id, f.rows[1].top1.len()), (1, 1));

        let reference = vec![row(0, &[10, 11], &[1.0, 1.0]), row(1, &[20], &[1.0])];
        assert!(exact_covers(&f.rows, &reference).is_ok());
        let longer = vec![
            row(0, &[10, 11], &[1.0, 1.0]),
            row(1, &[20, 21], &[1.0, 1.0]),
        ];
        assert!(exact_covers(&f.rows, &longer).is_err());
        let other = vec![row(0, &[10, 11], &[1.0, 1.0]), row(2, &[20], &[1.0])];
        assert!(exact_covers(&f.rows, &other).is_err());

        for (name, body) in [
            ("gap", "0\t0\t10\t7\t2.0\n0\t2\t11\t9\t0.25\n"),
            (
                "split",
                "0\t0\t10\t7\t2.0\n1\t0\t20\t3\t1.5\n0\t1\t11\t9\t0.25\n",
            ),
            ("short", "0\t0\t10\t7\n"),
        ] {
            let pb = dir.join(format!("bloomery-read-exact-{name}-{uniq}.tsv"));
            std::fs::write(&pb, format!("{head}{body}"))?;
            assert!(read_exact(&pb).is_err(), "{name} must not parse");
        }
        Ok(())
    }

    /// `read_greedy` parses only the 7-column `--gen` format, and enforces
    /// the writer's `gen_ids[0] == argmax` duplication.
    #[test]
    fn read_greedy_takes_only_the_gen_format() -> std::io::Result<()> {
        let uniq = std::process::id();
        let p = std::env::temp_dir().join(format!("bloomery-read-greedy-{uniq}.tsv"));
        std::fs::write(
            &p,
            "# argmax_ref\tik_llama.cpp\tmodel=x\tgen=3\n\
             #id\tn_tokens\targmax\ttop5_ids\ttop5_logits\tgen_ids\tgen_margins\n\
             0\t4\t10\t10,7,9,11,13\t2.0,1.0,0.5,0.2,0.1\t10,11,12\t2.0,1.5,0.9\n",
        )?;
        let rows = read_greedy(&p).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, 0);
        assert_eq!(rows[0].n_tokens, 4);
        assert_eq!(rows[0].argmax, 10);
        assert_eq!(rows[0].top5, vec![10, 7, 9, 11, 13]);
        assert_eq!(rows[0].gen_ids, vec![10, 11, 12]);
        assert_eq!(rows[0].gen_margins, vec![2.0, 1.5, 0.9]);

        let p5 = std::env::temp_dir().join(format!("bloomery-read-argmax-{uniq}.tsv"));
        std::fs::write(&p5, "0\t4\t10\t10,7,9,11,13\t2.0,1.0,0.5,0.2,0.1\n")?;
        assert!(read_greedy(&p5).is_err());

        let pc = std::env::temp_dir().join(format!("bloomery-read-corrupt-{uniq}.tsv"));
        std::fs::write(
            &pc,
            "0\t4\t10\t10,7,9,11,13\t2.0,1.0,0.5,0.2,0.1\t11,11,12\t2.0,1.5,0.9\n",
        )?;
        assert!(read_greedy(&pc).is_err());
        Ok(())
    }
}
