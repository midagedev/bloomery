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
//! There are two comparers, and they answer different questions.
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
/// was fed, against the reference's token there and its margin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForcedPos {
    /// The prompt's id, from the reference row.
    pub id: usize,
    /// The step inside the continuation, indexed like `gen_ids`.
    pub step: usize,
    /// Our argmax at that step.
    pub ours: u32,
    /// The reference's token there.
    pub theirs: u32,
    /// The reference's own top1-top2 margin at that step, `gen_margins[step]`.
    pub ref_margin: f32,
}

/// Margin bucket edges of [`ForcedReport::buckets`], as half-open lower
/// bounds: `[0, 0.1)`, `[0.1, 0.5)`, `[0.5, 1.0)`, `[1.0, inf)`. Fixed, not
/// derived from `margin_floor`: the shape of the disagreement distribution is
/// what the reader wants, and it must not move when a threshold does.
pub const FORCED_BUCKETS: [f32; 3] = [0.1, 0.5, 1.0];

/// `compare_forced`'s whole-set result.
#[derive(Debug, Clone)]
pub struct ForcedReport {
    /// Positions compared: the sum over rows of the shorter of the two
    /// lengths. The denominator of every count below.
    pub positions: usize,
    /// Every position where our argmax is not the reference's token, in row
    /// then step order.
    pub disagree: Vec<ForcedPos>,
    /// Disagreements by reference margin, cut at [`FORCED_BUCKETS`]. A margin
    /// exactly on an edge falls in the upper bucket.
    pub buckets: [usize; 4],
    /// Disagreements whose reference margin is at least `margin_floor` — the
    /// ones the reference itself was clear about, and the decision quantity.
    pub disagree_ge_floor: usize,
    /// The largest reference margin at any disagreement, `None` when there
    /// are none.
    pub max_margin: Option<f32>,
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
    let mut positions = 0usize;
    let mut disagree = Vec::new();
    let mut buckets = [0usize; 4];
    let mut disagree_ge_floor = 0usize;
    let mut max_margin: Option<f32> = None;
    for (table, r) in ours.iter().zip(reference) {
        let span = table.len().min(r.gen_ids.len());
        positions += span;
        for step in 0..span {
            if table[step] == r.gen_ids[step] {
                continue;
            }
            let m = r.gen_margins[step];
            let b = FORCED_BUCKETS.iter().filter(|&&e| m >= e).count();
            buckets[b] += 1;
            if m >= margin_floor {
                disagree_ge_floor += 1;
            }
            max_margin = Some(max_margin.map_or(m, |x: f32| x.max(m)));
            disagree.push(ForcedPos {
                id: r.id,
                step,
                ours: table[step],
                theirs: r.gen_ids[step],
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
