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
//! Only the FIRST difference is judged. After it the two continuations are
//! conditioned on different text, so every later token answers a different
//! question and carries no evidence. `compare_greedy` therefore reports the
//! first differing index, the reference's own top1-top2 margin there, and one
//! of three classes; what happens past that index never enters the verdict.

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
