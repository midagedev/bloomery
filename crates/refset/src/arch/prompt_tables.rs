//! A prompt set's own files beside the node dumps (`tools/ref/cb_dump.cpp`, the mainline-API dumper of every family whose
//! oracle is a llama.cpp tree): the prompt's ids from the manifest's `# tokens` line, `topk.tsv` (the last position's 40 largest
//! logits with their f64 log-softmax) and `greedy.tsv` (the argmax tokens continued from the prompt, each with its margin to the
//! runner-up). The node files and the last logits row are the manifest's ([`crate::ik`]); these are the two tables the dumper
//! writes next to them.
//!
//! Each reader checks the table against what the manifest states of it (`# topk\ttopk.tsv\t<k>`, `# greedy\tgreedy.tsv\t<n>`)
//! and against the other table: the greedy continuation's first token is the top-k table's first id.

use crate::RefError;
use crate::ik::RefManifest;
use std::path::Path;

/// One row of `topk.tsv`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Top {
    pub id: u32,
    pub logit: f32,
    /// The f64 log-softmax of the whole logits row at `id`, as the dumper computed it.
    pub logprob: f64,
}

/// The greedy continuation of `greedy.tsv`: step `s` is the argmax after the prompt and the `s` tokens before it.
#[derive(Debug, Clone, PartialEq)]
pub struct Greedy {
    pub ids: Vec<u32>,
    /// The gap between the step's largest logit and the runner-up's, in logit units.
    pub margins: Vec<f32>,
    pub logprobs: Vec<f64>,
}

/// The `# <key>\t<file>\t<count>` line of the manifest: the table's file name and the count it states.
fn table_line(man: &RefManifest, key: &str) -> Result<(String, usize), RefError> {
    let head = format!("# {key}\t");
    let line = man
        .header
        .other
        .iter()
        .find_map(|l| l.strip_prefix(head.as_str()))
        .ok_or_else(|| {
            RefError::missing(
                &man.dir,
                format!("{}: no `# {key}` line", man.dir.display()),
            )
        })?;
    let bad =
        |what: &str| RefError::malformed(man.dir.display(), format!("`# {key}` {line:?}: {what}"));
    let (file, count) = line
        .split_once('\t')
        .ok_or_else(|| bad("want <file>\\t<count>"))?;
    let count = count
        .parse()
        .map_err(|_| bad("the count is not a number"))?;
    Ok((file.to_string(), count))
}

fn read_table(dir: &Path, file: &str) -> Result<String, RefError> {
    let path = dir.join(file);
    std::fs::read_to_string(&path)
        .map_err(|e| RefError::missing(&path, format!("{}: {e}", path.display())))
}

fn field<T: std::str::FromStr>(
    at: &Path,
    n: usize,
    f: &[&str],
    i: usize,
    what: &str,
) -> Result<T, RefError> {
    f.get(i).and_then(|v| v.parse().ok()).ok_or_else(|| {
        RefError::malformed(
            format!("{}:{n}", at.display()),
            format!("{what} is not a number"),
        )
    })
}

/// The prompt's ids: the manifest's `# tokens` line.
pub fn tokens(man: &RefManifest) -> Result<&[u32], RefError> {
    man.header.tokens.as_deref().ok_or_else(|| {
        RefError::missing(
            &man.dir,
            format!("{}: no `# tokens` line", man.dir.display()),
        )
    })
}

/// The last position's top logits, rank order: the table's rank column counts 0, 1, 2..., the logits do not rise, the log-softmax
/// is at most 0 and does not rise, and no id repeats.
pub fn topk(man: &RefManifest) -> Result<Vec<Top>, RefError> {
    let (file, k) = table_line(man, "topk")?;
    let path = man.dir.join(&file);
    let mut rows: Vec<Top> = Vec::new();
    for (n, line) in read_table(&man.dir, &file)?.lines().enumerate() {
        let f: Vec<&str> = line.split('\t').collect();
        let at = n + 1;
        if f.len() != 4 {
            return Err(RefError::malformed(
                format!("{}:{at}", path.display()),
                format!("{} fields, want rank, id, logit, logprob", f.len()),
            ));
        }
        let rank: usize = field(&path, at, &f, 0, "rank")?;
        if rank != rows.len() {
            return Err(RefError::malformed(
                format!("{}:{at}", path.display()),
                format!("rank {rank}, want {}", rows.len()),
            ));
        }
        let top = Top {
            id: field(&path, at, &f, 1, "id")?,
            logit: field(&path, at, &f, 2, "logit")?,
            logprob: field(&path, at, &f, 3, "logprob")?,
        };
        let rises = rows
            .last()
            .is_some_and(|p| top.logit > p.logit || top.logprob > p.logprob);
        if top.logprob.is_nan() || top.logprob > 0.0 || rises || rows.iter().any(|p| p.id == top.id)
        {
            return Err(RefError::malformed(
                format!("{}:{at}", path.display()),
                "the table is not in descending order of distinct ids with logprobs at most 0"
                    .to_string(),
            ));
        }
        rows.push(top);
    }
    if rows.len() != k {
        return Err(RefError::malformed(
            path.display(),
            format!("{} rows, the manifest states {k}", rows.len()),
        ));
    }
    Ok(rows)
}

/// The greedy continuation: `step` counts 0, 1, 2..., the margins are not negative and the count is the manifest's. The first
/// token is the top-k table's first id, the prompt's own argmax.
pub fn greedy(man: &RefManifest) -> Result<Greedy, RefError> {
    let (file, n) = table_line(man, "greedy")?;
    let path = man.dir.join(&file);
    let mut g = Greedy {
        ids: Vec::new(),
        margins: Vec::new(),
        logprobs: Vec::new(),
    };
    for (at, line) in read_table(&man.dir, &file)?.lines().enumerate() {
        if line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 4 {
            return Err(RefError::malformed(
                format!("{}:{}", path.display(), at + 1),
                format!("{} fields, want step, id, margin, logprob", f.len()),
            ));
        }
        let step: usize = field(&path, at + 1, &f, 0, "step")?;
        let margin: f32 = field(&path, at + 1, &f, 2, "margin")?;
        if step != g.ids.len() || margin.is_nan() || margin < 0.0 {
            return Err(RefError::malformed(
                format!("{}:{}", path.display(), at + 1),
                format!(
                    "step {step} margin {margin}, want step {} and a margin of at least 0",
                    g.ids.len()
                ),
            ));
        }
        g.ids.push(field(&path, at + 1, &f, 1, "id")?);
        g.margins.push(margin);
        g.logprobs.push(field(&path, at + 1, &f, 3, "logprob")?);
    }
    if g.ids.len() != n {
        return Err(RefError::malformed(
            path.display(),
            format!("{} steps, the manifest states {n}", g.ids.len()),
        ));
    }
    if let Some(&first) = g.ids.first()
        && topk(man)?.first().map(|t| t.id) != Some(first)
    {
        return Err(RefError::malformed(
            path.display(),
            format!("the first token {first} is not the top-k table's first id"),
        ));
    }
    Ok(g)
}
