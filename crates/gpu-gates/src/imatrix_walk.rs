//! The collecting walk behind a generate CLI's `--imatrix` (`generate_kimi_k3`, `generate_exaone`):
//! what `llama-imatrix` runs, as one owner — the text read and tokenized as `llama-imatrix` does
//! (special tokens added, none parsed), cut into chunks of `--ctx` ids (the remainder dropped; at
//! least two chunks, as llama requires; `--chunks N` takes the first N), each chunk's first id the
//! BOS where the vocabulary adds one, and every chunk run from an empty model through the engine's
//! own prompt walk, the collector attached ([`bloomery_gpu::imatrix`]). The family's body is the
//! caller's: one `run` closure takes the model from empty through a chunk.
//!
//! A long run carries a resume story: `--save-every K` writes the collection so far to `OUT`
//! every K chunks (a partial file, llama's own form, moved into place whole), and `--resume`
//! seeds the collector from `OUT` before the first chunk continues it at the chunk the file's
//! `imatrix.chunk_count` names — a run that died mid-way restarts where it left off. One
//! `imatrix chunk=` line a chunk (its wall, the chunk's whole run) and a closing `imatrix
//! chunks=` line, plain text on stdout.

use std::path::Path;

use bloomery_gpu::{Gpu, imatrix};
use gguf::Value;
use tokenizer::Tokenizer;

use crate::GateError;

/// What every error of the walk names: the shared walk's own name, so the calling CLI's lines
/// stay distinguishable in a log both write.
const WHAT: &str = "imatrix_walk";

/// One entry of a partial collection ([`partial_of`]): its weight name, `ne0`, `n_mat`, the sums
/// and the counts, as [`imatrix::Collector::seed`] takes it.
type Partial = (String, usize, usize, Vec<f32>, Vec<f32>);

/// The text of `path` as ids: `llama-imatrix`'s reading of a plain-text corpus.
fn ids_of(tok: &Tokenizer, path: &str) -> Result<Vec<u32>, GateError> {
    let raw = std::fs::read(path).map_err(|e| format!("{WHAT}: --text {path}: read: {e}"))?;
    Ok(tok.encode(&raw, true, false))
}

/// The chunks of `ids`: `ctx` ids each, the first `chunks` of them (all when `None`), the first
/// id of each the BOS where the vocabulary adds one. Refused by name: a text shorter than two
/// chunks, `--chunks` below 2, and a vocabulary that adds a BOS and names none.
fn chunks_of(
    tok: &Tokenizer,
    (path, ids): (&str, &[u32]),
    (ctx, chunks): (usize, Option<usize>),
) -> Result<Vec<Vec<u32>>, GateError> {
    if ids.len() < 2 * ctx {
        return Err(format!(
            "{WHAT}: --text {path} tokenizes to {} ids; at least {} (two chunks of --ctx {ctx}) \
             are needed",
            ids.len(),
            2 * ctx
        )
        .into());
    }
    let n = chunks.map_or(ids.len() / ctx, |c| c.min(ids.len() / ctx));
    if n < 2 {
        return Err(format!("{WHAT}: --chunks {chunks:?}: at least 2 chunks").into());
    }
    let bos = match (tok.add_bos(), tok.specials().bos) {
        (true, Some(b)) => Some(b),
        (true, None) => {
            return Err(format!("{WHAT}: the vocabulary adds a BOS and names none").into());
        }
        (false, _) => None,
    };
    Ok((0..n)
        .map(|c| {
            let mut chunk = ids[c * ctx..(c + 1) * ctx].to_vec();
            if let Some(b) = bos {
                chunk[0] = b;
            }
            chunk
        })
        .collect())
}

/// The partial collection at `path` as [`imatrix::Collector::seed`] takes it, and the chunk it
/// ends at (the file's `imatrix.chunk_count`): every `<name>.in_sum2` with its `.counts`, the
/// entry's shape the tensors' own. Refused by name: a file that is no GGUF, a name whose sums
/// and counts do not split evenly, and a missing `imatrix.chunk_count`.
fn partial_of(path: &Path) -> Result<(usize, Vec<Partial>), GateError> {
    let g =
        gguf::Gguf::open(path).map_err(|e| format!("{WHAT}: --resume {}: {e}", path.display()))?;
    let count = match g.value("imatrix.chunk_count") {
        Some(Value::U32(n)) => *n as usize,
        _ => {
            return Err(format!(
                "{WHAT}: --resume {}: no u32 imatrix.chunk_count",
                path.display()
            )
            .into());
        }
    };
    let floats = |name: &str| -> Result<Vec<f32>, GateError> {
        let t = g
            .find(name)
            .ok_or_else(|| format!("{WHAT}: --resume {}: no {name}", path.display()))?;
        let bytes = g
            .data(t)
            .map_err(|e| format!("{WHAT}: --resume {}: {name}: {e}", path.display()))?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    };
    let mut rows = Vec::new();
    for t in g.iter_tensors() {
        let Some(name) = t.name.strip_suffix(".in_sum2") else {
            continue;
        };
        let (sums, counts) = (
            floats(&format!("{name}.in_sum2")),
            floats(&format!("{name}.counts")),
        );
        let (Ok(sums), Ok(counts)) = (sums, counts) else {
            return Err(format!(
                "{WHAT}: --resume {}: {name} lacks its in_sum2 or counts",
                path.display()
            )
            .into());
        };
        if counts.is_empty() || sums.len() % counts.len() != 0 {
            return Err(format!(
                "{WHAT}: --resume {}: {name}: {} sums over {} counts do not split evenly",
                path.display(),
                sums.len(),
                counts.len()
            )
            .into());
        }
        rows.push((
            name.to_owned(),
            sums.len() / counts.len(),
            counts.len(),
            sums,
            counts,
        ));
    }
    Ok((count, rows))
}

/// Seed the collector from the partial collection at `out` (`--resume`, module doc) and return
/// the chunk it ends at, the count the file's own `imatrix.chunk_count` names. The caller holds
/// the model open on `gpu` and runs [`collect`] from the chunk returned. Refused by name:
/// everything [`partial_of`] refuses, and a seed beside a collection already under way.
pub fn resume(gpu: &Gpu, out: &str) -> Result<usize, GateError> {
    let (at, rows) = partial_of(Path::new(out))?;
    let collector = imatrix::attach();
    for (name, ne0, n_mat, sums, counts) in rows {
        collector
            .seed(gpu, &name, (ne0, n_mat), &sums, &counts)
            .map_err(|e| format!("{WHAT}: --resume {out}: {e}"))?;
    }
    Ok(at)
}

/// The collection so far written to `out` whole: the GGUF lands at `out.part` and is renamed
/// over `out`, so a run that dies mid-write leaves the last whole file for `--resume`.
fn write_whole(
    collector: &imatrix::Collector,
    out: &str,
    (chunk_size, chunk_count): (u32, u32),
    text: &str,
) -> Result<(), GateError> {
    let part = format!("{out}.part");
    collector
        .write(Path::new(&part), (chunk_size, chunk_count), text)
        .map_err(|e| format!("{WHAT}: {e}"))?;
    std::fs::rename(&part, out).map_err(|e| format!("{WHAT}: the write to {out}: {e}"))?;
    Ok(())
}

/// The collection of `text` over the model `m`, every chunk through `run` (which takes the model
/// from empty through the chunk), written to `out` (module doc): `save_every` chunks a partial
/// write to `out`, and `done` names the chunk the walk continues at (0, or [`resume`]'s return).
pub fn collect<M>(
    m: &mut M,
    tok: &Tokenizer,
    (text, out): (&str, &str),
    (ctx, chunks): (usize, Option<usize>),
    save_every: Option<usize>,
    done: usize,
    mut run: impl FnMut(&mut M, &[u32]) -> Result<(), GateError>,
) -> Result<(), GateError> {
    let ids = ids_of(tok, text)?;
    let chunks = chunks_of(tok, (text, &ids), (ctx, chunks))?;
    if done > chunks.len() {
        return Err(format!(
            "{WHAT}: continuing at chunk {done} of a text that holds {}",
            chunks.len()
        )
        .into());
    }
    let collector = imatrix::attach();
    for (c, chunk) in chunks.iter().enumerate().skip(done) {
        let t = std::time::Instant::now();
        run(m, chunk)?;
        if save_every.is_some_and(|k| (c + 1) % k == 0 && c + 1 < chunks.len()) {
            write_whole(collector, out, (ctx as u32, (c + 1) as u32), text)?;
        }
        println!(
            "imatrix chunk={}/{} tokens={ctx} wall_s={:.3}",
            c + 1,
            chunks.len(),
            t.elapsed().as_secs_f64()
        );
    }
    write_whole(collector, out, (ctx as u32, chunks.len() as u32), text)?;
    println!(
        "imatrix chunks={} chunk_size={ctx} entries={} wrote={out}",
        chunks.len(),
        collector.names().len()
    );
    Ok(())
}
