//! A rank of a tensor-parallel deepseek41 load of a checkpoint folder: the file bytes it reads
//! ([`rank_reads`]) and what it needs from its pool ([`preflight`]), both from the plan the load
//! reads by. A rank's part of each tensor is [`rank_span`]'s; an Engram table's rows are the
//! buckets of the rank's hash columns ([`engram_cols`], [`engram_rows`]), at their original row
//! ids, and every other row table is read whole.

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::PathBuf;

use engram::hash::Hash;
use model::arch::deepseek41::hparams::Hparams;
use model::arch::deepseek41::tp::{
    RunShape, TpUnits, engram_cols, engram_rows, rank_need, rank_span, tp_tensors,
};
use model::part::{Planned, Source, Take};
use model::placement::PlacementError;
use model::placement::tp::{self, RankLedger, RankPool};

use crate::folder::{Error, Opened, refuse};
use crate::part::merge_runs;

fn model(e: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::Model(Box::new(e))
}

/// The header's hyperparameters, cut units and Engram hash.
fn facts(opened: &Opened) -> Result<(Hparams, TpUnits, Option<Hash>), Error> {
    let hp = Hparams::read(&opened.header).map_err(model)?;
    let u = TpUnits::of(&hp);
    let hash = match hp.engram {
        Some(_) => Some(Hash::from_keys(&opened.header).map_err(model)?),
        None => None,
    };
    Ok((hp, u, hash))
}

/// The parts rank `rank` of `ranks` reads, planned as the load plans them.
fn planned(opened: &Opened, u: &TpUnits, rank: usize, ranks: usize) -> Result<Vec<Planned>, Error> {
    let rule = |name: &str, ty, dims: &[u64]| -> Result<Take, PlacementError> {
        Ok(match rank_span(name, ty, dims, u, rank, ranks)? {
            Some((dim, at)) => Take::Part(dim, at),
            None => Take::Whole,
        })
    };
    Ok(opened.source.plan(rule)?)
}

/// The widest gap between two runs of one file that [`rank_reads`] reads through as one run: a
/// part cut along its input (an expert's down) reads one short run per row, and listing each
/// would cost more than reading the bytes between them.
const READ_GAP: u64 = 1 << 20;

/// The file byte runs rank `rank` of `ranks` reads of the folder `opened`, each file's runs in
/// order and merged where they overlap or lie at most `READ_GAP` (1 MiB) apart: every file of the folder's top level that is not a shard
/// (the config, the index and the tokenizer files) whole, a link by its target's bytes, every shard's header, each tensor part
/// the load reads
/// ([`part::Checkpoint::reads`](crate::part::Checkpoint::reads)), the rows of each Engram table
/// its hash columns gather and every other row table whole. A folder holding exactly these
/// bytes at their offsets loads rank `rank` as the whole folder does.
///
/// # Errors
///
/// A header the hyperparameters or the hash refuse, a part the plan or its read refuses, and a
/// rank or column split [`engram_cols`] refuses, and a folder whose files cannot be listed, each by
/// name.
pub fn rank_reads(
    opened: &Opened,
    rank: usize,
    ranks: usize,
) -> Result<Vec<(PathBuf, Range<u64>)>, Error> {
    let (hp, u, hash) = facts(opened)?;
    let folder = opened.source.folder();
    let mut runs = folder.headers();
    let listed = std::fs::read_dir(folder.dir())
        .map_err(|e| Error::Refused(format!("{}: {e}", folder.dir().display())))?;
    for entry in listed {
        let entry =
            entry.map_err(|e| Error::Refused(format!("{}: {e}", folder.dir().display())))?;
        let path = entry.path();
        // The load reads a linked file through its link, so its target's kind and length count;
        // a dangling link is no file the load reads.
        let meta = match path.metadata() {
            Ok(meta) => meta,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(Error::Refused(format!("{}: {e}", path.display()))),
        };
        if meta.is_file() && !runs.iter().any(|(p, _)| *p == path) {
            runs.push((path, 0..meta.len()));
        }
    }
    for p in planned(opened, &u, rank, ranks)? {
        runs.extend(opened.source.reads(&p, READ_GAP)?);
    }
    let sites = hp.engram.as_ref().map_or(&[][..], |e| &e.layer_ids[..]);
    for (name, rows) in &opened.rows {
        let site = sites
            .iter()
            .position(|l| *name == format!("blk.{l}.engram_embd.weight"));
        let at = match (site, &hash) {
            (Some(s), Some(h)) => {
                let cols = engram_cols(&u, rank, ranks).map_err(model)?;
                let (primes, offsets) = (h.primes(s).map_err(model)?, h.offsets(s).map_err(model)?);
                engram_rows(primes, offsets, cols)
                    .ok_or_else(|| refuse(name, "the rank's hash columns pass its buckets"))?
            }
            _ => 0..u64::MAX,
        };
        runs.extend(rows.rows_at(at));
    }
    let mut by_file: BTreeMap<PathBuf, Vec<Range<u64>>> = BTreeMap::new();
    for (path, r) in runs {
        by_file.entry(path).or_default().push(r);
    }
    Ok(by_file
        .into_iter()
        .flat_map(|(path, runs)| {
            merge_runs(runs, READ_GAP)
                .into_iter()
                .map(move |r| (path.clone(), r))
        })
        .collect())
}

/// Each rank's ledger of the folder `opened` over `pools` (one a rank) at `run`, from headers
/// alone: its weights by the plan's cuts ([`tp_tensors`]), what it holds beside them
/// ([`rank_need`]), and as its load peak the most one part's read holds at once
/// ([`Source::read_bytes`]).
///
/// # Errors
///
/// A refusal of the header, the cuts or the plan, and a rank whose pool its need passes, with
/// every item it needs ([`tp::preflight`]).
pub fn preflight(
    opened: &Opened,
    pools: &[RankPool],
    run: RunShape,
) -> Result<Vec<RankLedger>, Error> {
    let (hp, u, _) = facts(opened)?;
    let tensors = tp_tensors(&opened.header, &u).map_err(model)?;
    let ranks = pools.len();
    let mut needs = Vec::with_capacity(ranks);
    for rank in 0..ranks {
        let mut need = rank_need(&opened.header, &hp, run, rank, ranks).map_err(model)?;
        for p in planned(opened, &u, rank, ranks)? {
            need.load_peak = need.load_peak.max(opened.source.read_bytes(&p)?);
        }
        needs.push(need);
    }
    tp::preflight(pools, &tensors, &|rank| needs[rank]).map_err(model)
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::path::Path;

    use model::arch::deepseek41::tp::RunShape;
    use model::part::Source;
    use model::placement::tp::RankPool;
    use serde_json::{Value as Json, json};

    use super::super::fixture::{self, Shape};
    use super::{READ_GAP, preflight, rank_reads};
    use crate::folder;
    use crate::test_support::{Entries, json, temp_dir};

    /// A folder of the 1017's tensor headers ([`header_1017`](super::super::tests::header_1017): its first 16 experts, the vocabulary up to the cut tokenizer's last id) over
    /// sparse shard bytes, which is all [`fixture::write`] reads, its entries first passed to
    /// `edit`.
    fn headers_only(dir: &Path, edit: impl FnOnce(&mut Entries)) {
        let files = json("deepseek-v4.1-flash-1017-tokenizer-files.json.gz");
        let mut tok = files["tokenizer.json"].clone();
        let last = tok["model"]["vocab"]
            .as_object()
            .unwrap()
            .values()
            .filter_map(Json::as_u64)
            .max()
            .unwrap();
        // Four ranks split the head by vocabulary row.
        let vocab = (last + 1).next_multiple_of(4);
        tok["added_tokens"]
            .as_array_mut()
            .unwrap()
            .retain(|a| a["id"].as_u64().unwrap() < vocab);
        let tokens = super::super::tokens_by_id(&tok, vocab as usize).unwrap();
        let tokens: Vec<&str> = tokens.iter().map(String::as_str).collect();
        let mut config = files["config.json"].clone();
        config["text_config"]["vocab_size"] = json!(vocab);
        config["text_config"]["engram_compressed_vocab_size"] =
            json!(hf::engram::token_map(&tokens).1);
        let mut entries = super::super::tests::header_1017(|_| 16);
        for e in &mut entries {
            if e.0 == "embed.weight" || e.0 == "head.weight" {
                e.2[0] = vocab as usize;
            }
            // The fixture sizes its Engram tables from its own hash vocabulary, so one source
            // row keeps the sparse shard small.
            if e.0.contains(".engram.embed.") {
                e.2[0] = 1;
            }
        }
        edit(&mut entries);
        let size = |e: &(String, safetensors::Dtype, Vec<usize>)| {
            e.2.iter().product::<usize>() * e.1.size()
        };
        let table: Vec<(&str, &str, &[usize], usize)> = entries
            .iter()
            .map(|e| (e.0.as_str(), e.1.name(), e.2.as_slice(), size(e)))
            .collect();
        let head = safetensors::header(&table);
        let data: usize = table.iter().map(|t| t.3).sum();
        let shard = "model-00001-of-00001.safetensors";
        let file = std::fs::File::create(dir.join(shard)).unwrap();
        std::io::Write::write_all(&mut &file, &head).unwrap();
        file.set_len((head.len() + data) as u64).unwrap();
        let map: serde_json::Map<String, Json> = entries
            .iter()
            .map(|e| (e.0.clone(), json!(shard)))
            .collect();
        let put = |name: &str, v: &Json| std::fs::write(dir.join(name), v.to_string()).unwrap();
        put(safetensors::Folder::INDEX, &json!({ "weight_map": map }));
        put("config.json", &config);
        put("tokenizer.json", &tok);
        put("tokenizer_config.json", &files["tokenizer_config.json"]);
    }

    /// A fixture write that fails once it has begun writing (here at a tensor of a format the
    /// fixture does not draw) leaves neither the fixture folder nor its temporary one.
    #[test]
    fn a_failed_fixture_write_leaves_no_folder() {
        let root = temp_dir("ds41-fixture-fails");
        let source = root.join("source");
        std::fs::create_dir(&source).unwrap();
        headers_only(&source, |entries| {
            for e in entries.iter_mut().filter(|e| e.0 == "norm.weight") {
                e.1 = safetensors::Dtype::F16;
            }
        });
        let shape = Shape {
            layers: vec![1, 20],
            ff: fixture::FF,
            experts: Some(16),
        };
        let err = fixture::write(&source, &root.join("fixture"), &shape, 7).unwrap_err();
        assert!(err.to_string().contains("F16"), "{err}");
        let left: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(left, ["source"]);
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// The original-format fixture of the 1017's headers opens as deepseek41 and its preflight
    /// names every item of each rank at one, two and four ranks, its row cache every page a row
    /// of the rank's columns can touch in each stored plane. A folder holding only the bytes
    /// of a rank's read-set, at their offsets, gives that rank every part the whole folder gives,
    /// the config, the index and the tokenizer files read through links to their bytes and a
    /// dangling link left out.
    /// It writes and reads about a GiB, so it runs on request:
    /// `cargo test --release -p bloomery-model-hf -- --ignored hw_rank_read_set`.
    #[test]
    #[ignore = "writes a fixture folder of about a GiB"]
    fn hw_rank_read_set_loads_its_parts_as_the_whole_folder_does() {
        let root = temp_dir("ds41-reads");
        let source = root.join("source");
        std::fs::create_dir(&source).unwrap();
        headers_only(&source, |_| {});
        let out = root.join("fixture");
        let shape = Shape {
            layers: vec![1, 20],
            ff: fixture::FF,
            experts: Some(16),
        };
        fixture::write(&source, &out, &shape, 7).unwrap();
        let linked = root.join("linked");
        std::fs::create_dir(&linked).unwrap();
        for name in [
            "config.json",
            safetensors::Folder::INDEX,
            "tokenizer.json",
            "tokenizer_config.json",
        ] {
            std::fs::rename(out.join(name), linked.join(name)).unwrap();
            std::os::unix::fs::symlink(linked.join(name), out.join(name)).unwrap();
        }
        std::os::unix::fs::symlink(root.join("gone"), out.join("dangling")).unwrap();
        let opened = folder::open(&out).unwrap();
        let (hp, u, hash) = super::facts(&opened).unwrap();
        let hash = hash.unwrap();
        let page = model::placement::host_lock::page_bytes().unwrap();
        // The pages one row of every site's table can touch: each stored plane may straddle a
        // page boundary.
        let row_pages: u64 = hp
            .engram
            .as_ref()
            .unwrap()
            .layer_ids
            .iter()
            .map(|l| {
                let t = &opened.rows[&format!("blk.{l}.engram_embd.weight")];
                [&t.values, t.block_scales.as_ref().unwrap()]
                    .iter()
                    .map(|p| 1 + (p.row_bytes - 1).div_ceil(page))
                    .sum::<u64>()
            })
            .sum();
        let run = RunShape {
            ctx: 4352,
            slots: 1,
            batch: 1,
        };
        for ranks in [1, 2, 4] {
            let pools: Vec<RankPool> = (0..ranks)
                .map(|r| RankPool {
                    node: format!("n{r}"),
                    pool_bytes: 128 << 30,
                    device: None,
                })
                .collect();
            let ledgers = preflight(&opened, &pools, run).unwrap();
            assert_eq!(ledgers.len(), ranks);
            for l in &ledgers {
                // Layers [1, 20] at 4,352 positions, one slot: every rank holds the whole cache.
                assert_eq!(
                    (l.need.kv, l.need.shadow),
                    (5_832_704, 8_912_896),
                    "{}",
                    l.line()
                );
                assert!(l.need.row_staging > 0 && l.need.gathers > 0, "{}", l.line());
                let cols = super::engram_cols(&u, l.rank, ranks).unwrap();
                assert_eq!(
                    l.need.row_cache,
                    (cols.end - cols.start) * row_pages * page,
                    "{}",
                    l.line()
                );
            }
            let no_ctx = RunShape { ctx: 0, ..run };
            assert!(preflight(&opened, &pools, no_ctx).is_err());
        }
        let ranks = 4;
        for rank in 0..ranks {
            let sparse = root.join(format!("rank{rank}"));
            std::fs::create_dir(&sparse).unwrap();
            let reads = rank_reads(&opened, rank, ranks).unwrap();
            for w in reads.windows(2) {
                if w[0].0 == w[1].0 {
                    assert!(
                        w[1].1.start > w[0].1.end + READ_GAP,
                        "{:?} then {:?}",
                        w[0],
                        w[1]
                    );
                }
            }
            for (path, r) in reads {
                use std::os::unix::fs::FileExt;
                let from = std::fs::File::open(&path).unwrap();
                let to = std::fs::OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(false)
                    .open(sparse.join(path.file_name().unwrap()))
                    .unwrap();
                to.set_len(from.metadata().unwrap().len()).unwrap();
                let mut buf = vec![0u8; (r.end - r.start) as usize];
                from.read_exact_at(&mut buf, r.start).unwrap();
                to.write_all_at(&buf, r.start).unwrap();
            }
            let part = folder::open(&sparse).unwrap();
            for p in super::planned(&opened, &u, rank, ranks).unwrap() {
                let (a, b) = (
                    opened.source.read(&p).unwrap(),
                    part.source.read(&p).unwrap(),
                );
                assert_eq!(
                    (a.spans().concat(), a.scale),
                    (b.spans().concat(), b.scale),
                    "rank {rank}: {}",
                    p.info.name
                );
            }
            // The rows of the rank's hash columns, both planes, read back from the sparse folder.
            let cols = super::engram_cols(&u, rank, ranks).unwrap();
            let bytes = |path: &Path, r: &Range<u64>| {
                use std::os::unix::fs::FileExt;
                let mut buf = vec![0u8; (r.end - r.start) as usize];
                std::fs::File::open(path)
                    .unwrap()
                    .read_exact_at(&mut buf, r.start)
                    .unwrap();
                buf
            };
            for (site, l) in hp.engram.as_ref().unwrap().layer_ids.iter().enumerate() {
                let name = format!("blk.{l}.engram_embd.weight");
                let (primes, offsets) = (hash.primes(site).unwrap(), hash.offsets(site).unwrap());
                let at = super::engram_rows(primes, offsets, cols.clone()).unwrap();
                let (whole, own) = (
                    opened.rows[&name].rows_at(at.clone()),
                    part.rows[&name].rows_at(at.clone()),
                );
                // One run of each plane, every row of `at` whole: no plane dropped, no row clamped.
                let table = &opened.rows[&name];
                let row_bytes =
                    table.values.row_bytes + table.block_scales.as_ref().unwrap().row_bytes;
                assert_eq!(whole.len(), 2, "rank {rank}: {name} needs both planes");
                assert_eq!(
                    whole.iter().map(|(_, r)| r.end - r.start).sum::<u64>(),
                    (at.end - at.start) * row_bytes,
                    "rank {rank}: {name} {at:?}"
                );
                assert_eq!(whole.len(), own.len(), "rank {rank}: {name}");
                for ((p, r), (q, t)) in whole.iter().zip(&own) {
                    assert_eq!(r, t, "rank {rank}: {name}");
                    let want = bytes(p, r);
                    assert!(want.iter().any(|&b| b != 0), "rank {rank}: {name} {r:?}");
                    assert_eq!(want, bytes(q, t), "rank {rank}: {name} {r:?}");
                }
            }
            std::fs::remove_dir_all(&sparse).unwrap();
        }
        std::fs::remove_dir_all(&root).unwrap();
    }
}
