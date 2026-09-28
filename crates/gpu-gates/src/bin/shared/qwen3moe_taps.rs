//! The qwen3moe layer-tap dump: per sequence, a prompt fed one eager step per
//! id with the layer taps on (`GpuModel::set_layer_taps`), then greedy steps,
//! and at every position the output residual of the layers [`TAPS`] — what an
//! offline draft is fed. `generate_qwen3moe --dump-taps` writes it; the e2e
//! gate's clause (d) holds it to the taps and the plain run's ids.
//!
//! The directory:
//! - `manifest.tsv`: the header `seq n_prompt n_total ids taps`, then a row
//!   per finished sequence (`ids` and `taps` the file names below).
//! - `seq_<k>.ids`: u32 LE, `n_total` ids — the prompt, then the greedy
//!   continuation; the id at position `p` is the one fed there.
//! - `seq_<k>.taps`: f32 LE, row-major `[n_total, TAPS.len(), hidden]`, the
//!   taps after position `p`'s step in [`TAPS`] order. The argmax after the
//!   last position is not in the ids.
//! - `sources.tsv`: each sequence's ids file, first id's index in it and that
//!   file's sha256.
//! - `model.txt`: the model file's path.

use bloomery_gpu::Qwen3moeModel;
use bloomery_gpu_gates::GateError;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// The layers whose output residual (the stream after the layer's combine,
/// before the next layer's norm) a position's row holds, in row order: the
/// draft's auxiliary states — its capture ids 1, 12, 23, 34, 45 are taken
/// after layers 0, 11, 22, 33, 44 — and the last layer's, the final state
/// before the output norm.
pub const TAPS: [usize; 6] = [0, 11, 22, 33, 44, 47];

/// The manifest's header line.
const HEADER: &str = "seq\tn_prompt\tn_total\tids\ttaps";

/// One sequence to dump: its prompt and where it was cut from.
pub struct Seq {
    /// The ids file.
    pub source: PathBuf,
    /// The index of the prompt's first id in that file.
    pub offset: usize,
    pub prompt: Vec<u32>,
}

/// A finished sequence, as its manifest row names it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub seq: usize,
    pub n_prompt: usize,
    pub n_total: usize,
    pub ids: String,
    pub taps: String,
}

/// The ids of `path`, one decimal id a line; an empty or non-numeric line is
/// refused by its line number.
pub fn read_ids(path: &Path) -> Result<Vec<u32>, GateError> {
    std::fs::read_to_string(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .lines()
        .enumerate()
        .map(|(i, l)| {
            l.trim().parse::<u32>().map_err(|e| {
                format!(
                    "{} line {}: {l:?} is not an id ({e})",
                    path.display(),
                    i + 1
                )
                .into()
            })
        })
        .collect()
}

/// The sha256 of `path`, by `sha256sum`.
pub fn sha256(path: &Path) -> Result<String, GateError> {
    let out = std::process::Command::new("sha256sum").arg(path).output()?;
    let text = String::from_utf8_lossy(&out.stdout);
    match text.split_whitespace().next() {
        Some(d) if out.status.success() && d.len() == 64 => Ok(d.to_string()),
        _ => Err(format!("sha256sum {}: {:?}", path.display(), text.trim()).into()),
    }
}

/// `count` prompts of `len` ids from the ids file `path`, window `k` starting
/// at `k · ⌊n / count⌋` of its `n` ids: spread over the whole file and never
/// overlapping. A file too short for that is refused by name.
pub fn windows(path: &Path, count: usize, len: usize) -> Result<Vec<Seq>, GateError> {
    let ids = read_ids(path)?;
    if count == 0 || len == 0 {
        return Err(format!("{count} windows of {len} ids: both must be at least 1").into());
    }
    let stride = ids.len() / count;
    if stride < len {
        return Err(format!(
            "{} holds {} ids: {count} windows of {len} do not fit apart (stride {stride})",
            path.display(),
            ids.len()
        )
        .into());
    }
    Ok((0..count)
        .map(|k| Seq {
            source: path.to_path_buf(),
            offset: k * stride,
            prompt: ids[k * stride..k * stride + len].to_vec(),
        })
        .collect())
}

/// The [`TAPS`] rows of every layer's output residual `taps` (one row per
/// layer, `hidden` values each). A tapped layer past the chain, a last tap
/// that is not the chain's last layer, or a row of another width is refused
/// by name.
pub fn select(taps: &[Vec<f32>], hidden: usize) -> Result<[&[f32]; TAPS.len()], GateError> {
    let n = taps.len();
    if TAPS[TAPS.len() - 1] + 1 != n {
        return Err(format!(
            "the last tap is layer {}, the final state; this chain's last layer is {}",
            TAPS[TAPS.len() - 1],
            n.saturating_sub(1)
        )
        .into());
    }
    let mut out = [&[][..]; TAPS.len()];
    for (o, &l) in out.iter_mut().zip(&TAPS) {
        let row = taps
            .get(l)
            .ok_or_else(|| format!("tap layer {l} past the chain's {n} layers"))?;
        if row.len() != hidden {
            return Err(format!("layer {l}'s tap has {} values, want {hidden}", row.len()).into());
        }
        *o = row;
    }
    Ok(out)
}

/// A dump directory being written: the rows finished so far.
pub struct Dump {
    dir: PathBuf,
    hidden: usize,
    rows: Vec<Row>,
    sources: Vec<String>,
}

impl Dump {
    /// A new dump into `dir` (created; one that holds any file is refused,
    /// so no sequence of an earlier dump stays beside this one's) of a model
    /// of `hidden` values a row, opened from `model`.
    pub fn create(dir: &Path, model: &Path, hidden: usize) -> Result<Dump, GateError> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        if std::fs::read_dir(dir)?.next().is_some() {
            return Err(format!(
                "{} is not empty: a dump goes into an empty directory",
                dir.display()
            )
            .into());
        }
        std::fs::write(
            dir.join("model.txt"),
            format!("{}\nsha256 not recorded in the tree\n", model.display()),
        )?;
        Ok(Dump {
            dir: dir.to_path_buf(),
            hidden,
            rows: Vec::new(),
            sources: Vec::new(),
        })
    }

    /// Sequence `seq` from a reset: its prompt one eager step per id, then
    /// `n_gen` greedy steps, each fed the argmax of the step before; every
    /// step's [`TAPS`] rows into `seq_<k>.taps` as it runs, its ids into
    /// `seq_<k>.ids`, then its manifest and sources rows. A non-finite tap
    /// value or a file of another size than the rows it must hold is refused
    /// by name. Leaves the model eager with its taps on.
    pub fn seq(
        &mut self,
        m: &mut Qwen3moeModel,
        seq: &Seq,
        n_gen: usize,
    ) -> Result<&Row, GateError> {
        let k = self.rows.len();
        let n_prompt = seq.prompt.len();
        let n_total = n_prompt + n_gen;
        if n_prompt == 0 {
            return Err(format!("sequence {k}: a prompt of no id").into());
        }
        m.set_layer_taps(true)?;
        m.reset()?;
        let (ids_name, taps_name) = (format!("seq_{k}.ids"), format!("seq_{k}.taps"));
        let taps_path = self.dir.join(&taps_name);
        let mut w = BufWriter::with_capacity(1 << 20, File::create(&taps_path)?);
        let mut ids = Vec::with_capacity(n_total);
        let mut next = 0;
        for p in 0..n_total {
            let id = seq.prompt.get(p).copied().unwrap_or(next);
            next = m.step(&[id])?;
            ids.push(id);
            let all = m.layer_taps()?;
            for (&l, row) in TAPS.iter().zip(select(&all, self.hidden)?) {
                if let Some(i) = row.iter().position(|v| !v.is_finite()) {
                    return Err(format!(
                        "sequence {k} position {p}: layer {l}'s tap value {i} is {}",
                        row[i]
                    )
                    .into());
                }
                let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
                w.write_all(&bytes)?;
            }
        }
        w.into_inner().map_err(|e| e.into_error())?.sync_all()?;
        let ids_bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(self.dir.join(&ids_name), ids_bytes)?;
        let row = Row {
            seq: k,
            n_prompt,
            n_total,
            ids: ids_name,
            taps: taps_name,
        };
        check_sizes(&self.dir, &row, self.hidden)?;
        self.sources.push(format!(
            "{k}\t{}\t{}\t{}",
            seq.source.display(),
            seq.offset,
            sha256(&seq.source)?
        ));
        self.rows.push(row);
        self.write_tables()?;
        Ok(&self.rows[k])
    }

    /// `manifest.tsv` and `sources.tsv` over the rows finished so far.
    fn write_tables(&self) -> Result<(), GateError> {
        let mut manifest = format!("{HEADER}\n");
        for r in &self.rows {
            manifest.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\n",
                r.seq, r.n_prompt, r.n_total, r.ids, r.taps
            ));
        }
        std::fs::write(self.dir.join("manifest.tsv"), manifest)?;
        let mut sources = "seq\tsource\toffset\tsha256\n".to_string();
        for s in &self.sources {
            sources.push_str(s);
            sources.push('\n');
        }
        std::fs::write(self.dir.join("sources.tsv"), sources)?;
        Ok(())
    }
}

/// The byte count `row`'s files must hold: `n_total` u32 ids and `n_total ·
/// TAPS.len() · hidden` f32 values.
pub fn sizes(row: &Row, hidden: usize) -> (u64, u64) {
    let n = row.n_total as u64;
    (4 * n, 4 * n * TAPS.len() as u64 * hidden as u64)
}

/// `row`'s two files under `dir`, refused by name unless each holds the
/// bytes [`sizes`] says.
fn check_sizes(dir: &Path, row: &Row, hidden: usize) -> Result<(), GateError> {
    let (want_ids, want_taps) = sizes(row, hidden);
    for (name, want) in [(&row.ids, want_ids), (&row.taps, want_taps)] {
        let got = std::fs::metadata(dir.join(name))?.len();
        if got != want {
            return Err(format!("{}/{name}: {got} bytes, want {want}", dir.display()).into());
        }
    }
    Ok(())
}

/// The manifest of the dump in `dir`, every row's files checked for their
/// sizes; a header of other columns, a row out of sequence order or of
/// another shape is refused by name.
pub fn read_manifest(dir: &Path, hidden: usize) -> Result<Vec<Row>, GateError> {
    let path = dir.join("manifest.tsv");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut lines = text.lines();
    if lines.next() != Some(HEADER) {
        return Err(format!("{}: the header is not {HEADER:?}", path.display()).into());
    }
    let mut rows = Vec::new();
    for (i, line) in lines.enumerate() {
        let c: Vec<&str> = line.split('\t').collect();
        let bad = || format!("{} row {i}: {line:?}", path.display());
        let [seq, n_prompt, n_total, ids, taps] = c[..] else {
            return Err(bad().into());
        };
        let row = Row {
            seq: seq.parse().map_err(|_| bad())?,
            n_prompt: n_prompt.parse().map_err(|_| bad())?,
            n_total: n_total.parse().map_err(|_| bad())?,
            ids: ids.to_string(),
            taps: taps.to_string(),
        };
        if row.seq != i || row.n_prompt == 0 || row.n_prompt > row.n_total {
            return Err(bad().into());
        }
        check_sizes(dir, &row, hidden)?;
        rows.push(row);
    }
    Ok(rows)
}
