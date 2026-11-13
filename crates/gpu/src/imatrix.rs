//! The importance-matrix collector: what llama.cpp's `llama-imatrix` records of a prompt pass, taken
//! from the card's own activations. One accumulator per weight name holds the column sum of squares
//! of the rows that reach the weight (`in_sum2`, f32, `[ne0, 1]` for a dense linear and
//! `[ne0, n_expert]` for a routed stack, the rows of an expert being those the router sent there)
//! and the row counts, and [`Collector::write`] renders them in llama's GGUF form
//! (`general.type = "imatrix"`, `imatrix.datasets`, `imatrix.chunk_count`, `imatrix.chunk_size`;
//! per weight `<name>.in_sum2` and `<name>.counts`, names sorted).
//!
//! One owner for every model: a layer's code hands the collector `(name, input rows)` at each linear
//! it runs ([`Site::project`], the rows kit, the head128 attention, the routed block's batch and the
//! card experts' run) and the collector knows no architecture. Nothing is attached by default:
//! [`collector`] is one atomic read, and a call that finds none does nothing, so the unattached
//! walk's launches and graph are the ones without this file. [`attach`] is the CLI's, once per
//! process.
//!
//! Semantics held to llama's (`tools/imatrix/imatrix.cpp`, `collect_imatrix`): a dense weight adds
//! `x²` of every input row in row order, one count per row; a routed weight adds the row of each
//! `(row, slot)` the router picked to that expert's column sums, one count per pick. Where ours
//! differs: a weight whose input the engine never holds as an f32 buffer is not collected, and the
//! walk names it — the host tier's and the tier cards' routed *downs* (their combines are quantized
//! inside the host union call and the tier leg, never an f32 buffer the walk holds; the *gate* and
//! *up* of every pick are collected, for the router's normed rows sit on the card whatever tier runs
//! the expert), a gated block's *down* on the card (the wide path's SwiGLU writes the down's own
//! quantized form, not f32 rows), and the head (as `llama-imatrix` without `--process-output`).
//! [`Collector::feed_routed_carded`] keeps only the picks whose place is a card slot: the tiers'
//! downs are named above, and their columns stay zero with zero counts, which the quantizer reads
//! as weight 1 ([`gguf::iq_quant::imatrix_weights`]).

use crate::{Gpu, GpuError, hybrid::HOST, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use gguf::Value;
use gguf::write::{Layout, TensorDecl, Writer};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

const WHAT: &str = "imatrix";

/// ggml's type id of f32.
const GGML_F32: u32 = 0;

#[cuda_module]
mod imatrix_kernels {
    use super::*;

    /// `acc[c] += Σ_r x[r·k + c]²` over `rows` rows, rows in order, one thread a column; the first
    /// thread also adds `rows` to `counts[0]`. A column's sum is one thread's, so the order is the
    /// row order and nothing is atomic.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (x.len() >= rows * k, acc.len() >= k, counts.len() >= 1)
    )]
    pub fn imat_dense_sum2(
        x: &[f32],
        rows: u32,
        k: u32,
        mut acc: DisjointSlice<f32>,
        mut counts: DisjointSlice<f32>,
    ) {
        let c = thread::index_1d().get();
        let (rows, k) = (rows as usize, k as usize);
        if c >= k {
            return;
        }
        // SAFETY: c < k <= acc.len() by the contract and thread c is acc[c]'s only writer; r·k + c
        // < rows·k <= x.len().
        unsafe {
            let mut s = *acc.get_unchecked_mut(c);
            for r in 0..rows {
                let v = *x.get_unchecked(r * k + c);
                s += v * v;
            }
            *acc.get_unchecked_mut(c) = s;
            if c == 0 {
                // SAFETY: thread 0 is counts[0]'s only writer; counts.len() >= 1.
                *counts.get_unchecked_mut(0) += rows as f32;
            }
        }
    }

    /// For each pick `(r, j)` of `rows × used` (expert `ids[r·used + j]`), `acc[e·k + c] += x[xr·k +
    /// c]²` with `xr = r` (the experts' shared input row, `per_pick = 0`) or `xr = r·used + j` (a row
    /// per pick, `per_pick = 1`), picks in order, one thread a column; the first thread also adds one
    /// to `counts[e]`. An id past `n_expert` is skipped and counted nowhere: the host holds the
    /// counts' total to `rows × used` and refuses the entry by name when a pick went missing.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            ids.len() >= rows * used,
            x.len() >= x_rows * k,
            acc.len() >= n_expert * k,
            counts.len() >= n_expert
        )
    )]
    pub fn imat_routed_sum2(
        x: &[f32],
        ids: &[u32],
        rows: u32,
        used: u32,
        x_rows: u32,
        per_pick: u32,
        k: u32,
        n_expert: u32,
        mut acc: DisjointSlice<f32>,
        mut counts: DisjointSlice<f32>,
    ) {
        let c = thread::index_1d().get();
        let (rows, used, x_rows, k, n_expert) = (
            rows as usize,
            used as usize,
            x_rows as usize,
            k as usize,
            n_expert as usize,
        );
        if c >= k {
            return;
        }
        for r in 0..rows {
            for j in 0..used {
                // SAFETY: r·used + j < rows·used <= ids.len() by the contract.
                let e = unsafe { *ids.get_unchecked(r * used + j) } as usize;
                let xr = if per_pick == 0 { r } else { r * used + j };
                if e >= n_expert || xr >= x_rows {
                    continue;
                }
                // SAFETY: e < n_expert and c < k put e·k + c below n_expert·k <= acc.len(); thread c
                // is the only writer of the indices congruent to c mod k; xr·k + c < x_rows·k <=
                // x.len(); thread 0 is counts[e]'s only writer, e < n_expert <= counts.len().
                unsafe {
                    let v = *x.get_unchecked(xr * k + c);
                    *acc.get_unchecked_mut(e * k + c) += v * v;
                    if c == 0 {
                        *counts.get_unchecked_mut(e) += 1.0;
                    }
                }
            }
        }
    }

    /// The picks the card holds, one thread a pick: `out[i]` is `ids[i]` when pick `i`'s expert is
    /// below `n_expert` and its place `places[i]` is a card slot, else `n_expert` (the sum kernel's
    /// skip mark); each kept pick adds one to `kept[0]`, the count the entry's exactness check holds
    /// ([`Collector::write`]), so a pick the filter dropped that the sums kept, or the other way
    /// round, is refused by name at the write.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            ids.len() >= n,
            places.len() >= n,
            out.len() >= n,
            kept.len() >= 1
        )
    )]
    pub fn imat_card_picks(
        ids: &[u32],
        places: &[u32],
        n: u32,
        n_expert: u32,
        mut out: DisjointSlice<u32>,
        mut kept: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let n = n as usize;
        if i >= n {
            return;
        }
        // SAFETY: i < n <= ids.len(), places.len() and out.len() by the contract.
        unsafe {
            let (id, place) = (*ids.get_unchecked(i), *places.get_unchecked(i));
            let on = id < n_expert && place != HOST;
            *out.get_unchecked_mut(i) = if on { id } else { n_expert };
            if on {
                // SAFETY: kept.len() >= 1 by the contract; one relaxed add a kept pick, the only
                // writers of cell 0 being this kernel's launches, in stream order.
                DeviceAtomicF32::from_ptr(kept.as_mut_ptr())
                    .fetch_add(1.0, AtomicOrdering::Relaxed);
            }
        }
    }
}

/// One weight's accumulator: `n_mat` matrices of `ne0` columns (1 for a dense linear, the expert
/// count for a routed stack) and the counts, on the stream of the card the weight lives on.
struct Entry {
    stream: Arc<CudaStream>,
    ne0: usize,
    n_mat: usize,
    acc: DeviceBuffer<f32>,
    counts: DeviceBuffer<f32>,
    /// The count total the entry must hold: rows fed (dense), picks fed (routed) or the total a
    /// seed carried over ([`Collector::seed`]).
    expect: u64,
    /// The picks [`Collector::feed_routed_carded`] kept, one cell the filter kernel adds to; the
    /// exactness check reads it beside `expect`. `None` until a carded feed.
    kept: Option<DeviceBuffer<f32>>,
}

/// The accumulators of a pass, by GGUF weight name, and the marked picks the carded feeds share:
/// one buffer on the stream of the first carded feed, reused launch to launch in stream order.
pub struct Collector {
    entries: Mutex<BTreeMap<String, Entry>>,
    picks: Mutex<Option<Picks>>,
}

/// The carded feeds' marked picks, on one card's stream.
struct Picks {
    stream: Arc<CudaStream>,
    ids: DeviceBuffer<u32>,
}

static COLLECTOR: OnceLock<Collector> = OnceLock::new();

/// The collector, once attached; `None` otherwise. One atomic read.
#[must_use]
pub fn collector() -> Option<&'static Collector> {
    COLLECTOR.get()
}

/// Attach the collector for this process: from here every linear the walk runs feeds it. Idempotent.
pub fn attach() -> &'static Collector {
    COLLECTOR.get_or_init(|| Collector {
        entries: Mutex::new(BTreeMap::new()),
        picks: Mutex::new(None),
    })
}

/// The 1-D launch over `k` columns.
fn columns(what: &'static str, k: usize) -> Result<LaunchConfig1D, GpuError> {
    Ok(LaunchConfig1D::new(
        launch_u32(what, "grid", k.div_ceil(256))?,
        256,
        0,
    ))
}

impl Collector {
    /// Layer code's hook for a dense linear: `rows` token-major rows of `k` values of `x` reach
    /// `name`'s weight. The sums land on the stream `gpu` runs the layer on, in launch order.
    pub fn feed_dense(
        &self,
        gpu: &Gpu,
        name: &str,
        x: &DeviceBuffer<f32>,
        (rows, k): (usize, usize),
    ) -> Result<(), GpuError> {
        const FEED: &str = "imatrix::feed_dense";
        if rows == 0 || k == 0 || x.len() < rows * k {
            return Err(GpuError::shape(
                FEED,
                format!("{name}: {rows} rows of {k}, x.len() {}", x.len()),
            ));
        }
        let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        let e = Self::entry(&mut entries, gpu, name, (k, 1), FEED)?;
        // SAFETY: this package owns the embedded device bundle of the module above; the launcher
        // checks the launch contract.
        let module = unsafe { crate::shared_module!(imatrix_kernels, gpu.context())? };
        let prep = module.prepare_imat_dense_sum2(columns(FEED, k)?)?;
        module.imat_dense_sum2(
            gpu.stream(),
            &prep,
            x,
            launch_u32(FEED, "rows", rows)?,
            launch_u32(FEED, "k", k)?,
            &mut e.acc,
            &mut e.counts,
        )?;
        e.expect += rows as u64;
        Ok(())
    }

    /// Layer code's hook for a routed stack: the router picked `ids` (`rows × used`, global expert
    /// ids) and each pick's input row of `k` values is `x`'s row `r` (`per_pick` false: the experts'
    /// shared input) or row `r · used + j` (`per_pick` true: a row per pick, the down's).
    pub fn feed_routed(
        &self,
        gpu: &Gpu,
        name: &str,
        (x, ids): (&DeviceBuffer<f32>, &DeviceBuffer<u32>),
        (rows, used, k, n_expert): (usize, usize, usize, usize),
        per_pick: bool,
    ) -> Result<(), GpuError> {
        const FEED: &str = "imatrix::feed_routed";
        let x_rows = if per_pick { rows * used } else { rows };
        if rows == 0 || used == 0 || k == 0 || x.len() < x_rows * k || ids.len() < rows * used {
            return Err(GpuError::shape(
                FEED,
                format!(
                    "{name}: {rows} rows of {used} picks, {k} values, x.len() {}, ids.len() {}",
                    x.len(),
                    ids.len()
                ),
            ));
        }
        let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        let e = Self::entry(&mut entries, gpu, name, (k, n_expert), FEED)?;
        // SAFETY: as in `feed_dense`.
        let module = unsafe { crate::shared_module!(imatrix_kernels, gpu.context())? };
        let prep = module.prepare_imat_routed_sum2(columns(FEED, k)?)?;
        module.imat_routed_sum2(
            gpu.stream(),
            &prep,
            x,
            ids,
            launch_u32(FEED, "rows", rows)?,
            launch_u32(FEED, "used", used)?,
            launch_u32(FEED, "x_rows", x_rows)?,
            u32::from(per_pick),
            launch_u32(FEED, "k", k)?,
            launch_u32(FEED, "n_expert", n_expert)?,
            &mut e.acc,
            &mut e.counts,
        )?;
        e.expect += (rows * used) as u64;
        Ok(())
    }

    /// Layer code's hook for a routed stack whose down rows the card route holds its own picks of:
    /// [`Collector::feed_routed`] over `x`'s per-pick rows (`per_pick` true), keeping only the picks
    /// whose place in `places` (the card route's `sel`, one a pick) is a card slot — the host and
    /// tier picks' combines never sit on the card as f32 rows, and the module doc names them. The
    /// kept picks land in the entry's own count beside `expect`, so the write's exactness check
    /// holds here too. The marked picks ride one scratch buffer, reused in stream order.
    pub fn feed_routed_carded(
        &self,
        gpu: &Gpu,
        name: &str,
        (x, ids, places): (&DeviceBuffer<f32>, &DeviceBuffer<u32>, &DeviceBuffer<u32>),
        (rows, used, k, n_expert): (usize, usize, usize, usize),
    ) -> Result<(), GpuError> {
        const FEED: &str = "imatrix::feed_routed_carded";
        let picks = rows.checked_mul(used).ok_or(GpuError::shape(
            FEED,
            format!("{name}: {rows} rows of {used} picks over u32"),
        ))?;
        if rows == 0
            || used == 0
            || k == 0
            || x.len() < picks * k
            || ids.len() < picks
            || places.len() < picks
        {
            return Err(GpuError::shape(
                FEED,
                format!(
                    "{name}: {rows} rows of {used} picks, {k} values, x.len() {}, ids.len() {}, \
                     places.len() {}",
                    x.len(),
                    ids.len(),
                    places.len()
                ),
            ));
        }
        // SAFETY: as in `feed_dense`.
        let module = unsafe { crate::shared_module!(imatrix_kernels, gpu.context())? };
        let mut marked = self.picks.lock().unwrap_or_else(PoisonError::into_inner);
        let stream = gpu.stream_handle();
        let scratch = match marked.as_mut() {
            Some(p) if Arc::ptr_eq(&p.stream, &stream) => &mut p.ids,
            Some(_) => {
                return Err(GpuError::state(
                    FEED,
                    "a carded feed from a second card's stream: the marked picks sit on one",
                ));
            }
            None => {
                &mut marked
                    .insert(Picks {
                        stream,
                        ids: DeviceBuffer::zeroed(gpu.stream(), picks)?,
                    })
                    .ids
            }
        };
        if scratch.len() < picks {
            *scratch = DeviceBuffer::zeroed(gpu.stream(), picks)?;
        }
        let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        let e = Self::entry(&mut entries, gpu, name, (k, n_expert), FEED)?;
        if e.kept.is_none() {
            e.kept = Some(DeviceBuffer::zeroed(gpu.stream(), 1)?);
        }
        let Entry {
            acc, counts, kept, ..
        } = e;
        let n = launch_u32(FEED, "picks", picks)?;
        let grid = LaunchConfig1D::new(launch_u32(FEED, "grid", picks.div_ceil(256))?, 256, 0);
        let prep = module.prepare_imat_card_picks(grid)?;
        module.imat_card_picks(
            gpu.stream(),
            &prep,
            ids,
            places,
            n,
            launch_u32(FEED, "n_expert", n_expert)?,
            scratch,
            kept.as_mut().expect("made above"),
        )?;
        let x_rows = picks;
        let prep = module.prepare_imat_routed_sum2(columns(FEED, k)?)?;
        module.imat_routed_sum2(
            gpu.stream(),
            &prep,
            x,
            scratch,
            launch_u32(FEED, "rows", rows)?,
            launch_u32(FEED, "used", used)?,
            launch_u32(FEED, "x_rows", x_rows)?,
            1,
            launch_u32(FEED, "k", k)?,
            launch_u32(FEED, "n_expert", n_expert)?,
            acc,
            counts,
        )?;
        Ok(())
    }

    /// Start (or grow) `name`'s entry from a partial collection — a run that `--save-every` wrote
    /// and a later run continues: `sums` and `counts` are that file's `<name>.in_sum2` and
    /// `.counts`, and the entry's count bookkeeping takes their total, so the write's exactness
    /// check keeps holding over the runs' sums together. A name seeded twice, or with sums of
    /// another shape, is refused by name.
    pub fn seed(
        &self,
        gpu: &Gpu,
        name: &str,
        (ne0, n_mat): (usize, usize),
        sums: &[f32],
        counts: &[f32],
    ) -> Result<(), GpuError> {
        const SEED: &str = "imatrix::seed";
        if sums.len() != ne0 * n_mat || counts.len() != n_mat {
            return Err(GpuError::shape(
                SEED,
                format!(
                    "{name}: {} sums and {} counts of {ne0} × {n_mat}",
                    sums.len(),
                    counts.len()
                ),
            ));
        }
        let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(e) = entries.get(name)
            && (e.expect != 0 || e.kept.is_some())
        {
            return Err(GpuError::shape(
                SEED,
                format!("{name}: seeded beside a collection already under way"),
            ));
        }
        let e = Self::entry(&mut entries, gpu, name, (ne0, n_mat), SEED)?;
        e.acc.copy_from_host(gpu.stream(), sums)?;
        e.counts.copy_from_host(gpu.stream(), counts)?;
        let total: u64 = counts
            .iter()
            .try_fold(0u64, |a, &c| {
                let c = c.round();
                (c >= 0.0 && c <= u32::MAX as f32)
                    .then(|| a.checked_add(c as u64))
                    .flatten()
            })
            .ok_or_else(|| {
                GpuError::shape(SEED, format!("{name}: a count that is no count of picks"))
            })?;
        e.expect = total;
        Ok(())
    }

    /// `name`'s entry, made on `gpu`'s stream at the first feed; a later feed of another shape is
    /// refused by name.
    fn entry<'a>(
        entries: &'a mut BTreeMap<String, Entry>,
        gpu: &Gpu,
        name: &str,
        (ne0, n_mat): (usize, usize),
        what: &'static str,
    ) -> Result<&'a mut Entry, GpuError> {
        if !entries.contains_key(name) {
            let stream = gpu.stream_handle();
            let e = Entry {
                acc: DeviceBuffer::zeroed(&stream, ne0 * n_mat)?,
                counts: DeviceBuffer::zeroed(&stream, n_mat)?,
                stream,
                ne0,
                n_mat,
                expect: 0,
                kept: None,
            };
            entries.insert(name.to_string(), e);
        }
        let e = entries.get_mut(name).expect("inserted above");
        if (e.ne0, e.n_mat) != (ne0, n_mat) {
            return Err(GpuError::shape(
                what,
                format!(
                    "{name}: fed as {ne0} × {n_mat}, collected as {} × {}",
                    e.ne0, e.n_mat
                ),
            ));
        }
        Ok(e)
    }

    /// The names collected so far, sorted.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        let entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        entries.keys().cloned().collect()
    }

    /// Read every entry back and write llama's GGUF form to `path`: the dataset's name, the chunk
    /// size and the chunk count are the caller's. An entry whose counts do not total what was fed
    /// (a pick the kernel skipped) or that holds a non-finite sum is refused by name, the file
    /// unwritten. Blocking.
    pub fn write(
        &self,
        path: &Path,
        (chunk_size, chunk_count): (u32, u32),
        dataset: &str,
    ) -> Result<(), GpuError> {
        let entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        let mut host = Vec::with_capacity(entries.len());
        for (name, e) in entries.iter() {
            let sums = crate::tensor::to_host(&e.stream, &e.acc)?;
            let counts = crate::tensor::to_host(&e.stream, &e.counts)?;
            let kept: f64 = e
                .kept
                .as_ref()
                .map(|k| crate::tensor::to_host(&e.stream, k))
                .transpose()?
                .map_or(0.0, |k| k.iter().map(|&c| f64::from(c)).sum());
            let total: f64 = counts.iter().map(|&c| f64::from(c)).sum();
            let want = e.expect as f64 + kept;
            if total != want {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "{name}: counts total {total}, {want} fed: a pick the kernel could not place",
                    ),
                ));
            }
            if let Some(i) = sums.iter().position(|v| !v.is_finite()) {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{name}: a non-finite sum at {i} (column {})", i % e.ne0),
                ));
            }
            host.push((name.as_str(), e.ne0, e.n_mat, sums, counts));
        }
        render(path, (chunk_size, chunk_count), dataset, &host)
    }
}

/// One weight as it is written: its name, `ne0`, `n_mat`, the sums and the counts.
type Row<'a> = (&'a str, usize, usize, Vec<f32>, Vec<f32>);

/// llama's file from `rows` (sorted by name): metadata, then each weight's `in_sum2` and `counts`.
fn render(
    path: &Path,
    (chunk_size, chunk_count): (u32, u32),
    dataset: &str,
    rows: &[Row<'_>],
) -> Result<(), GpuError> {
    let io = |e: &dyn std::fmt::Display| GpuError::shape(WHAT, format!("{}: {e}", path.display()));
    let kvs = vec![
        (
            "general.type".to_string(),
            Value::String("imatrix".to_string()),
        ),
        (
            "imatrix.datasets".to_string(),
            Value::Array(vec![Value::String(dataset.to_string())]),
        ),
        ("imatrix.chunk_count".to_string(), Value::U32(chunk_count)),
        ("imatrix.chunk_size".to_string(), Value::U32(chunk_size)),
    ];
    let mut decls = Vec::with_capacity(rows.len() * 2);
    let mut data: Vec<(String, Vec<u8>)> = Vec::with_capacity(rows.len() * 2);
    let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    for (name, ne0, n_mat, sums, counts) in rows {
        for (suffix, dims, vals) in [
            ("in_sum2", [*ne0 as u64, *n_mat as u64], sums),
            ("counts", [1, *n_mat as u64], counts),
        ] {
            let full = format!("{name}.{suffix}");
            decls.push(TensorDecl {
                name: full.clone(),
                dims: dims.to_vec(),
                type_id: GGML_F32,
                nbytes: 4 * vals.len() as u64,
            });
            data.push((full, bytes(vals)));
        }
    }
    let layout = Layout::new(&kvs, decls).map_err(|e| io(&e))?;
    let file = std::fs::File::create(path).map_err(|e| io(&e))?;
    let mut w = Writer::new(std::io::BufWriter::new(file), layout).map_err(|e| io(&e))?;
    for (name, b) in &data {
        w.tensor(name, b).map_err(|e| io(&e))?;
    }
    w.finish().map_err(|e| io(&e))?;
    Ok(())
}
