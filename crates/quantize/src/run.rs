//! The set writer: every shard of the input written again under the output prefix with the same metadata and
//! the same tensors in the same order, each tensor the plan says to quantize quantized by a [`Backend`] and every
//! other tensor copied byte for byte. Each shard is written as `<name>.partial` and renamed when whole; a failed
//! run removes every file it made. The next batch of experts is read and scanned while the backend works on the
//! current one.

use crate::check::{check_rows, compare_row, host_row};
use crate::cli::Args;
use crate::plan::{Fate, LlamaPlan, Plan, importance};
use crate::rules::Rules;
use crate::scan::{row_bytes, scan_rows};
use crate::{Error, Result};
use gguf::iq_quant::{GridTables, IqKind};
use gguf::quant::GgmlType;
use gguf::write::Writer;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// A backend's error, whatever it is made of.
pub type BackendError = Box<dyn std::error::Error>;

/// What quantizes a batch of rows: the card (the binary supplies it) or the host rule ([`crate::host`]).
pub trait Backend {
    /// What the run prints it as.
    fn name(&self) -> String;
    /// The tensor quantized next: its kind and its importance, `ne0` values an expert.
    fn bind(&mut self, kind: IqKind, qw: &[f32]) -> std::result::Result<(), BackendError>;
    /// Rows `row0 ..` of the bound tensor, as `src` bytes of `ty` (whole rows of `ne0` values), into the file's
    /// block bytes.
    fn quantize(
        &self,
        ty: GgmlType,
        src: &[u8],
        ne0: usize,
        row0: usize,
        rows_per_expert: usize,
    ) -> std::result::Result<Vec<u8>, BackendError>;
}

/// The opened inputs and the plan over them.
pub struct Prepared {
    pub set: gguf::Split,
    pub im: gguf::Gguf,
    pub plan: Plan,
}

fn exists(p: &Path) -> Result<()> {
    std::fs::metadata(p)
        .map(|_| ())
        .map_err(|source| Error::Io {
            path: p.display().to_string(),
            source,
        })
}

fn io(p: &Path, source: std::io::Error) -> Error {
    Error::Io {
        path: p.display().to_string(),
        source,
    }
}

/// Open the set, the imatrix, the rules and the optional llama plan, and decide every tensor; reads no tensor data
/// but the imatrix's.
pub fn prepare(a: &Args) -> Result<Prepared> {
    for p in [&a.first, &a.imatrix, &a.rules]
        .into_iter()
        .chain(a.llama_plan.as_ref())
    {
        exists(p)?;
    }
    let text = |p: &Path| std::fs::read_to_string(p).map_err(|e| io(p, e));
    let rules = Rules::parse(&text(&a.rules)?)
        .map_err(|e| Error::refused(format!("{}: {e}", a.rules.display())))?;
    let llama = a
        .llama_plan
        .as_ref()
        .map(|p| LlamaPlan::parse(&text(p)?, &p.display().to_string()))
        .transpose()?;
    let set = gguf::Split::open(&a.first)
        .map_err(|e| Error::refused(format!("{}: {e}", a.first.display())))?;
    let im = gguf::Gguf::open(&a.imatrix)
        .map_err(|e| Error::refused(format!("{}: {e}", a.imatrix.display())))?;
    let plan = Plan::build(&set, &rules, llama.as_ref(), &im, a.only, &a.out)?;
    Ok(Prepared { set, im, plan })
}

/// The room check: the output's drive has the bytes the output files need, or the run is refused with both numbers.
pub fn check_room(a: &Args, plan: &Plan) -> Result<()> {
    let dir = a.out_dir();
    let free =
        os::free_bytes(&dir).map_err(|e| Error::refused(format!("{}: {e}", dir.display())))?;
    if free < plan.out_file_bytes {
        return Err(Error::refused(format!(
            "the output set needs {} B under {}, the drive has {free} B free",
            plan.out_file_bytes,
            dir.display()
        )));
    }
    Ok(())
}

/// The outputs of the plan exist nowhere yet (a stale `.partial` counts).
fn check_absent(plan: &Plan) -> Result<()> {
    for o in &plan.outs {
        for p in [o.clone(), PathBuf::from(format!("{}.partial", o.display()))] {
            if p.exists() {
                return Err(Error::refused(format!(
                    "{} exists: not overwritten",
                    p.display()
                )));
            }
        }
    }
    Ok(())
}

/// What a run did.
#[derive(Debug)]
pub struct Summary {
    pub quantized: usize,
    pub weights: f64,
    pub quant_s: f64,
    pub wall_s: f64,
    pub files: Vec<PathBuf>,
}

/// Write the set. `log` gets each progress line. A failure removes the files this run made.
pub fn run(
    a: &Args,
    prep: Prepared,
    backend: &mut dyn Backend,
    log: &mut dyn FnMut(&str),
) -> Result<Summary> {
    let Prepared { set, im, plan } = prep;
    check_absent(&plan)?;
    check_room(a, &plan)?;
    let mut made = Vec::new();
    let r = write_all(a, &set, &im, plan, backend, log, &mut made);
    if r.is_err() {
        for f in &made {
            let _ = std::fs::remove_file(f);
        }
    }
    r
}

fn write_all(
    a: &Args,
    set: &gguf::Split,
    im: &gguf::Gguf,
    plan: Plan,
    backend: &mut dyn Backend,
    log: &mut dyn FnMut(&str),
    made: &mut Vec<PathBuf>,
) -> Result<Summary> {
    let n = plan.shards.len();
    log(&format!(
        "bloomery-quantize: {n} shard(s) from {}; {} tensors quantized by the {}, the rest copied; {:.2} GB to write under {}",
        a.first.display(),
        plan.quantized,
        backend.name(),
        plan.out_file_bytes as f64 / 1e9,
        a.out
    ));
    let t_all = Instant::now();
    let (mut t_q, mut w_q, mut n_q) = (0.0f64, 0.0f64, 0usize);
    let mut tables: [Option<GridTables>; 2] = [None, None];
    let mut files = Vec::new();
    for (i, layout) in plan.layouts.into_iter().enumerate() {
        let shard = set.shard(i).expect("i < shard_count");
        let out = plan.outs[i].clone();
        let partial = PathBuf::from(format!("{}.partial", out.display()));
        made.push(partial.clone());
        let file = std::fs::File::create(&partial).map_err(|e| io(&partial, e))?;
        let wr = |e: gguf::write::WriteError| Error::refused(format!("{}: {e}", partial.display()));
        let mut w =
            Writer::new(std::io::BufWriter::with_capacity(1 << 24, file), layout).map_err(wr)?;
        for (t, item) in shard.iter_tensors().zip(&plan.shards[i]) {
            let data = shard
                .data(t)
                .map_err(|e| Error::refused(format!("{}: {e}", t.name)))?;
            let Fate::Quantize(kind) = item.fate else {
                w.tensor(&t.name, data).map_err(wr)?;
                continue;
            };
            let qw = importance(im, t)?;
            backend
                .bind(kind, &qw)
                .map_err(|e| Error::refused(format!("{}: {e}", t.name)))?;
            let (ne0, ne1) = (t.dims[0] as usize, t.dims[1] as usize);
            let n_exp: usize = t.dims[2..].iter().product::<u64>() as usize;
            let tab = &*tables[usize::from(kind == IqKind::Iq1S)]
                .get_or_insert_with(|| GridTables::new(kind));
            let t0 = Instant::now();
            quantize_tensor(
                a,
                backend,
                &mut w,
                t,
                data,
                kind,
                tab,
                &qw,
                (ne0, ne1, n_exp),
            )?;
            let s = t0.elapsed().as_secs_f64();
            let wt = (ne0 * ne1 * n_exp) as f64;
            (t_q, w_q, n_q) = (t_q + s, w_q + wt, n_q + 1);
            log(&format!(
                "  {} {:?} -> {kind:?}: {wt:.3e} weights in {s:.1} s (read overlapped) = {:.3e} weights/s",
                t.name,
                t.ty,
                wt / s
            ));
        }
        w.finish()
            .map_err(wr)?
            .into_inner()
            .map_err(|e| io(&partial, e.into_error()))?
            .sync_all()
            .map_err(|e| io(&partial, e))?;
        std::fs::rename(&partial, &out).map_err(|e| io(&out, e))?;
        made.pop();
        made.push(out.clone());
        log(&format!("  shard {}/{n}: {}", i + 1, out.display()));
        files.push(out);
    }
    Ok(Summary {
        quantized: n_q,
        weights: w_q,
        quant_s: t_q,
        wall_s: t_all.elapsed().as_secs_f64(),
        files,
    })
}

/// One tensor: batches of experts read and scanned on a thread while the backend quantizes the previous batch,
/// each batch's `--check` rows run through the host rule, the bytes written in order.
#[allow(
    clippy::too_many_arguments,
    reason = "one tensor's writer, source, grids, importance and geometry"
)]
fn quantize_tensor<W: std::io::Write>(
    a: &Args,
    backend: &dyn Backend,
    w: &mut Writer<W>,
    t: &gguf::TensorInfo,
    data: &[u8],
    kind: IqKind,
    tables: &GridTables,
    qw: &[f32],
    (ne0, ne1, n_exp): (usize, usize, usize),
) -> Result<()> {
    let rb = row_bytes(t.ty, ne0).expect("the plan checked the row geometry");
    let eb = data.len() / n_exp;
    debug_assert_eq!(eb, rb * ne1);
    let picks = check_rows(ne1 * n_exp, a.check);
    let batch = a.batch;
    let mut err: Option<Error> = None;
    let wrote = std::thread::scope(|sc| {
        let read = |e0: usize, e1: usize| -> Result<Vec<u8>> {
            let src = data[e0 * eb..e1 * eb].to_vec();
            scan_rows(&t.name, t.ty, &src, ne0, e0 * ne1, ne1)?;
            Ok(src)
        };
        let mut pending = Some(sc.spawn(move || read(0, batch.min(n_exp))));
        let mut e0 = 0;
        let parts = std::iter::from_fn(|| {
            let joined = pending.take()?.join();
            let src = match joined {
                Ok(Ok(s)) => s,
                Ok(Err(e)) => {
                    err = Some(e);
                    return None;
                }
                Err(_) => {
                    err = Some(Error::refused("the reader thread panicked"));
                    return None;
                }
            };
            let e1 = (e0 + batch).min(n_exp);
            if e1 < n_exp {
                let e2 = (e1 + batch).min(n_exp);
                pending = Some(sc.spawn(move || read(e1, e2)));
            }
            let out = match backend.quantize(t.ty, &src, ne0, e0 * ne1, ne1) {
                Ok(o) => o,
                Err(x) => {
                    err = Some(Error::refused(format!(
                        "{} experts {e0}..{e1}: {x}",
                        t.name
                    )));
                    return None;
                }
            };
            let ob = ne0 / 256 * kind.block_bytes();
            let (r0, r1) = (e0 * ne1, e1 * ne1);
            for &r in picks.iter().filter(|&&r| (r0..r1).contains(&r)) {
                let k = r - r0;
                let want = host_row(
                    kind,
                    tables,
                    t.ty,
                    &src[k * rb..(k + 1) * rb],
                    &qw[(r / ne1) * ne0..(r / ne1 + 1) * ne0],
                );
                let r = want.and_then(|want| {
                    compare_row(&t.name, kind, r, ne1, &out[k * ob..(k + 1) * ob], &want)
                });
                if let Err(e) = r {
                    err = Some(e);
                    return None;
                }
            }
            e0 = e1;
            Some(out)
        });
        w.tensor_parts(&t.name, parts)
    });
    if let Some(e) = err {
        return Err(e);
    }
    wrote
        .map(|_| ())
        .map_err(|e| Error::refused(format!("{}: {e}", t.name)))
}
