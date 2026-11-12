//! The set writer: every shard of the input written again under the output prefix with the same metadata and
//! the same tensors in the same order, each tensor the plan says to quantize quantized by a [`Backend`] and every
//! other tensor copied byte for byte. Each shard is written as `<name>.partial` and renamed when whole; a failed
//! run removes every file it made, or under `--resume` only the `.partial` it was writing, keeping every whole
//! shard for the next run of the same plan. The next batch of experts is read and scanned while the backend works
//! on the current one.

use crate::check::{check_rows, compare_row, host_row};
use crate::cli::{Args, Input};
use crate::host::convert_rows;
use crate::imatrix::Imatrix;
use crate::plan::{Fate, LlamaPlan, Plan, Target};
use crate::rules::Rules;
use crate::scan::{row_bytes, scan_rows};
use crate::source::{Source, SplitSource};
use crate::{Error, Result};
use gguf::iq_quant::{GridTables, IqKind};
use gguf::quant::GgmlType;
use gguf::write::Writer;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Instant;

/// A backend's error, whatever it is made of.
pub type BackendError = Box<dyn std::error::Error>;

/// What quantizes a batch of IQ rows: the card (the binary supplies it) or the host rule ([`crate::host`]).
pub trait Backend {
    /// What the run prints it as.
    fn name(&self) -> String;
    /// Whether the backend's work runs on the host's resident pool, so the host rules of the other tensors cannot
    /// run beside it.
    fn uses_host_pool(&self) -> bool;
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
    pub set: Box<dyn Source + Send>,
    pub im: Option<Imatrix>,
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

/// [`prepare_from`] over the GGUF split the command line names; a checkpoint folder is opened by the caller that
/// links its reader (`model_hf::quantize_source`) and handed to [`prepare_from`], and is refused by name here.
pub fn prepare(a: &Args) -> Result<Prepared> {
    let Input::First(first) = &a.input else {
        return Err(Error::refused(format!(
            "{}: this build reads no checkpoint folder (--hf needs the `hf` feature)",
            a.input.path().display()
        )));
    };
    exists(first)?;
    let set = gguf::Split::open(first)
        .map_err(|e| Error::refused(format!("{}: {e}", first.display())))?;
    prepare_from(a, Box::new(SplitSource(set)))
}

/// Open the imatrix, the rules and the optional llama plan, and decide every tensor of `set`; reads no tensor data
/// but the imatrix's.
pub fn prepare_from(a: &Args, set: Box<dyn Source + Send>) -> Result<Prepared> {
    let mut inputs: Vec<&Path> = Vec::new();
    inputs.extend(a.rules.as_deref());
    inputs.extend(a.llama_plan.as_deref());
    inputs.extend(a.imatrix.as_deref().map(Path::new));
    for p in inputs {
        exists(p)?;
    }
    let text = |p: &Path| std::fs::read_to_string(p).map_err(|e| io(p, e));
    let rules = match &a.rules {
        Some(p) => {
            Rules::parse(&text(p)?).map_err(|e| Error::refused(format!("{}: {e}", p.display())))?
        }
        None => Rules::parse("")?,
    };
    let lcpp = a
        .llama_plan
        .as_ref()
        .map(|p| LlamaPlan::parse(&text(p)?, &p.display().to_string()))
        .transpose()?;
    let im = a.imatrix.as_deref().map(Imatrix::open).transpose()?;
    let plan = Plan::build(&*set, &rules, lcpp.as_ref(), im.as_ref(), a)?;
    Ok(Prepared { set, im, plan })
}

/// The room check: the output's drive has the bytes the output files need, or the run is refused with both numbers.
/// Under `--resume` the shards already on disk whole ([`kept_whole`]) are not needed again.
pub fn check_room(a: &Args, plan: &Plan) -> Result<()> {
    let kept_bytes = if a.resume {
        kept_whole(plan)?
            .iter()
            .zip(&plan.layouts)
            .filter(|(k, _)| **k)
            .map(|(_, l)| l.file_len())
            .sum()
    } else {
        0
    };
    let dir = a.out_dir();
    let free =
        os::free_bytes(&dir).map_err(|e| Error::refused(format!("{}: {e}", dir.display())))?;
    if free + kept_bytes < plan.out_file_bytes {
        let note = if kept_bytes > 0 {
            format!(
                " ({} B of the set already on disk as whole shards)",
                kept_bytes
            )
        } else {
            String::new()
        };
        return Err(Error::refused(format!(
            "the output set needs {} B under {}{note}, the drive has {free} B free",
            plan.out_file_bytes - kept_bytes,
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

/// Under `--resume`, whether each planned output shard is already on disk whole: a file whose length
/// is the plan's and whose first bytes are the header this plan's writer would write — the metadata
/// pairs and every tensor's name, dims, type and offset — compared without reading any tensor data.
/// A whole file that differs in any of that refuses the run naming the path and the first field that
/// differs. Reads nothing and removes nothing, so a refusal leaves every file as it was.
fn kept_whole(plan: &Plan) -> Result<Vec<bool>> {
    let mut keep = vec![false; plan.outs.len()];
    for (s, (out, layout)) in plan.outs.iter().zip(&plan.layouts).enumerate() {
        let Ok(md) = std::fs::metadata(out) else {
            continue;
        };
        let want = layout.file_len();
        if md.len() != want {
            return Err(Error::refused(format!(
                "{}: {} B on disk, this plan's shard is {want} B: a whole file that is not this plan's \
                 shard is not kept",
                out.display(),
                md.len()
            )));
        }
        let mut head = vec![0u8; layout.header_bytes().len()];
        let mut f = std::fs::File::open(out).map_err(|e| io(out, e))?;
        f.read_exact(&mut head).map_err(|e| io(out, e))?;
        if let Some(at) = head
            .iter()
            .zip(layout.header_bytes())
            .position(|(a, b)| a != b)
        {
            let field = header_field(layout.header_bytes(), at);
            return Err(Error::refused(format!(
                "{}: header byte {at} ({field}) differs from this plan's: a whole file that is not \
                 this plan's shard is not kept",
                out.display()
            )));
        }
        keep[s] = true;
    }
    Ok(keep)
}

/// [`kept_whole`] acted on for a run: after every whole shard has been accepted, each remaining
/// stale `.partial` is removed and the log names it.
fn resume_states(plan: &Plan, log: &mut dyn FnMut(&str)) -> Result<Vec<bool>> {
    let keep = kept_whole(plan)?;
    for (out, k) in plan.outs.iter().zip(&keep) {
        if *k {
            continue;
        }
        let partial = PathBuf::from(format!("{}.partial", out.display()));
        if partial.exists() {
            std::fs::remove_file(&partial).map_err(|e| io(&partial, e))?;
            log(&format!("  removed stale {}", partial.display()));
        }
    }
    Ok(keep)
}

/// A little-endian `u32` at the head of `b`, or `None` when it is shorter.
fn le32(b: &[u8]) -> Option<u32> {
    Some(u32::from_le_bytes(*b.first_chunk()?))
}

/// A little-endian `u64` at the head of `b`, or `None` when it is shorter.
fn le64(b: &[u8]) -> Option<u64> {
    Some(u64::from_le_bytes(*b.first_chunk()?))
}

/// The `n` bytes at `*o`, as their span; `o` moves past them.
fn span(b: &[u8], o: &mut usize, n: usize) -> Option<std::ops::Range<usize>> {
    let r = *o..*o + n;
    b.get(r.clone())?;
    *o = r.end;
    Some(r)
}

/// A GGUF string at `*o` — its `u64` length then its bytes — as text; `o` moves past it.
fn header_text(b: &[u8], o: &mut usize) -> Option<String> {
    let n = le64(b.get(span(b, o, 8)?)?)? as usize;
    Some(String::from_utf8_lossy(b.get(span(b, o, n)?)?).into_owned())
}

/// A GGUF value of tag `t` at `*o`, its tag already consumed; `o` moves past it. A string is its
/// length and bytes, an array its element tag, its count and its elements, as the writer lays them out.
fn header_value(b: &[u8], o: &mut usize, t: u32) -> Option<()> {
    let scalar = match t {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        8 => {
            header_text(b, o)?;
            return Some(());
        }
        9 => {
            let elem = le32(b.get(span(b, o, 4)?)?)?;
            let n = le64(b.get(span(b, o, 8)?)?)? as usize;
            for _ in 0..n {
                header_value(b, o, elem)?;
            }
            return Some(());
        }
        _ => return None,
    };
    span(b, o, scalar)?;
    Some(())
}

/// The field of a GGUF header byte `at` lands in — the counts, a metadata pair's key or value, a
/// tensor's name, dims, type or offset, the zero padding — named by walking `header`, the bytes
/// this plan's layout would write, so a kept-shard refusal names the first field that differs.
fn header_field(header: &[u8], at: usize) -> String {
    let field = (|| -> Option<String> {
        let mut o = 24; // the magic, the version, the tensor and metadata counts
        if at < o {
            return Some("the header's counts".to_string());
        }
        let n_tensors = le64(header.get(8..16)?)? as usize;
        let n_kv = le64(header.get(16..24)?)? as usize;
        for _ in 0..n_kv {
            let kstart = o;
            let key = header_text(header, &mut o)?;
            let key_end = o;
            let vstart = o;
            let t = le32(header.get(span(header, &mut o, 4)?)?)?;
            header_value(header, &mut o, t)?;
            if (kstart..key_end).contains(&at) {
                return Some(format!("metadata key {key:?}"));
            }
            if (vstart..o).contains(&at) {
                return Some(format!("metadata {key:?} value"));
            }
        }
        for _ in 0..n_tensors {
            let nstart = o;
            let name = header_text(header, &mut o)?;
            let name_end = o;
            let ddstart = o;
            let dims = le32(header.get(span(header, &mut o, 4)?)?)? as usize;
            span(header, &mut o, dims * 8)?;
            let ddend = o;
            let ty = span(header, &mut o, 4)?;
            let off = span(header, &mut o, 8)?;
            if (nstart..name_end).contains(&at) {
                return Some(format!("tensor {name:?} name"));
            }
            if (ddstart..ddend).contains(&at) {
                return Some(format!("tensor {name:?} dims"));
            }
            if ty.contains(&at) {
                return Some(format!("tensor {name:?} type"));
            }
            if off.contains(&at) {
                return Some(format!("tensor {name:?} offset"));
            }
        }
        Some("the header's zero padding".to_string())
    })();
    field.unwrap_or_else(|| "the header".to_string())
}

/// What a run did.
#[derive(Debug)]
pub struct Summary {
    /// Tensors converted, and the weights of the IQ ones with the seconds they took (the card's or the host's).
    pub quantized: usize,
    pub weights: f64,
    pub quant_s: f64,
    pub wall_s: f64,
    /// The shards this run wrote; under `--resume` the kept ones are not among them.
    pub files: Vec<PathBuf>,
    /// The shards `--resume` kept whole from an earlier run of this plan.
    pub kept: usize,
}

/// Write the output. `log` gets each progress line. Under `--resume` a whole shard of this same
/// plan is kept and every shard is classified before any tensor is read; a failure then removes
/// only the `.partial` this run was writing, so the next `--resume` goes on. Without it any
/// existing output is refused, and a failure removes the files this run made.
pub fn run(
    a: &Args,
    prep: Prepared,
    backend: &mut dyn Backend,
    log: &mut dyn FnMut(&str),
) -> Result<Summary> {
    let keep = if a.resume {
        check_room(a, &prep.plan)?;
        resume_states(&prep.plan, log)?
    } else {
        check_absent(&prep.plan)?;
        check_room(a, &prep.plan)?;
        vec![false; prep.plan.shards.len()]
    };
    let mut made = Vec::new();
    let r = write_all(a, prep, backend, log, &keep, &mut made);
    if r.is_err() {
        for f in &made {
            let _ = std::fs::remove_file(f);
        }
    }
    r
}

/// The host-rule conversion of one tensor, whole: the importance, the finite scan, the rules, and the size check.
fn host_convert(
    set: &dyn Source,
    im: Option<&Imatrix>,
    item: &crate::plan::Item,
    target: Target,
) -> Result<Vec<u8>> {
    let data = set.read(&item.name, 0..item.bytes_in)?;
    let (ne0, ne1) = (item.dims[0] as usize, item.dims[1] as usize);
    let qw = match (item.importance, im) {
        (true, Some(m)) => Some(m.weights(&item.name)?),
        _ => None,
    };
    scan_rows(&item.name, item.src, &data, ne0, 0, ne1)?;
    let out = convert_rows(target, item.src, &data, ne0, 0, ne1, qw.as_deref())
        .map_err(|e| Error::refused(format!("{}: {e}", item.name)))?;
    if out.len() as u64 != item.bytes_out {
        return Err(Error::refused(format!(
            "{}: {} bytes converted, the plan says {}",
            item.name,
            out.len(),
            item.bytes_out
        )));
    }
    Ok(out)
}

/// The host-converted tensors in output order, each converted ahead of the writer on the host's pool while the card
/// works on the IQ stacks: a bounded queue (a few tensors, a few hundred MB) of results, in the order the writer
/// asks for them.
const LOOKAHEAD: usize = 3;

/// Seconds as `h:mm:ss`.
fn hour_clock(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    format!("{}:{:02}:{:02}", s / 3600, s % 3600 / 60, s % 60)
}

/// Seconds as `m:ss` under an hour, `h:mm:ss` above it.
fn short_clock(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    if s >= 3600 {
        hour_clock(secs)
    } else {
        format!("{}:{:02}", s / 60, s % 60)
    }
}

fn write_all(
    a: &Args,
    prep: Prepared,
    backend: &mut dyn Backend,
    log: &mut dyn FnMut(&str),
    keep: &[bool],
    made: &mut Vec<PathBuf>,
) -> Result<Summary> {
    let Prepared { set, im, plan } = prep;
    let Plan {
        items,
        shards,
        layouts,
        outs,
        out_file_bytes,
        ..
    } = plan;
    let n = shards.len();
    debug_assert_eq!(keep.len(), n, "run passes one state a shard");
    let n_conv = items
        .iter()
        .filter(|i| matches!(i.fate, Fate::Convert(_)))
        .count();
    log(&format!(
        "bloomery-quantize: {n} output shard(s) from {}; {n_conv} of {} tensors converted (IQ by the {}, the rest by the host rules), the rest copied; {:.2} GB to write to {}",
        set.label(),
        items.len(),
        backend.name(),
        out_file_bytes as f64 / 1e9,
        a.out
    ));
    let t_all = Instant::now();
    let (mut t_q, mut w_q, mut n_q) = (0.0f64, 0.0f64, 0usize);
    let mut tables: [Option<GridTables>; 2] = [None, None];
    let mut files = Vec::new();
    let ahead = !backend.uses_host_pool();
    // A kept shard's items leave the host-ahead list with it: nothing of a kept shard is read,
    // converted or waited for, and the results stay in the order the writer asks for them.
    let host_items: Vec<usize> = shards
        .iter()
        .zip(keep)
        .filter(|(_, k)| !**k)
        .flat_map(|(sh, _)| sh.iter().copied())
        .filter(|&i| matches!(items[i].fate, Fate::Convert(Target::K(_) | Target::F32)))
        .collect();
    let (tx, rx) = mpsc::sync_channel::<(usize, Result<Vec<u8>>)>(LOOKAHEAD);
    let (items_ref, set_ref, im_ref) = (&items, &*set, im.as_ref());
    let mut err_out: Option<Error> = None;
    let (mut bytes_done, mut written_bytes, mut n_kept) = (0u64, 0u64, 0usize);
    std::thread::scope(|sc| {
        if ahead && !host_items.is_empty() {
            sc.spawn(move || {
                for i in host_items {
                    let Fate::Convert(t) = items_ref[i].fate else {
                        unreachable!("host_items holds conversions");
                    };
                    let r = host_convert(set_ref, im_ref, &items_ref[i], t);
                    let bad = r.is_err();
                    if tx.send((i, r)).is_err() || bad {
                        return;
                    }
                }
            });
        } else {
            drop(tx);
        }
        let r = (|| -> Result<()> {
            for (s, layout) in layouts.into_iter().enumerate() {
                let out = outs[s].clone();
                let shard_len = layout.file_len();
                if keep[s] {
                    bytes_done += shard_len;
                    n_kept += 1;
                    log(&format!("  shard {}/{n}: kept {}", s + 1, out.display()));
                    continue;
                }
                let partial = PathBuf::from(format!("{}.partial", out.display()));
                made.push(partial.clone());
                let file = std::fs::File::create(&partial).map_err(|e| io(&partial, e))?;
                let wr = |e: gguf::write::WriteError| {
                    Error::refused(format!("{}: {e}", partial.display()))
                };
                let mut w = Writer::new(std::io::BufWriter::with_capacity(1 << 24, file), layout)
                    .map_err(wr)?;
                for &i in &shards[s] {
                    let item = &items[i];
                    let name = &item.name;
                    match item.fate {
                        Fate::Copy => {
                            let data = set.read(name, 0..item.bytes_in)?;
                            w.tensor(name, &data).map_err(wr)?;
                        }
                        Fate::Convert(Target::Iq(kind)) => {
                            let qw = im
                                .as_ref()
                                .ok_or_else(|| Error::refused("an IQ target needs the imatrix"))?
                                .weights(name)?;
                            backend
                                .bind(kind, &qw)
                                .map_err(|e| Error::refused(format!("{name}: {e}")))?;
                            let (ne0, ne1) = (item.dims[0] as usize, item.dims[1] as usize);
                            let n_exp: usize = item.dims[2..].iter().product::<u64>() as usize;
                            let tab = &*tables[usize::from(kind == IqKind::Iq1S)]
                                .get_or_insert_with(|| GridTables::new(kind));
                            let t0 = Instant::now();
                            quantize_tensor(
                                a,
                                backend,
                                &mut w,
                                &*set,
                                item,
                                kind,
                                tab,
                                &qw,
                                (ne0, ne1, n_exp),
                            )?;
                            let secs = t0.elapsed().as_secs_f64();
                            let wt = (ne0 * ne1 * n_exp) as f64;
                            (t_q, w_q, n_q) = (t_q + secs, w_q + wt, n_q + 1);
                            log(&format!(
                                "  {name} {:?} -> {kind:?}: {wt:.3e} weights in {secs:.1} s (read overlapped) = {:.3e} weights/s",
                                item.src,
                                wt / secs
                            ));
                        }
                        Fate::Convert(target) => {
                            let t0 = Instant::now();
                            let bytes = if ahead {
                                let (got, r) = rx.recv().map_err(|_| {
                                    Error::refused("the host conversion thread ended early")
                                })?;
                                debug_assert_eq!(got, i);
                                r?
                            } else {
                                host_convert(&*set, im.as_ref(), item, target)?
                            };
                            w.tensor(name, &bytes).map_err(wr)?;
                            n_q += 1;
                            log(&format!(
                                "  {name} {:?} -> {:?}: {} B on the host ({:.1} s waited for)",
                                item.src,
                                item.dst,
                                bytes.len(),
                                t0.elapsed().as_secs_f64()
                            ));
                        }
                    }
                }
                w.finish()
                    .map_err(wr)?
                    .into_inner()
                    .map_err(|e| io(&partial, e.into_error()))?
                    .sync_all()
                    .map_err(|e| io(&partial, e))?;
                std::fs::rename(&partial, &out).map_err(|e| io(&out, e))?;
                made.pop();
                if !a.resume {
                    // a failure without --resume removes what the run made, whole shards included
                    made.push(out.clone());
                }
                bytes_done += shard_len;
                written_bytes += shard_len;
                // the time left runs at the mean rate of the shards this run wrote, so a kept
                // shard's bytes count as done and not as rate
                let elapsed = t_all.elapsed().as_secs_f64();
                let left = if written_bytes > 0 && elapsed > 0.0 {
                    format!(
                        "~{}",
                        short_clock(
                            (out_file_bytes - bytes_done) as f64 / (written_bytes as f64 / elapsed)
                        )
                    )
                } else {
                    "?".to_string()
                };
                log(&format!(
                    "  shard {}/{n}: {} ({:.1} of {:.1} GB, {} elapsed, {left} left)",
                    s + 1,
                    out.display(),
                    bytes_done as f64 / 1e9,
                    out_file_bytes as f64 / 1e9,
                    hour_clock(elapsed)
                ));
                files.push(out);
            }
            Ok(())
        })();
        drop(rx);
        err_out = r.err();
    });
    if let Some(e) = err_out {
        return Err(e);
    }
    log(&format!(
        "bloomery-quantize: {} shard(s) written, {n_kept} kept",
        files.len()
    ));
    Ok(Summary {
        quantized: n_q,
        weights: w_q,
        quant_s: t_q,
        wall_s: t_all.elapsed().as_secs_f64(),
        files,
        kept: n_kept,
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
    set: &dyn Source,
    t: &crate::plan::Item,
    kind: IqKind,
    tables: &GridTables,
    qw: &[f32],
    (ne0, ne1, n_exp): (usize, usize, usize),
) -> Result<()> {
    let rb = row_bytes(t.src, ne0).expect("the plan checked the row geometry");
    let eb = rb * ne1;
    debug_assert_eq!(eb * n_exp, t.bytes_in as usize);
    let picks = check_rows(ne1 * n_exp, a.check);
    let batch = a.batch;
    let mut err: Option<Error> = None;
    let wrote = std::thread::scope(|sc| {
        let read = |e0: usize, e1: usize| -> Result<Vec<u8>> {
            let src = set
                .read(&t.name, (e0 * eb) as u64..(e1 * eb) as u64)?
                .into_owned();
            scan_rows(&t.name, t.src, &src, ne0, e0 * ne1, ne1)?;
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
            let out = match backend.quantize(t.src, &src, ne0, e0 * ne1, ne1) {
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
                    t.src,
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
