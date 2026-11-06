//! The K3.1 pipeline's intermediate set (`tools/ref/k31-quant.sh --gpu-quant`):
//! every shard of the input set written again under `--out`'s prefix with the
//! same metadata and the same tensors in the same order, where each tensor
//! llama-quantize's own plan (`--plan`, the plan step's TSV:
//! `name TAB TYPE TAB dims TAB bytes`) types IQ1_S or IQ2_XXS is quantized on
//! the card (`bloomery_gpu::iq_quant`, ggml's bytes: `just gate-gpu-iq-quant`)
//! with the importance llama-quantize computes from `--imatrix`, and every
//! other tensor is copied byte for byte. llama-quantize run over the result
//! with the same rules copies the pre-quantized tensors (`cur_type ==
//! new_type`) and quantizes the rest.
//!
//!   iq_quant_set <first shard> --imatrix FILE --plan TSV --out PREFIX [--batch N]
//!
//! Shard `i` of `n` goes to `PREFIX-0000i-of-0000n.gguf` (a set of one file
//! without `split.count` to `PREFIX.gguf`), written as
//! `<name>.partial` and renamed when whole; an output that exists is refused,
//! not overwritten. The next batch of experts is read while the card
//! quantizes the current one. Exit: 0 every shard written; 1 a refusal,
//! named (a tensor the plan does not name or names with other dims, a plan
//! row the set does not hold, a source type the card does not read, an
//! imatrix entry missing or of another length).

#[cfg(feature = "gpu")]
use bloomery_gpu::Gpu;
#[cfg(feature = "gpu")]
use bloomery_gpu::iq_quant::IqQuant;
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::GateError;
#[cfg(feature = "gpu")]
use cuda_core::DeviceBuffer;
#[cfg(feature = "gpu")]
use gguf::iq_quant::{IqKind, imatrix_weights};
#[cfg(feature = "gpu")]
use gguf::quant::GgmlType;
#[cfg(feature = "gpu")]
use gguf::write::{Layout, TensorDecl, Writer};
#[cfg(feature = "gpu")]
use std::collections::HashMap;
#[cfg(feature = "gpu")]
use std::path::PathBuf;
#[cfg(feature = "gpu")]
use std::time::Instant;

fn main() -> std::process::ExitCode {
    #[cfg(feature = "gpu")]
    let r = run();
    #[cfg(not(feature = "gpu"))]
    let r: Result<(), Box<dyn std::error::Error>> =
        Err("iq_quant_set needs the `gpu` feature".into());
    bloomery_gpu_gates::exit_with("iq_quant_set", r)
}

#[cfg(feature = "gpu")]
struct Args {
    first: PathBuf,
    imatrix: PathBuf,
    plan: PathBuf,
    out: String,
    batch: usize,
}

#[cfg(feature = "gpu")]
fn args() -> Result<Args, GateError> {
    let mut it = std::env::args().skip(1);
    let usage =
        "usage: iq_quant_set <first shard> --imatrix FILE --plan TSV --out PREFIX [--batch N]";
    let first = PathBuf::from(it.next().ok_or(usage)?);
    let (mut imatrix, mut plan, mut out, mut batch) = (None, None, None, 32usize);
    while let Some(a) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{a} needs a value"));
        match a.as_str() {
            "--imatrix" => imatrix = Some(PathBuf::from(v()?)),
            "--plan" => plan = Some(PathBuf::from(v()?)),
            "--out" => out = Some(v()?),
            "--batch" => batch = v()?.parse()?,
            o => return Err(format!("unknown argument {o}; {usage}").into()),
        }
    }
    Ok(Args {
        first,
        imatrix: imatrix.ok_or("--imatrix is required")?,
        plan: plan.ok_or("--plan is required")?,
        out: out.ok_or("--out is required")?,
        batch: batch.max(1),
    })
}

/// The plan's rows by name: the type llama-quantize gives the tensor and its
/// dims as the plan spells them (`x`-joined, trailing 1s dropped).
#[cfg(feature = "gpu")]
fn read_plan(path: &PathBuf) -> Result<HashMap<String, (String, String)>, GateError> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut rows = HashMap::new();
    for (i, line) in text.lines().enumerate() {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 4 {
            return Err(format!(
                "{} line {}: not name TAB type TAB dims TAB bytes",
                path.display(),
                i + 1
            )
            .into());
        }
        if rows
            .insert(f[0].to_string(), (f[1].to_uppercase(), f[2].to_string()))
            .is_some()
        {
            return Err(format!("{}: {} is planned twice", path.display(), f[0]).into());
        }
    }
    Ok(rows)
}

/// `dims` as the plan spells them.
#[cfg(feature = "gpu")]
fn plan_dims(dims: &[u64]) -> String {
    let mut d: Vec<String> = dims.iter().map(u64::to_string).collect();
    while d.len() > 1 && d.last().is_some_and(|x| x == "1") {
        d.pop();
    }
    d.join("x")
}

/// Tensor `t`'s importance, `ne0` per expert, as llama-quantize computes it
/// from the imatrix; an entry missing or of another length is refused.
#[cfg(feature = "gpu")]
fn importance(im: &gguf::Gguf, t: &gguf::TensorInfo) -> Result<Vec<f32>, GateError> {
    let qw = imatrix_weights(im, &t.name)?;
    let (ne0, n_exp) = (t.dims[0], t.dims[2..].iter().product::<u64>());
    if qw.len() as u64 != ne0 * n_exp {
        return Err(format!(
            "imatrix {}: {} values, the tensor needs {} ({ne0} x {n_exp} experts)",
            t.name,
            qw.len(),
            ne0 * n_exp
        )
        .into());
    }
    Ok(qw)
}

/// What becomes of one tensor.
#[cfg(feature = "gpu")]
enum Fate {
    Copy,
    Quantize(IqKind),
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), GateError> {
    let a = args()?;
    let set = gguf::Split::open(&a.first)?;
    let plan = read_plan(&a.plan)?;
    let im = gguf::Gguf::open(&a.imatrix)?;
    let n = set.shard_count();
    if plan.len() != set.tensor_count() {
        let missing: Vec<&String> = plan
            .keys()
            .filter(|k| set.find(k).is_none())
            .take(3)
            .collect();
        return Err(format!(
            "the plan names {} tensors, the set holds {}; plan rows the set does not hold: {missing:?}",
            plan.len(),
            set.tensor_count()
        )
        .into());
    }
    // Every decision first: a refusal costs no write.
    let mut fates: Vec<Vec<Fate>> = Vec::with_capacity(n);
    let mut layouts = Vec::with_capacity(n);
    let (mut q_tensors, mut q_weights) = (0usize, 0f64);
    for i in 0..n {
        let shard = set.shard(i).expect("i < shard_count");
        let mut fate = Vec::new();
        let mut decls = Vec::new();
        for t in shard.iter_tensors() {
            let (ty, dims) = plan
                .get(&t.name)
                .ok_or_else(|| format!("the plan has no row for {}", t.name))?;
            if *dims != plan_dims(&t.dims) {
                return Err(format!(
                    "{}: the plan's dims {dims}, the set's {}",
                    t.name,
                    plan_dims(&t.dims)
                )
                .into());
            }
            let kind = match ty.as_str() {
                "IQ1_S" => Some(IqKind::Iq1S),
                "IQ2_XXS" => Some(IqKind::Iq2Xxs),
                _ => None,
            };
            let target = kind.map(|k| match k {
                IqKind::Iq1S => GgmlType::IQ1_S,
                IqKind::Iq2Xxs => GgmlType::IQ2_XXS,
            });
            match kind {
                Some(k) if target != Some(t.ty) => {
                    if !matches!(
                        t.ty,
                        GgmlType::MXFP4 | GgmlType::BF16 | GgmlType::F16 | GgmlType::F32
                    ) {
                        return Err(format!(
                            "{} is {:?}: the card reads F32, F16, BF16 or MXFP4",
                            t.name, t.ty
                        )
                        .into());
                    }
                    let ne0 = t.dims[0] as usize;
                    if !ne0.is_multiple_of(256) {
                        return Err(
                            format!("{}: ne0 {ne0} is not a multiple of 256", t.name).into()
                        );
                    }
                    importance(&im, t)?;
                    let rows: u64 = t.dims[1..].iter().product();
                    decls.push(TensorDecl {
                        name: t.name.clone(),
                        dims: t.dims.clone(),
                        type_id: target.expect("kind is Some").as_u32(),
                        nbytes: rows * (ne0 / 256 * k.block_bytes()) as u64,
                    });
                    q_tensors += 1;
                    q_weights += (ne0 as u64 * rows) as f64;
                    fate.push(Fate::Quantize(k));
                }
                _ => {
                    decls.push(TensorDecl {
                        name: t.name.clone(),
                        dims: t.dims.clone(),
                        type_id: t.ty.as_u32(),
                        nbytes: t.nbytes,
                    });
                    fate.push(Fate::Copy);
                }
            }
        }
        layouts.push(Layout::new(shard.keys(), decls)?);
        fates.push(fate);
    }
    let split_count = set.shard(0).and_then(|g| g.value("split.count"));
    let outs: Vec<PathBuf> = match split_count {
        None if n == 1 => vec![PathBuf::from(format!("{}.gguf", a.out))],
        Some(c) if c.as_unsigned() == Some(n as u64) => (0..n)
            .map(|i| PathBuf::from(format!("{}-{:05}-of-{:05}.gguf", a.out, i + 1, n)))
            .collect(),
        c => {
            return Err(format!(
                "the set opens as {n} files and its split.count is {c:?}: a companion file is not written here"
            )
            .into());
        }
    };
    if let Some(o) = outs.iter().find(|o| o.exists()) {
        return Err(format!("{} exists: not overwritten", o.display()).into());
    }
    let total: u64 = layouts.iter().map(Layout::file_len).sum();
    println!(
        "iq_quant_set: {n} shard(s) from {}; {q_tensors} tensors ({q_weights:.3e} weights) quantized on the card, \
         the rest copied; {:.2} GB to write under {}",
        a.first.display(),
        total as f64 / 1e9,
        a.out
    );
    let gpu = Gpu::new()?;
    let stream = gpu.stream();
    let mut quant: [Option<IqQuant>; 2] = [None, None];
    let t_all = Instant::now();
    let (mut t_q, mut w_q) = (0.0f64, 0.0f64);
    for (i, (layout, fate)) in layouts.into_iter().zip(&fates).enumerate() {
        let shard = set.shard(i).expect("i < shard_count");
        let partial = PathBuf::from(format!("{}.partial", outs[i].display()));
        let file =
            std::fs::File::create(&partial).map_err(|e| format!("{}: {e}", partial.display()))?;
        let mut w = Writer::new(std::io::BufWriter::with_capacity(1 << 24, file), layout)?;
        for (t, f) in shard.iter_tensors().zip(fate) {
            let data = shard.data(t)?;
            let Fate::Quantize(kind) = f else {
                w.tensor(&t.name, data)?;
                continue;
            };
            let slot = &mut quant[usize::from(*kind == IqKind::Iq1S)];
            if slot.is_none() {
                *slot = Some(IqQuant::new(gpu.context(), stream, *kind)?);
            }
            let q = slot.as_ref().expect("filled above");
            let (ne0, ne1) = (t.dims[0] as usize, t.dims[1] as usize);
            let n_exp: usize = t.dims[2..].iter().product::<u64>() as usize;
            let qw = importance(&im, t)?;
            let qw_d = DeviceBuffer::from_host(stream, &qw)?;
            let eb = data.len() / n_exp;
            let t0 = Instant::now();
            let mut err: Option<GateError> = None;
            let wrote = std::thread::scope(|sc| {
                let read = |e0: usize, e1: usize| data[e0 * eb..e1 * eb].to_vec();
                let batch = a.batch;
                let mut pending = Some(sc.spawn(move || read(0, batch.min(n_exp))));
                let mut e0 = 0;
                let parts = std::iter::from_fn(|| {
                    let Ok(src) = pending.take()?.join() else {
                        err = Some("the reader thread panicked".into());
                        return None;
                    };
                    let e1 = (e0 + batch).min(n_exp);
                    if e1 < n_exp {
                        let e2 = (e1 + batch).min(n_exp);
                        pending = Some(sc.spawn(move || read(e1, e2)));
                    }
                    match q.quantize(stream, t.ty, &src, ne0, e0 * ne1, ne1, &qw_d) {
                        Ok(out) => {
                            e0 = e1;
                            Some(out)
                        }
                        Err(x) => {
                            err = Some(format!("{} experts {e0}..{e1}: {x}", t.name).into());
                            None
                        }
                    }
                });
                w.tensor_parts(&t.name, parts)
            });
            if let Some(e) = err {
                return Err(e);
            }
            wrote?;
            let s = t0.elapsed().as_secs_f64();
            let wt = (ne0 * ne1 * n_exp) as f64;
            (t_q, w_q) = (t_q + s, w_q + wt);
            println!(
                "  {} {:?} -> {kind:?}: {wt:.3e} weights in {s:.1} s (read overlapped) = {:.3e} weights/s",
                t.name,
                t.ty,
                wt / s
            );
        }
        w.finish()?
            .into_inner()
            .map_err(|e| e.into_error())?
            .sync_all()?;
        std::fs::rename(&partial, &outs[i])?;
        println!("  shard {}/{n}: {}", i + 1, outs[i].display());
    }
    println!(
        "iq_quant_set: done in {:.0} s; quantized tensors {:.0} s at {:.3e} weights/s",
        t_all.elapsed().as_secs_f64(),
        t_q,
        if t_q > 0.0 { w_q / t_q } else { 0.0 }
    );
    Ok(())
}
