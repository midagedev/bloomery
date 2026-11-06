//! One tensor of a GGUF set quantized to IQ1_S or IQ2_XXS on the card
//! (`bloomery_gpu::iq_quant`), whole, in batches of experts, with each
//! batch's read and card time; then the bytes of the experts `--check` names
//! against ggml's (`iq_quant_ref`, `just build-iq-quant-ref`). Reads the set,
//! writes only the oracle's inputs and outputs under `--scratch`.
//!
//!   iq_quant_probe <first shard> --tensor NAME --type iq1_s|iq2_xxs
//!       [--imatrix FILE] [--batch N] [--check E,E,...] [--scratch DIR]
//!
//! Without `--imatrix` the importance is a fixed synthetic vector (seeded,
//! positive), named so in the output: the bytes still must equal ggml's on
//! the same input. Exit: 0 every checked expert equal; 1 a difference or a
//! refusal, named.

#[cfg(feature = "gpu")]
use bloomery_gpu::Gpu;
#[cfg(feature = "gpu")]
use bloomery_gpu::iq_quant::IqQuant;
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::{GateError, data_dir};
#[cfg(feature = "gpu")]
use cuda_core::DeviceBuffer;
#[cfg(feature = "gpu")]
use gguf::iq_quant::{IqKind, imatrix_weights};
#[cfg(feature = "gpu")]
use gguf::quant::GgmlType;
#[cfg(feature = "gpu")]
use std::path::PathBuf;
#[cfg(feature = "gpu")]
use std::time::Instant;

fn main() -> std::process::ExitCode {
    #[cfg(feature = "gpu")]
    let r = run();
    #[cfg(not(feature = "gpu"))]
    let r: Result<(), Box<dyn std::error::Error>> =
        Err("iq_quant_probe needs the `gpu` feature".into());
    bloomery_gpu_gates::exit_with("iq_quant_probe", r)
}

#[cfg(feature = "gpu")]
struct Args {
    first: PathBuf,
    tensor: String,
    kind: IqKind,
    imatrix: Option<PathBuf>,
    batch: usize,
    check: Vec<usize>,
    scratch: PathBuf,
}

#[cfg(feature = "gpu")]
fn args() -> Result<Args, GateError> {
    let mut it = std::env::args().skip(1);
    let first = PathBuf::from(
        it.next()
            .ok_or("usage: iq_quant_probe <first shard> --tensor NAME --type T")?,
    );
    let (mut tensor, mut kind, mut imatrix) = (None, None, None);
    let (mut batch, mut check, mut scratch) =
        (32, vec![0], PathBuf::from("/home/user/gpuquant-scratch"));
    while let Some(a) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{a} needs a value"));
        match a.as_str() {
            "--tensor" => tensor = Some(v()?),
            "--type" => {
                kind = Some(match v()?.as_str() {
                    "iq1_s" => IqKind::Iq1S,
                    "iq2_xxs" => IqKind::Iq2Xxs,
                    o => return Err(format!("--type {o}: iq1_s or iq2_xxs").into()),
                })
            }
            "--imatrix" => imatrix = Some(PathBuf::from(v()?)),
            "--batch" => batch = v()?.parse()?,
            "--check" => check = v()?.split(',').map(str::parse).collect::<Result<_, _>>()?,
            "--scratch" => scratch = PathBuf::from(v()?),
            o => return Err(format!("unknown argument {o}").into()),
        }
    }
    Ok(Args {
        first,
        tensor: tensor.ok_or("--tensor is required")?,
        kind: kind.ok_or("--type is required")?,
        imatrix,
        batch: batch.max(1),
        check,
        scratch,
    })
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), GateError> {
    let a = args()?;
    let oracle = data_dir().join("bin/iq_quant_ref");
    if !oracle.is_file() {
        return Err(format!(
            "no oracle at {} (just build-iq-quant-ref)",
            oracle.display()
        )
        .into());
    }
    let set = gguf::Split::open(&a.first)?;
    let (_, t) = set
        .find(&a.tensor)
        .ok_or_else(|| format!("no tensor {} in the set", a.tensor))?;
    let (ty, dims) = (t.ty, t.dims.clone());
    if !matches!(
        ty,
        GgmlType::MXFP4 | GgmlType::BF16 | GgmlType::F16 | GgmlType::F32
    ) {
        return Err(format!("{} is {ty:?}: F32, F16, BF16 or MXFP4", a.tensor).into());
    }
    let (ne0, ne1) = (dims[0] as usize, dims[1] as usize);
    let n_exp = dims.get(2).copied().unwrap_or(1) as usize;
    let data = set.data(&a.tensor)?;
    let expert_bytes = data.len() / n_exp;
    let qw: Vec<f32> = match &a.imatrix {
        Some(p) => imatrix_weights(&gguf::Gguf::open(p)?, &a.tensor)?,
        None => {
            let mut s = 0x2545_f491_4f6c_dd1du64;
            (0..ne0 * n_exp)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    1e-3 + (s >> 40) as f32 / 16_777_216.0
                })
                .collect()
        }
    };
    if qw.len() != ne0 * n_exp {
        return Err(format!(
            "importance holds {} values, the tensor needs {}",
            qw.len(),
            ne0 * n_exp
        )
        .into());
    }
    println!(
        "tensor {} {ty:?} {dims:?} ({:.2} GB) -> {:?}; importance: {}",
        a.tensor,
        data.len() as f64 / 1e9,
        a.kind,
        a.imatrix
            .as_ref()
            .map_or("synthetic (seeded, positive)".to_string(), |p| p
                .display()
                .to_string())
    );
    let gpu = Gpu::new()?;
    let stream = gpu.stream();
    let q = IqQuant::new(gpu.context(), stream, a.kind)?;
    let qw_d = DeviceBuffer::from_host(stream, &qw)?;
    let row_out = ne0 / 256 * a.kind.block_bytes();
    let mut kept: Vec<(usize, Vec<u8>)> = Vec::new();
    let (mut t_read, mut t_card) = (0.0f64, 0.0f64);
    let t_all = Instant::now();
    let mut e0 = 0;
    while e0 < n_exp {
        let e1 = (e0 + a.batch).min(n_exp);
        let t = Instant::now();
        let src = data[e0 * expert_bytes..e1 * expert_bytes].to_vec();
        t_read += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let out = q.quantize(stream, ty, &src, ne0, e0 * ne1, ne1, &qw_d)?;
        t_card += t.elapsed().as_secs_f64();
        for &e in a.check.iter().filter(|&&e| (e0..e1).contains(&e)) {
            let at = (e - e0) * ne1 * row_out;
            kept.push((e, out[at..at + ne1 * row_out].to_vec()));
        }
        e0 = e1;
    }
    let wall = t_all.elapsed().as_secs_f64();
    let w = (ne0 * ne1 * n_exp) as f64;
    println!(
        "card quantize: {:.3e} weights in {wall:.1} s wall: read (page-in copy) {t_read:.1} s, quantize call (H2D, three \
         launches, D2H, pack) {t_card:.1} s = {:.3e} weights/s",
        w,
        w / t_card
    );
    let tm = q.times();
    println!(
        "  terms: host words {:.2} s, upload {:.2} s, widen {:.2} s, search {:.2} s ({:.3e} weights/s), finish {:.2} s, \
         download {:.2} s, pack {:.2} s",
        tm.host_words,
        tm.upload,
        tm.widen,
        tm.search,
        w / tm.search,
        tm.finish,
        tm.download,
        tm.pack
    );
    std::fs::create_dir_all(&a.scratch)?;
    let mut ok = true;
    for (e, card) in &kept {
        let mut buf = b"IQQ1".to_vec();
        let dst: u32 = if a.kind == IqKind::Iq1S { 19 } else { 16 };
        let src: u32 = match ty {
            GgmlType::MXFP4 => 39,
            GgmlType::F32 => 0,
            o => return Err(format!("the oracle reads F32 or MXFP4, not {o:?}").into()),
        };
        buf.extend_from_slice(&dst.to_le_bytes());
        buf.extend_from_slice(&src.to_le_bytes());
        for v in [ne0, ne1, ne1, 1] {
            buf.extend_from_slice(&(v as u64).to_le_bytes());
        }
        buf.extend_from_slice(&data[e * expert_bytes..(e + 1) * expert_bytes]);
        qw[e * ne0..(e + 1) * ne0]
            .iter()
            .for_each(|v| buf.extend_from_slice(&v.to_le_bytes()));
        let (inp, outp) = (
            a.scratch.join(format!("probe-{e}.in")),
            a.scratch.join(format!("probe-{e}.out")),
        );
        std::fs::write(&inp, &buf)?;
        let t = Instant::now();
        let st = std::process::Command::new(&oracle)
            .arg(&inp)
            .arg(&outp)
            .status()?;
        let cpu_s = t.elapsed().as_secs_f64();
        if !st.success() {
            return Err(format!("{}: {st}", oracle.display()).into());
        }
        let want = std::fs::read(&outp)?;
        std::fs::remove_file(&inp)?;
        std::fs::remove_file(&outp)?;
        let diff = card.iter().zip(&want).position(|(x, y)| x != y);
        let same = card.len() == want.len() && diff.is_none();
        println!(
            "expert {e}: {} rows x {ne0}: card {} ggml ({} bytes; ggml one thread {cpu_s:.1} s){}",
            ne1,
            if same { "=" } else { "!=" },
            want.len(),
            diff.map_or(String::new(), |p| format!(
                ", first difference row {} byte {}",
                p / row_out,
                p % row_out
            ))
        );
        ok &= same;
    }
    if !ok {
        return Err("card bytes differ from ggml's".into());
    }
    Ok(())
}
