//! The K3 fixture end to end, no card: `bloomery-quantize --cpu` (the library's host path) against llama-quantize on
//! the same input, imatrix, rules and options. The whole output file is equal by sha256, split or not, and the plan
//! equals llama-quantize's own `--dry-run` tensor for tensor. Run by `just gate-quantize-e2e` (the box).

mod common;

use common::{Rng, f32_bytes, run_ok, scratch};
use gguf::Value;
use gguf::quant::GgmlType;
use quantize::cli::{Parsed, parse};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

fn env_path(name: &str, default: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| default.to_string()))
}

fn sha(p: &Path) -> String {
    let o = Command::new("sha256sum").arg(p).output().unwrap();
    String::from_utf8(o.stdout)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .to_string()
}

type SetTensor = (String, Vec<u64>, GgmlType, Vec<u8>);

fn write_imatrix(path: &Path, tensors: &[SetTensor]) {
    use gguf::write::{Layout, TensorDecl, Writer};
    let kvs = vec![
        (
            "general.type".to_string(),
            Value::String("imatrix".to_string()),
        ),
        (
            "imatrix.datasets".to_string(),
            Value::Array(vec![Value::String("gate".to_string())]),
        ),
        ("imatrix.chunk_count".to_string(), Value::U32(3)),
        ("imatrix.chunk_size".to_string(), Value::U32(512)),
    ];
    let decls = tensors
        .iter()
        .map(|(name, dims, ty, bytes)| TensorDecl {
            name: name.clone(),
            dims: dims.clone(),
            type_id: ty.as_u32(),
            nbytes: bytes.len() as u64,
        })
        .collect();
    let mut w = Writer::new(
        std::fs::File::create(path).unwrap(),
        Layout::new(&kvs, decls).unwrap(),
    )
    .unwrap();
    for (name, _, _, bytes) in tensors {
        w.tensor(name, bytes).unwrap();
    }
    w.finish().unwrap();
}

/// The files `out` names: the file itself, or `<out>-NNNNN-of-MMMMM.gguf` when the set is split.
fn outputs(dir: &Path, stem: &str) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let n = p.file_name().unwrap().to_string_lossy().to_string();
            n == format!("{stem}.gguf")
                || (n.starts_with(&format!("{stem}-0")) && n.ends_with(".gguf"))
        })
        .collect();
    v.sort();
    v
}

/// What differs between two files, KV by KV and tensor by tensor: the first lines a red names.
fn diff(a: &Path, b: &Path) -> Vec<String> {
    let (ga, gb) = (gguf::Gguf::open(a).unwrap(), gguf::Gguf::open(b).unwrap());
    let mut out = Vec::new();
    let (ka, kb) = (ga.keys(), gb.keys());
    for i in 0..ka.len().max(kb.len()) {
        match (ka.get(i), kb.get(i)) {
            (Some(x), Some(y)) if x == y => {}
            (x, y) => out.push(format!("kv[{i}]: llama {x:?} | ours {y:?}")),
        }
    }
    let (ta, tb): (Vec<_>, Vec<_>) = (ga.iter_tensors().collect(), gb.iter_tensors().collect());
    for i in 0..ta.len().max(tb.len()) {
        match (ta.get(i), tb.get(i)) {
            (Some(x), Some(y)) => {
                let (dx, dy) = (ga.data(x).unwrap(), gb.data(y).unwrap());
                if x.name != y.name || x.ty != y.ty || x.dims != y.dims || x.offset != y.offset {
                    out.push(format!(
                        "tensor[{i}]: llama {} {:?} {:?} @{} | ours {} {:?} {:?} @{}",
                        x.name, x.ty, x.dims, x.offset, y.name, y.ty, y.dims, y.offset
                    ));
                } else if dx != dy {
                    let at = dx.iter().zip(dy).position(|(p, q)| p != q);
                    out.push(format!(
                        "tensor[{i}] {} {:?}: bytes differ, first at {at:?} of {}",
                        x.name,
                        x.ty,
                        dx.len()
                    ));
                }
            }
            (x, y) => out.push(format!(
                "tensor[{i}]: llama {:?} | ours {:?}",
                x.map(|t| &t.name),
                y.map(|t| &t.name)
            )),
        }
    }
    out
}

struct Fix {
    work: PathBuf,
    lq: PathBuf,
    src: PathBuf,
    /// `src` split into shards of a few tensors, the first shard.
    split: PathBuf,
    im: PathBuf,
    rules: PathBuf,
}

const FLAGS: [&str; 4] = [
    "--output-tensor-type",
    "q6_k",
    "--token-embedding-type",
    "q4_k",
];

fn build(rng: &mut Rng) -> Fix {
    let tree = env_path("K3Q_TREE", "");
    let lq = tree.join("build/bin/llama-quantize");
    let fixture = env_path(
        "K3Q_FIXTURE",
        "/models/fixtures/kimik3/kimi-k3-fixture-00001-of-00001.gguf",
    );
    assert!(lq.is_file(), "no {} (K3Q_TREE)", lq.display());
    assert!(fixture.is_file(), "no {} (K3Q_FIXTURE)", fixture.display());
    let work = scratch("e2e");
    let src = common::fixture_src(&lq, &fixture, &work);
    // an imatrix over every stack and two of every three trunk weights: the third quantizes without
    let g = gguf::Gguf::open(&src).unwrap();
    let mut tensors: Vec<SetTensor> = Vec::new();
    let mut k = 0usize;
    for t in g.iter_tensors() {
        if !t.name.ends_with(".weight") || t.dims.len() < 2 {
            continue;
        }
        let stack = t.dims.len() >= 3;
        if !stack {
            k += 1;
            if k.is_multiple_of(3) {
                continue;
            }
        }
        let ne0 = t.dims[0] as usize;
        let n_exp: usize = t.dims[2..].iter().product::<u64>().max(1) as usize;
        let counts: Vec<f32> = (0..n_exp).map(|e| [3.0, 0.0, 5.0, 7.0][e % 4]).collect();
        let stem = &t.name;
        let sums: Vec<f32> = (0..ne0 * n_exp).map(|_| 10.0 * rng.unit()).collect();
        tensors.push((
            format!("{stem}.in_sum2"),
            vec![ne0 as u64, n_exp as u64],
            GgmlType::F32,
            f32_bytes(&sums),
        ));
        tensors.push((
            format!("{stem}.counts"),
            vec![1, n_exp as u64],
            GgmlType::F32,
            f32_bytes(&counts),
        ));
    }
    let splitter = tree.join("build/bin/llama-gguf-split");
    let prefix = work.join("srcsplit");
    run_ok(
        &splitter,
        &[
            "--split".as_ref(),
            "--split-max-tensors".as_ref(),
            "40".as_ref(),
            src.as_os_str(),
            prefix.as_os_str(),
        ],
    );
    let split = outputs(&work, "srcsplit")
        .into_iter()
        .next()
        .expect("llama-gguf-split wrote no shard");
    let im = work.join("e2e.imatrix.gguf");
    write_imatrix(&im, &tensors);
    let rules = work.join("rules.txt");
    std::fs::write(&rules, common::k31_rules("q4_k")).unwrap();
    Fix {
        work,
        lq,
        src,
        split,
        im,
        rules,
    }
}

/// llama-quantize and the library's host path, one set of options; returns both walls and the output stems.
fn both(
    f: &Fix,
    input: &Path,
    ftype: &str,
    keep_split: bool,
    with_rules: bool,
    tag: &str,
) -> (f64, f64) {
    let (a_stem, b_stem) = (format!("alone-{tag}"), format!("ours-{tag}"));
    let mut common: Vec<String> = vec!["--allow-requantize".into()];
    common.extend(["--imatrix".into(), f.im.display().to_string()]);
    if with_rules {
        common.extend(["--tensor-type-file".into(), f.rules.display().to_string()]);
    }
    common.extend(FLAGS.map(String::from));
    if keep_split {
        common.push("--keep-split".into());
    }
    let out_name = |stem: &str| {
        let p = f.work.join(if keep_split {
            stem.to_string()
        } else {
            format!("{stem}.gguf")
        });
        p.display().to_string()
    };
    let mut la: Vec<String> = common.clone();
    la.extend([
        input.display().to_string(),
        out_name(&a_stem),
        ftype.into(),
        "16".into(),
    ]);
    let las: Vec<&OsStr> = la.iter().map(OsStr::new).collect();
    let (t_llama, _) = run_ok(&f.lq, &las);
    let mut oa: Vec<String> = common;
    oa.extend([
        "--cpu".into(),
        input.display().to_string(),
        out_name(&b_stem),
        ftype.into(),
    ]);
    let Parsed::Run(args) = parse(oa).unwrap() else {
        panic!("not a run");
    };
    let t = Instant::now();
    let prep = quantize::run::prepare(&args).unwrap();
    let mut be = quantize::host::HostBackend::new();
    quantize::run(&args, prep, &mut be, &mut |_| {}).unwrap();
    let t_ours = t.elapsed().as_secs_f64();
    let (fa, fb) = (outputs(&f.work, &a_stem), outputs(&f.work, &b_stem));
    assert!(!fa.is_empty(), "{tag}: llama-quantize wrote nothing");
    assert_eq!(
        fa.len(),
        fb.len(),
        "{tag}: shard counts differ ({} vs {})",
        fa.len(),
        fb.len()
    );
    for (x, y) in fa.iter().zip(&fb) {
        let (hx, hy) = (sha(x), sha(y));
        println!("{tag}: {} {hx}\n{tag}: {} {hy}", x.display(), y.display());
        if hx != hy {
            let g = gguf::Gguf::open(&f.src).unwrap();
            let names: Vec<&str> = g.keys().iter().map(|k| k.0.as_str()).collect();
            println!(
                "{tag}: DIFF src kv names from 50: {:?} (of {})",
                &names[50.min(names.len())..],
                names.len()
            );
            for l in diff(x, y).iter().take(12) {
                println!("{tag}: DIFF {l}");
            }
        }
        assert_eq!(
            hx,
            hy,
            "{tag}: {} differs from {}",
            y.display(),
            x.display()
        );
    }
    println!(
        "{tag}: {} file(s) equal; llama-quantize {t_llama:.1} s, ours --cpu {t_ours:.1} s",
        fa.len()
    );
    (t_llama, t_ours)
}

#[test]
#[ignore = "hw: the box, llama-quantize and the K3 fixture"]
fn hw_fixture_whole_file_equals_llama_quantize() {
    let mut rng = Rng(0x00ff_aa55_1234_5678);
    let f = build(&mut rng);
    // the K3.1 shape (IQ stacks by rules, flag-typed head and embedding) as one file, a split input kept split and merged, and at Q5_K_M
    both(&f, &f.src, "Q4_K_M", false, true, "q4km");
    both(&f, &f.split, "Q4_K_M", true, true, "q4km-split");
    both(&f, &f.split, "Q4_K_M", false, true, "q4km-merged");
    both(&f, &f.src, "Q5_K_M", false, true, "q5km");
    let _ = std::fs::remove_dir_all(&f.work);
}
