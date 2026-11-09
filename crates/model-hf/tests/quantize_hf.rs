//! `bloomery-quantize --hf DIR`: the checkpoint folder read as the split `prepare` writes of it, without writing that
//! split. Three clauses, on a toy Kimi K3.1 folder (`quantize_hf_toy`, any machine) and on the 2-layer cut of the real
//! folder (`hw_quantize_hf_cut`, `just gate-quantize-hf`, the box; the cut is `just cut-kimik31q`):
//!
//! - (a) the input's header read from the folder (`quantize_source::open`) is the header of the split `prepare` wrote
//!   of it, key by key and tensor by tensor (name, dims, type, shard), and so is the header of the set written from it;
//! - (b) with the same plan and imatrix, the set written from `--hf` is byte for byte the set written from the
//!   `prepare` split, sha256 per shard;
//! - (c) a folder missing a shard its index names, a `weight_scale` with no `weight` and an E8M0 scale byte of 255
//!   each end at 1 by name, and no file is left.
//!
//! The split `prepare` writes is the pipeline's own intermediate (`tools/ref/k31-quant.sh prepare`).

#[path = "common/kimi_toy.rs"]
mod kimi_toy;
#[path = "../../quantize/tests/common/mod.rs"]
mod qcommon;

use std::path::{Path, PathBuf};
use std::process::Command;

use gguf::quant::GgmlType;
use gguf::write::{Layout, TensorDecl, Writer};
use gguf::{Gguf, Split, Value};
use quantize::cli::{Args, HfInput, Input, Parsed, parse};
use quantize::source::Source;
use safetensors::{Dtype, Safetensors};
use sha2::{Digest, Sha256};

const DROP_INDEXER: &str = "blk.*.indexer.*";
const FLAGS: [&str; 4] = [
    "--output-tensor-type",
    "q6_k",
    "--token-embedding-type",
    "q4_k",
];

fn work_dir(tag: &str) -> PathBuf {
    let base = std::env::var_os("K3Q_WORK").map_or_else(std::env::temp_dir, PathBuf::from);
    let d = base.join(format!("qhf-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The first shard `prepare` wrote at `prefix`.
fn first_shard(prefix: &Path) -> PathBuf {
    let dir = prefix.parent().unwrap();
    let stem = format!(
        "{}-00001-of-",
        prefix.file_name().unwrap().to_str().unwrap()
    );
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with(&stem))
        .collect();
    assert_eq!(found.len(), 1, "{found:?}");
    found.remove(0)
}

/// `prepare` of `folder` in K3's shape into `prefix`, shards of `gib` GiB: the pipeline's intermediate set.
fn prepare_split(folder: &Path, prefix: &Path, gib: u64) -> PathBuf {
    let o = Command::new(env!("CARGO_BIN_EXE_prepare"))
        .env_remove("BLOOMERY_PREPARE_Q8")
        .arg(folder)
        .arg(prefix)
        .args(["--shard-gib", &gib.to_string()])
        .args(["--drop", DROP_INDEXER, "--arch", "kimi-k3"])
        .output()
        .unwrap();
    assert!(
        o.status.success(),
        "prepare: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    first_shard(prefix)
}

fn hf_input(folder: &Path, gib: u64) -> HfInput {
    HfInput {
        dir: folder.to_path_buf(),
        arch: Some("kimi-k3".into()),
        drop: vec![DROP_INDEXER.into()],
        shard_gib: gib,
    }
}

/// An importance matrix over every stack and two of every three trunk weights of the split at `first`.
fn write_imatrix(path: &Path, first: &Path) {
    let set = Split::open(first).unwrap();
    let mut rng = qcommon::Rng(0x00ff_aa55_1234_5678);
    let mut tensors: Vec<(String, Vec<u64>, Vec<u8>)> = Vec::new();
    let mut k = 0usize;
    for (_, t) in set.iter_tensors() {
        if !t.name.ends_with(".weight") || t.dims.len() < 2 {
            continue;
        }
        if t.dims.len() < 3 {
            k += 1;
            if k.is_multiple_of(3) {
                continue;
            }
        }
        let ne0 = t.dims[0] as usize;
        let n_exp: usize = t.dims[2..].iter().product::<u64>().max(1) as usize;
        let counts: Vec<f32> = (0..n_exp).map(|e| [3.0, 0.0, 5.0, 7.0][e % 4]).collect();
        let sums: Vec<f32> = (0..ne0 * n_exp).map(|_| 10.0 * rng.unit()).collect();
        tensors.push((
            format!("{}.in_sum2", t.name),
            vec![ne0 as u64, n_exp as u64],
            qcommon::f32_bytes(&sums),
        ));
        tensors.push((
            format!("{}.counts", t.name),
            vec![1, n_exp as u64],
            qcommon::f32_bytes(&counts),
        ));
    }
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
        .map(|(name, dims, bytes)| TensorDecl {
            name: name.clone(),
            dims: dims.clone(),
            type_id: GgmlType::F32.as_u32(),
            nbytes: bytes.len() as u64,
        })
        .collect();
    let mut w = Writer::new(
        std::fs::File::create(path).unwrap(),
        Layout::new(&kvs, decls).unwrap(),
    )
    .unwrap();
    for (name, _, bytes) in &tensors {
        w.tensor(name, bytes).unwrap();
    }
    w.finish().unwrap();
}

/// The arguments of one run: `input` (the first shard or `--hf …`), the output prefix under `--keep-split`.
fn args(input: &[String], out: &Path, im: &Path, rules: &Path) -> Args {
    let mut a: Vec<String> = vec![
        "--allow-requantize".into(),
        "--keep-split".into(),
        "--cpu".into(),
        "--imatrix".into(),
        im.display().to_string(),
        "--tensor-type-file".into(),
        rules.display().to_string(),
    ];
    a.extend(FLAGS.map(String::from));
    a.extend(input.iter().cloned());
    a.extend([out.display().to_string(), "Q4_K_M".into()]);
    let Parsed::Run(args) = parse(a).unwrap() else {
        panic!("not a run");
    };
    args
}

fn hf_args(h: &HfInput) -> Vec<String> {
    vec![
        "--hf".into(),
        h.dir.display().to_string(),
        "--hf-arch".into(),
        h.arch.clone().unwrap(),
        "--hf-drop".into(),
        h.drop[0].clone(),
        "--hf-shard-gib".into(),
        h.shard_gib.to_string(),
    ]
}

/// The run `a` asks for, the folder's source opened as `--hf` opens it.
fn run_args(a: &Args) -> quantize::Result<quantize::run::Summary> {
    let prep = match &a.input {
        Input::Hf(h) => quantize::run::prepare_from(a, model_hf::quantize_source::open(h)?)?,
        Input::First(_) => quantize::run::prepare(a)?,
    };
    quantize::run(
        a,
        prep,
        &mut quantize::host::HostBackend::new(),
        &mut |_| {},
    )
}

fn sha256(p: &Path) -> String {
    let mut f = std::fs::File::open(p).unwrap();
    let mut h = Sha256::new();
    std::io::copy(&mut f, &mut h).unwrap();
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

type Tensors = Vec<(usize, String, Vec<u64>, GgmlType, u64)>;

fn without_split(keys: &[(String, Value)]) -> Vec<(String, Value)> {
    keys.iter()
        .filter(|(k, _)| !k.starts_with("split."))
        .cloned()
        .collect()
}

/// Clause (a), the input: the keys and the tensors of the folder's source and of the split `prepare` wrote.
fn assert_same_input(source: &dyn Source, first: &Path) {
    let set = Split::open(first).unwrap();
    assert_eq!(
        without_split(source.keys()),
        without_split(set.keys()),
        "the folder's header keys differ from the split's"
    );
    assert_eq!(source.shard_count(), set.shard_count(), "shards");
    let rows = |t: Vec<(usize, gguf::TensorInfo)>| -> Tensors {
        t.into_iter()
            .map(|(s, t)| (s, t.name, t.dims, t.ty, t.nbytes))
            .collect()
    };
    let mine = rows(source.tensors());
    let theirs = rows(set.iter_tensors().map(|(s, t)| (s, t.clone())).collect());
    assert_eq!(mine.len(), theirs.len(), "tensor counts");
    for (m, t) in mine.iter().zip(&theirs) {
        assert_eq!(m, t, "tensor {} of the folder's source", m.1);
    }
}

/// The output files `prefix` names, sorted.
fn outputs(dir: &Path, stem: &str) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let n = p.file_name().unwrap().to_string_lossy().to_string();
            n.starts_with(&format!("{stem}-0")) && n.ends_with(".gguf")
        })
        .collect();
    v.sort();
    v
}

/// Clauses (a) and (b): the set from the folder against the set from the `prepare` split.
fn equal_sets(folder: &Path, work: &Path, gib: u64) {
    let first = prepare_split(folder, &work.join("mid"), gib);
    let h = hf_input(folder, gib);
    let source = model_hf::quantize_source::open(&h).unwrap();
    assert_same_input(&*source, &first);
    println!("clause (a) the folder's header is the prepared split's: PASS");

    let (im, rules) = (work.join("im.gguf"), work.join("rules.txt"));
    write_imatrix(&im, &first);
    std::fs::write(&rules, qcommon::k31_rules("q4_k")).unwrap();
    let (from_set, from_hf) = (work.join("from-set"), work.join("from-hf"));
    run_args(&args(
        &[first.display().to_string()],
        &from_set,
        &im,
        &rules,
    ))
    .unwrap();
    run_args(&args(&hf_args(&h), &from_hf, &im, &rules)).unwrap();
    let (a, b) = (outputs(work, "from-set"), outputs(work, "from-hf"));
    assert!(!a.is_empty());
    assert_eq!(a.len(), b.len(), "shard counts");
    for (x, y) in a.iter().zip(&b) {
        let (gx, gy) = (Gguf::open(x).unwrap(), Gguf::open(y).unwrap());
        assert_eq!(gx.keys(), gy.keys(), "{}: keys", y.display());
        let names = |g: &Gguf| -> Vec<(String, Vec<u64>, GgmlType)> {
            g.iter_tensors()
                .map(|t| (t.name.clone(), t.dims.clone(), t.ty))
                .collect()
        };
        assert_eq!(names(&gx), names(&gy), "{}: tensors", y.display());
    }
    println!("clause (a) the written sets' headers are equal: PASS");
    for (x, y) in a.iter().zip(&b) {
        let (hx, hy) = (sha256(x), sha256(y));
        println!("sha256 {hx} {}\nsha256 {hy} {}", x.display(), y.display());
        assert_eq!(hx, hy, "{} differs from {}", y.display(), x.display());
    }
    println!("clause (b) {} shard(s) equal by sha256: PASS", a.len());
}

/// `src` with every file linked but `skip`, in a new folder `dst`.
fn link_folder(src: &Path, dst: &Path, skip: &[&str]) {
    std::fs::create_dir_all(dst).unwrap();
    for e in std::fs::read_dir(src).unwrap() {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_str().unwrap().to_string();
        if !skip.contains(&name.as_str()) {
            std::os::unix::fs::symlink(&p, dst.join(&name)).unwrap();
        }
    }
}

/// The shard of `src` that stores the tensor `stem.<suffix>` of the first expert-stack tensor `pick` matches.
fn shard_with(src: &Path, pick: &str) -> (String, String) {
    let map = safetensors::Folder::weight_map(&src.join(safetensors::Folder::INDEX)).unwrap();
    let (name, file) = map
        .iter()
        .find(|(n, _)| n.contains(pick))
        .unwrap_or_else(|| panic!("no tensor named like {pick}"));
    (name.clone(), file.clone())
}

type Rows = Vec<(String, Dtype, Vec<usize>, Vec<u8>)>;

/// `src` with the shard `file` rewritten by `edit` (which may drop a tensor: the index follows), in a new folder.
fn mutate(src: &Path, dst: &Path, file: &str, edit: impl FnOnce(&mut Rows)) {
    link_folder(src, dst, &[file, safetensors::Folder::INDEX]);
    let st = Safetensors::open(&src.join(file)).unwrap();
    let mut rows: Rows = st
        .names()
        .map(|n| {
            let r = st.raw(n).unwrap();
            (n.to_string(), r.dtype, r.shape.to_vec(), r.bytes.to_vec())
        })
        .collect();
    edit(&mut rows);
    let views: Vec<(&str, &str, &[usize], &[u8])> = rows
        .iter()
        .map(|(n, d, s, b)| (n.as_str(), d.name(), s.as_slice(), b.as_slice()))
        .collect();
    std::fs::write(dst.join(file), safetensors::write(&views)).unwrap();
    let mut map = safetensors::Folder::weight_map(&src.join(safetensors::Folder::INDEX)).unwrap();
    map.retain(|n, f| f != file || rows.iter().any(|r| &r.0 == n));
    let index = serde_json::json!({"metadata": {}, "weight_map": map});
    std::fs::write(dst.join(safetensors::Folder::INDEX), index.to_string()).unwrap();
}

/// Clause (c): each broken folder ends the run at 1 by name, and the output directory holds nothing after.
fn refusals(folder: &Path, work: &Path, gib: u64) {
    let (im, rules) = (work.join("im-c.gguf"), work.join("rules-c.txt"));
    let first = first_shard(&work.join("mid"));
    write_imatrix(&im, &first);
    std::fs::write(&rules, qcommon::k31_rules("q4_k")).unwrap();
    let (stack, shard) = shard_with(folder, ".experts.1.w1.weight_scale");
    let stem = stack.strip_suffix(".weight_scale").unwrap().to_string();
    let mut cases: Vec<(&str, PathBuf, Vec<&str>)> = Vec::new();

    let missing = work.join("c-missing");
    link_folder(folder, &missing, &[&shard]);
    cases.push(("a missing shard", missing, vec![shard.as_str()]));

    let orphan = work.join("c-orphan");
    let weight = format!("{stem}.weight_packed");
    mutate(folder, &orphan, &shard, |rows| {
        rows.retain(|r| r.0 != weight)
    });
    cases.push(("a weight_scale with no weight", orphan, vec![stem.as_str()]));

    let nan = work.join("c-nan");
    let scale = format!("{stem}.weight_scale");
    mutate(folder, &nan, &shard, |rows| {
        let r = rows.iter_mut().find(|r| r.0 == scale).unwrap();
        r.3[0] = 255;
    });
    cases.push(("an E8M0 scale byte of 255", nan, vec!["0xff"]));

    for (what, dir, names) in cases {
        let out_dir = work.join(format!(
            "out-{}",
            dir.file_name().unwrap().to_string_lossy()
        ));
        std::fs::create_dir_all(&out_dir).unwrap();
        let out = out_dir.join("q");
        let a = args(&hf_args(&hf_input(&dir, gib)), &out, &im, &rules);
        let e = run_args(&a).expect_err(what);
        let text = e.to_string();
        println!("clause (c) {what}: exit {}: {text}", e.status());
        assert_eq!(e.status(), 1, "{what}: {text}");
        assert!(
            names.iter().any(|n| text.contains(n)),
            "{what}: the refusal names none of {names:?}: {text}"
        );
        let left: Vec<_> = std::fs::read_dir(&out_dir).unwrap().collect();
        assert!(left.is_empty(), "{what}: files left: {left:?}");
    }
    println!("clause (c) three refusals, each at 1 by name with no file left: PASS");
}

fn gate(folder: &Path, tag: &str, gib: u64) {
    let work = work_dir(tag);
    equal_sets(folder, &work, gib);
    refusals(folder, &work, gib);
    let _ = std::fs::remove_dir_all(&work);
}

/// The toy folder, one shard in the split: the three clauses on any machine.
#[test]
fn quantize_hf_toy() {
    let dir = work_dir("toy-folder");
    let folder = dir.join("k31");
    kimi_toy::write_folder(&folder, true, None);
    gate(&folder, "toy", 1);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The 2-layer cut of the real K3.1 folder, shards of 2 GiB (`K3Q_HF_CUT`, else `<fixture root>/kimik31q`).
#[test]
#[ignore = "hw: the box and the kimik31q cut (`just cut-kimik31q`)"]
fn hw_quantize_hf_cut() {
    let folder = std::env::var_os("K3Q_HF_CUT").map_or_else(
        || PathBuf::from(refset::fixture::root().unwrap()).join("kimik31q"),
        PathBuf::from,
    );
    assert!(folder.is_dir(), "no cut folder at {}", folder.display());
    gate(&folder, "cut", 2);
}
