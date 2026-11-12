//! `--resume`'s contracts over a synthetic three-shard set, host only (`--cpu`), no card and no
//! oracle: a shard an earlier run of the same plan finished is kept and the finished set is
//! byte-identical to one pass; a whole file that is not this plan's shard is refused by name and
//! left in place; a failure keeps every whole shard and removes the `.partial` it was writing.

mod common;

use common::{Rng, f32_bytes, scratch};
use gguf::Value;
use gguf::iq_quant::IqKind;
use gguf::quant::GgmlType;
use gguf::write::{Layout, TensorDecl, Writer};
use quantize::cli::{Parsed, parse};
use quantize::run::Summary;
use quantize::source::{Source, SplitSource};
use quantize::{BackendError, Error, Result};
use std::borrow::Cow;
use std::ops::Range;
use std::path::{Path, PathBuf};

/// The synthetic set's tensors, shard by shard: quantized weights and one copied norm.
const SHARDS: [&[(&str, &[u64])]; 3] = [
    &[
        ("token_embd.weight", &[256, 2]),
        ("blk.0.attn_q.weight", &[256, 2]),
        ("blk.0.attn_v.weight", &[256, 2]),
        ("blk.0.attn_norm.weight", &[256]),
    ],
    &[
        ("blk.1.ffn_gate.weight", &[256, 2]),
        ("blk.1.ffn_up.weight", &[256, 2]),
    ],
    &[
        ("blk.1.ffn_down.weight", &[256, 2]),
        ("output.weight", &[256, 2]),
    ],
];

/// One shard of the set: seeded finite F32 tensors, the kimi-k3 header on the first.
fn write_shard(dir: &Path, i: usize, rng: &mut Rng) {
    let mut kvs: Vec<(String, Value)> = vec![(
        "general.architecture".to_string(),
        Value::String("kimi-k3".to_string()),
    )];
    if i == 0 {
        kvs.push(("kimi-k3.block_count".to_string(), Value::U32(2)));
        kvs.push(("kimi-k3.expert_count".to_string(), Value::U32(8)));
    }
    kvs.push(("split.no".to_string(), Value::U16(i as u16)));
    kvs.push(("split.tensors.count".to_string(), Value::I32(8)));
    kvs.push(("split.count".to_string(), Value::U16(3)));
    let tensors: Vec<(String, Vec<u64>, GgmlType, Vec<u8>)> = SHARDS[i]
        .iter()
        .map(|&(name, dims)| {
            let n: usize = dims.iter().product::<u64>() as usize;
            let values: Vec<f32> = (0..n).map(|_| 2.0 * rng.unit() - 1.0).collect();
            (
                name.to_string(),
                dims.to_vec(),
                GgmlType::F32,
                f32_bytes(&values),
            )
        })
        .collect();
    let decls: Vec<TensorDecl> = tensors
        .iter()
        .map(|(name, dims, ty, bytes)| TensorDecl {
            name: name.clone(),
            dims: dims.clone(),
            type_id: ty.as_u32(),
            nbytes: bytes.len() as u64,
        })
        .collect();
    let path = dir.join(format!("synth-{:05}-of-00003.gguf", i + 1));
    let mut w = Writer::new(
        std::fs::File::create(&path).unwrap(),
        Layout::new(&kvs, decls).unwrap(),
    )
    .unwrap();
    for (name, _, _, bytes) in &tensors {
        w.tensor(name, bytes).unwrap();
    }
    w.finish().unwrap();
}

/// The set's first shard under a fresh `work/synth`.
fn synth_set(work: &Path) -> PathBuf {
    let dir = work.join("synth");
    std::fs::create_dir_all(&dir).unwrap();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for i in 0..SHARDS.len() {
        write_shard(&dir, i, &mut rng);
    }
    dir.join("synth-00001-of-00003.gguf")
}

/// The set read through a `Source` that refuses the named tensors' bytes: a run fails on one
/// after the shards before it are whole, or proves it reads none of them (a kept shard's).
struct FailOn {
    inner: SplitSource,
    names: Vec<String>,
}

impl Source for FailOn {
    fn label(&self) -> String {
        self.inner.label()
    }
    fn keys(&self) -> &[(String, Value)] {
        self.inner.keys()
    }
    fn shard_count(&self) -> usize {
        self.inner.shard_count()
    }
    fn tensors(&self) -> Vec<(usize, gguf::TensorInfo)> {
        self.inner.tensors()
    }
    fn read(&self, name: &str, _bytes: Range<u64>) -> Result<Cow<'_, [u8]>> {
        if self.names.iter().any(|n| n == name) {
            return Err(Error::Refused(format!(
                "{name}: this read is what the test refuses"
            )));
        }
        self.inner.read(name, _bytes)
    }
}

/// A backend shaped like the card's for the host-ahead path: its work runs off the host pool, so
/// the host tensors are converted on the ahead thread and the writer waits for them in order. The
/// synthetic set holds no IQ tensor, so the quantizing half refuses by name if it is ever reached.
struct CardLikeBackend;

impl quantize::Backend for CardLikeBackend {
    fn name(&self) -> String {
        "card-like".to_string()
    }
    fn uses_host_pool(&self) -> bool {
        false
    }
    fn bind(&mut self, kind: IqKind, _qw: &[f32]) -> std::result::Result<(), BackendError> {
        Err(format!("bind of {kind:?}: the synthetic set holds no IQ tensor").into())
    }
    fn quantize(
        &self,
        _ty: GgmlType,
        _src: &[u8],
        _ne0: usize,
        _row0: usize,
        _rows: usize,
    ) -> std::result::Result<Vec<u8>, BackendError> {
        Err("no IQ tensor is quantized over this backend".into())
    }
}

/// A `--cpu --keep-split` run of the synthetic set under `out` (a fresh directory the caller made),
/// `--resume` when asked, its log collected; `fail` names the tensors whose reads refuse.
fn run(first: &Path, out: &Path, resume: bool, fail: &[&str]) -> (Result<Summary>, Vec<String>) {
    run_with(
        first,
        out,
        resume,
        fail,
        &mut quantize::host::HostBackend::new(),
    )
}

/// [`run`] over the caller's backend: the card-like one turns the host-ahead thread on.
fn run_with(
    first: &Path,
    out: &Path,
    resume: bool,
    fail: &[&str],
    backend: &mut dyn quantize::Backend,
) -> (Result<Summary>, Vec<String>) {
    let mut argv = vec!["--cpu".to_string(), "--keep-split".to_string()];
    if resume {
        argv.push("--resume".to_string());
    }
    argv.extend([
        first.display().to_string(),
        out.display().to_string(),
        "Q8_0".to_string(),
    ]);
    let Parsed::Run(a) = parse(argv).unwrap() else {
        panic!("a run");
    };
    let split = gguf::Split::open(first).unwrap();
    let set: Box<dyn Source + Send> = if fail.is_empty() {
        Box::new(SplitSource(split))
    } else {
        Box::new(FailOn {
            inner: SplitSource(split),
            names: fail.iter().map(|n| (*n).to_string()).collect(),
        })
    };
    let prep = quantize::run::prepare_from(&a, set).unwrap();
    let mut log = Vec::new();
    let r = quantize::run(&a, prep, backend, &mut |l: &str| log.push(l.to_string()));
    (r, log)
}

/// The run's three output shards, `PREFIX-00001-of-00003.gguf` and on.
fn outs(dir: &Path) -> Vec<PathBuf> {
    (1..=3)
        .map(|i| dir.join(format!("set-{:05}-of-00003.gguf", i)))
        .collect()
}

/// Every file under `dir` whose name ends in `.partial`.
fn partials(dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.as_encoded_bytes().ends_with(b".partial"))
        })
        .collect()
}

/// A whole run of the plan gives bytes A; a run made to fail once its first shard is whole leaves
/// that shard, and `--resume` finishes to the same bytes A, keeping the shard without reading or
/// converting it. Mutant (red at the resume run): a whole shard treated as `check_absent` treats
/// it — the resume run is refused "exists".
#[test]
fn resume_equals_one_pass() {
    let work = scratch("resume-one-pass");
    let first = synth_set(&work);
    let a_dir = work.join("a");
    std::fs::create_dir_all(&a_dir).unwrap();
    let (r, _) = run(&first, &a_dir.join("set"), false, &[]);
    let a = r.unwrap();
    assert_eq!(a.kept, 0, "a plain run keeps nothing");
    let want: Vec<Vec<u8>> = outs(&a_dir)
        .iter()
        .map(|p| std::fs::read(p).unwrap())
        .collect::<Vec<_>>();
    assert!(
        want.iter().all(|v| !v.is_empty()),
        "run A wrote three shards"
    );

    // a run of the same plan that fails reading shard 2's second tensor: shard 1 is whole, and the
    // `.partial` of shard 2 is removed by the failure
    let b_dir = work.join("b");
    std::fs::create_dir_all(&b_dir).unwrap();
    let (r, _) = run(&first, &b_dir.join("set"), true, &["blk.1.ffn_up.weight"]);
    let e = r.unwrap_err().to_string();
    assert!(e.contains("blk.1.ffn_up.weight"), "{e}");
    assert!(outs(&b_dir)[0].is_file(), "the failure left shard 1 whole");
    assert!(
        partials(&b_dir).is_empty(),
        "the failure removed its partial"
    );

    // the resume run: shard 1 kept (named in the log), shards 2 and 3 written, the bytes are A's;
    // the kept shard's tensors refuse their reads, so a green run proves none of them was read, and
    // the card-like backend turns the host-ahead thread on, so its list is proved skip-filtered too
    let kept_names: Vec<&str> = SHARDS[0].iter().map(|&(n, _)| n).collect();
    let (r, log) = run_with(
        &first,
        &b_dir.join("set"),
        true,
        &kept_names,
        &mut CardLikeBackend,
    );
    let s = r.unwrap();
    assert_eq!(
        (s.files, s.kept),
        (outs(&b_dir)[1..].to_vec(), 1),
        "the resume run wrote the two missing shards and kept one"
    );
    assert!(
        s.quantized < a.quantized,
        "the kept shard's tensors were not converted again"
    );
    let kept_line = format!("  shard 1/3: kept {}", outs(&b_dir)[0].display());
    let kept_at = log
        .iter()
        .position(|l| l == &kept_line)
        .unwrap_or_else(|| panic!("no kept line; log:\n{}", log.join("\n")));
    let wrote_2 = log
        .iter()
        .position(|l| l.starts_with("  shard 2/3: "))
        .expect("shard 2 was written");
    assert!(kept_at < wrote_2, "the consumer stays in shard order");
    assert!(
        log.last()
            .is_some_and(|l| l == "bloomery-quantize: 2 shard(s) written, 1 kept"),
        "the summary line: {:?}",
        log.last()
    );
    let got: Vec<Vec<u8>> = outs(&b_dir)
        .iter()
        .map(|p| std::fs::read(p).unwrap())
        .collect::<Vec<_>>();
    for (i, (w, g)) in want.iter().zip(&got).enumerate() {
        assert_eq!(w, g, "shard {} differs from the one-pass run", i + 1);
    }
    let _ = std::fs::remove_dir_all(&work);
}

/// A whole shard that is not this plan's — one metadata byte flipped, or a byte short — refuses
/// the run by name (the path and the first field that differs, both byte counts for a length) and
/// the file is left as it was. Mutant (red at the flip): the header comparison out of
/// `kept_whole`, so only the length is checked and the flipped shard is kept.
#[test]
fn a_foreign_shard_is_refused() {
    let work = scratch("resume-foreign");
    let first = synth_set(&work);
    let dir = work.join("out");
    std::fs::create_dir_all(&dir).unwrap();
    let (r, _) = run(&first, &dir.join("set"), true, &[]);
    r.unwrap();
    let files = outs(&dir);

    // one byte of the second shard's only metadata pair's value (split.no, U16): the pairs of a
    // non-first shard are the three split.* ones, so its value sits at 24 + 8 + 8 + 4
    let mut b = std::fs::read(&files[1]).unwrap();
    assert_eq!(
        &b[32..40],
        b"split.no",
        "the first pair of a non-first shard is split.no"
    );
    b[44] ^= 0x08;
    std::fs::write(&files[1], &b).unwrap();
    let (r, _) = run(&first, &dir.join("set"), true, &[]);
    let e = r.unwrap_err();
    assert_eq!(e.status(), 1, "a named refusal: {e}");
    let t = e.to_string();
    assert!(
        t.contains(files[1].display().to_string().as_str()) && t.contains("\"split.no\""),
        "{t}"
    );
    assert_eq!(
        std::fs::read(&files[1]).unwrap(),
        b,
        "the refusal removed nothing"
    );
    b[44] ^= 0x08;
    std::fs::write(&files[1], &b).unwrap();

    // a byte short: the lengths are named, both of them
    let whole = std::fs::read(&files[2]).unwrap();
    std::fs::write(&files[2], &whole[..whole.len() - 1]).unwrap();
    let (r, _) = run(&first, &dir.join("set"), true, &[]);
    let e = r.unwrap_err().to_string();
    assert!(
        e.contains(files[2].display().to_string().as_str())
            && e.contains(&format!("{} B", whole.len() - 1))
            && e.contains(&format!("{} B", whole.len())),
        "{e}"
    );
    assert_eq!(
        std::fs::metadata(&files[2]).unwrap().len() as usize,
        whole.len() - 1,
        "the truncated shard is still there"
    );
    let _ = std::fs::remove_dir_all(&work);
}

/// A failure under `--resume` keeps the whole shards (the kept ones and the ones this run
/// finished) and removes the `.partial` it was writing, so the next `--resume` goes on. Mutant
/// (red at the state after the failure): the cleanup without `--resume`, which removes the whole
/// shard this run finished as well.
#[test]
fn a_failure_under_resume_keeps_whole_shards_and_removes_the_partial() {
    let work = scratch("resume-failure");
    let first = synth_set(&work);
    let dir = work.join("out");
    std::fs::create_dir_all(&dir).unwrap();
    let (r, _) = run(&first, &dir.join("set"), true, &["blk.1.ffn_up.weight"]);
    assert!(r.is_err(), "the run fails on the refused read");
    assert!(
        outs(&dir)[0].is_file(),
        "the whole shard this run finished stays"
    );
    assert!(
        !outs(&dir)[1].exists() && !outs(&dir)[2].exists(),
        "the shards it did not reach are absent"
    );
    assert!(
        partials(&dir).is_empty(),
        "the .partial it was writing is removed"
    );
    let (r, log) = run(&first, &dir.join("set"), true, &[]);
    let s = r.unwrap();
    assert_eq!((s.files.len(), s.kept), (2, 1), "the next resume goes on");
    assert!(
        log.contains(&format!("  shard 1/3: kept {}", outs(&dir)[0].display())),
        "the finished shard is the kept one"
    );
    let _ = std::fs::remove_dir_all(&work);
}
