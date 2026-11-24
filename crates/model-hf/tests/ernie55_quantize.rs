//! ERNIE 5.5 Flash Preview in `bloomery-quantize`: a checkpoint folder quantized to the GGUF its description opens
//! and a load can host, host only. The quantizer's plan, rules and writer are the crate's; the input is the folder as
//! `--hf` reads it (`quantize_source::open`: the split `prepare` writes). `E55Q_SOURCE=folder` takes it as a load opens
//! it instead ([`common::FileSource`]: every tensor in the type and the bytes the part source makes of the stored
//! planes, which are what `--hf` reads of the same folder, and none of a cut folder's fixture keys).
//!
//! Five clauses, on a toy folder of the three layer kinds with bytes in every stored format
//! ([`a_toy_ernie55_folder_quantizes_to_the_gguf_its_description_opens`], any machine) and on the fixture cut
//! (`hw_the_ernie55_cut_quantizes_to_the_gguf_its_description_opens`, the fixture root's `ernie55`, `E55Q_CUT` to
//! name another; `E55Q_OUT` names a path the GGUF is kept at):
//!
//! - (a) the types are the plan's for an ernie5_5 set under Q8_0: every FP8 128-block projection Q8_0, and the rest
//!   as the folder load makes it (the routed stacks MXFP4, the embedding, the head, the routers and the compressor
//!   BF16, the vectors F32), each in the reader's list for its stem, under the names and dims of the folder's header;
//! - (b) every expert tensor is byte for byte the folder loader's made tensor, and decodes, expert by expert, to the
//!   OCP MXFP4 decode of the stored planes written out in this file ([`common::ocp_mxfp4`]): the gate and up stacks
//!   are the checkpoint's `w1` and `w3`, the down stack its `w2`;
//! - (c) every Q8_0 tensor is ggml's reference Q8_0 over the FP8 values the loader makes (block for block), within
//!   the band the rule derives ([`common::q8_0_band`]);
//! - (d) the description reads the GGUF as it reads the folder: `Hparams` field by field, every tensor's name, dims
//!   and type, the spec, the KV layout, a plan on one 32 GB card, and a tokenizer that encodes alike;
//! - (e) the experts are the GGUF file's own mapped bytes, where the folder's are made at load (held whole on one
//!   card): a hosted load can keep them where the file maps them.
//!
//! A rule naming a copied type (`--tensor-type-file`, `\.weight$=fp8_b128`) keeps every FP8 128-block as the loader
//! makes it: the lossless GGUF, every tensor byte for byte the folder's ([`lossless`]; the toy's second run, and
//! `hw_the_ernie55_cut_keeps_its_fp8_weights_lossless` on the cut, `E55Q_OUT_FP8` naming where it is kept).
//!
//! `the_hf_source_reads_an_ernie55_folder` holds `--hf`'s input to the opened folder's, and
//! `a_cut_folders_fixture_keys_land_in_the_gguf` holds the GGUF written from a cut folder to carry the folder's
//! `bloomery.fixture.*` keys (so the fixture tier opens it as the fixture it is) and one written from any other
//! folder to carry none.

#[path = "common/ernie55_quantize.rs"]
mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use common::*;
use gguf::arch::ernie55::allowed_types;
use gguf::quant::GgmlType;
use gguf::{GgmlType as Ty, Place, Split};
use model::arch::ernie55::hparams::Hparams;
use model::arch::ernie55::place::PlanInputs;
use model::placement::accepts::{Program, routed_rule};
use model::placement::workstation::{self, CardSpec, MIB};
use model::placement::{self, PlanLevers};
use model_hf::ModelFile;
use quantize::cli::{Args, HfInput, Parsed, parse};
use quantize::run::Summary;
use quantize::source::Source as _;
use safetensors::Folder;

/// The arguments of a run from the folder `dir` to `out` under the file type Q8_0: a quantized source (the FP8
/// 128-blocks) is quantized again, so `--allow-requantize`; `rules` names a rules file when there is one.
fn args(dir: &Path, out: &Path, rules: Option<&Path>) -> Args {
    let mut a: Vec<String> = vec![
        "--hf".into(),
        dir.display().to_string(),
        "--allow-requantize".into(),
        "--cpu".into(),
    ];
    if let Some(r) = rules {
        a.extend(["--tensor-type-file".into(), r.display().to_string()]);
    }
    a.extend([out.display().to_string(), "Q8_0".into()]);
    let Parsed::Run(args) = parse(a).unwrap_or_else(|e| panic!("the command line: {e}")) else {
        panic!("not a run");
    };
    args
}

/// Quantize `dir` to `out` as `a` asks, the folder read through `--hf`'s own source, the split `prepare` writes
/// (`E55Q_SOURCE=folder`: opened as a load opens it instead); the GGUF's path and the run's summary.
fn quantize(dir: &Path, a: &Args) -> (PathBuf, Summary) {
    let source: Box<dyn quantize::source::Source + Send> =
        match std::env::var("E55Q_SOURCE").ok().as_deref() {
            None | Some("hf") => hf_source(dir),
            Some("folder") => Box::new(FileSource::folder(dir)),
            Some(other) => panic!("E55Q_SOURCE {other:?}: `hf` (unset) or `folder`"),
        };
    let mut lines = Vec::new();
    let s = quantize::run::prepare_from(a, source)
        .and_then(|prep| {
            quantize::run(a, prep, &mut quantize::host::HostBackend::new(), &mut |l| {
                lines.push(l.to_string())
            })
        })
        .unwrap_or_else(|e| panic!("{}: {e}\n{}", dir.display(), lines.join("\n")));
    (PathBuf::from(&a.out), s)
}

/// The folder `dir` as `--hf` reads it.
fn hf_source(dir: &Path) -> Box<dyn quantize::source::Source + Send> {
    model_hf::quantize_source::open(&HfInput {
        dir: dir.to_path_buf(),
        arch: None,
        drop: vec![],
        shard_gib: 48,
    })
    .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
}

/// The GGUF the plan gives the folder's tensor of type `ty`: an FP8 128-block projection takes the file type Q8_0,
/// every other tensor stays as the folder load makes it.
fn planned(ty: GgmlType) -> GgmlType {
    match ty {
        GgmlType::FP8_B128 => GgmlType::Q8_0,
        kept => kept,
    }
}

/// Clauses (a) to (e) on the GGUF at `gguf`, written from the folder `folder`. Returns how many Q8_0 and MXFP4 tensors
/// it holds.
fn prove(folder: &Path, gguf: &Path) -> (usize, usize) {
    let opened = ModelFile::open(folder).unwrap_or_else(|e| panic!("{}: {e}", folder.display()));
    let ModelFile::Folder(ref opened) = opened else {
        panic!("{}: a directory opens as a folder", folder.display());
    };
    let header = &opened.header;
    let _ = &opened;
    let file = ModelFile::open(gguf).unwrap_or_else(|e| panic!("{}: {e}", gguf.display()));
    let ModelFile::Gguf(ref split) = file else {
        panic!("{}: a file opens as a GGUF split", gguf.display());
    };
    let source = FileSource::folder(folder);
    fn tensor<'a>(split: &'a Split, name: &str) -> &'a gguf::TensorInfo {
        split
            .find(name)
            .map(|(_, t)| t)
            .unwrap_or_else(|| panic!("{name} is not a tensor of the split"))
    }

    // (a) the types are the plan's, under the folder's names and dims, each in the reader's list.
    // by name: the writer orders the tensors as ggml's driver walks them, the folder's header as the map lists them
    let row = |(_, t): (usize, &gguf::TensorInfo)| (t.name.clone(), (t.dims.clone(), t.ty));
    let theirs: BTreeMap<String, (Vec<u64>, GgmlType)> = header.iter_tensors().map(row).collect();
    let ours: BTreeMap<String, (Vec<u64>, GgmlType)> = split.iter_tensors().map(row).collect();
    assert_eq!(ours.len(), theirs.len(), "tensor counts");
    let (mut n_q8, mut n_mx, mut n_fp8) = (0, 0, 0);
    for (name, o) in &ours {
        let t = theirs
            .get(name)
            .unwrap_or_else(|| panic!("{name}: not a tensor of the folder's header"));
        assert_eq!(&o.0, &t.0, "{name}: the GGUF's dims against the folder's");
        let allowed = allowed_types(quantize::fit::stem_of(name))
            .unwrap_or_else(|| panic!("{name}: the reader has no list"));
        assert!(
            allowed.contains(&o.1),
            "{name} is {:?}; the reader takes {:?}",
            o.1,
            allowed
        );
        assert_eq!(o.1, planned(t.1), "{name}: the folder makes it {:?}", t.1);
        n_q8 += usize::from(o.1 == Ty::Q8_0);
        n_mx += usize::from(o.1 == Ty::MXFP4);
        n_fp8 += usize::from(o.1 == Ty::FP8_B128);
    }
    assert!(
        n_q8 > 0 && n_mx > 0,
        "no Q8_0 ({n_q8}) or no MXFP4 ({n_mx}) tensor in the GGUF"
    );
    println!(
        "clause (a) {} tensors: Q8_0 {n_q8}, MXFP4 {n_mx}, FP8_B128 {n_fp8}, the rest as the folder makes them: PASS",
        ours.len()
    );

    // (b) the experts: the loader's made bytes, and the OCP decode of the stored planes.
    let planes = Folder::open(folder).unwrap_or_else(|e| panic!("{}: {e}", folder.display()));
    let mut experts = 0usize;
    let mut values = 0u64;
    for (_, t) in split.iter_tensors().filter(|(_, t)| t.ty == Ty::MXFP4) {
        let &[k, rows, n] = t.dims.as_slice() else {
            panic!("{}: dims {:?} are not a stack's", t.name, t.dims);
        };
        let made = source.whole(&t.name);
        let written = split.data(&t.name).unwrap();
        assert_eq!(
            written, made,
            "{}: not the folder loader's made bytes",
            t.name
        );
        // `blk.L.ffn_{gate,up,down}_exps.weight` is made of `model.layers.L.block_sparse_moe.experts.E.w{1,3,2}`.
        let (l, which) = t
            .name
            .strip_prefix("blk.")
            .and_then(|r| r.split_once('.'))
            .unwrap_or_else(|| panic!("{}", t.name));
        let w = match which {
            "ffn_gate_exps.weight" => "w1",
            "ffn_up_exps.weight" => "w3",
            "ffn_down_exps.weight" => "w2",
            other => panic!("{other}: not an expert stack"),
        };
        let (k, rows, n) = (k as usize, rows as usize, n as usize);
        let per = rows * (k / 32) * 17;
        let mut v = vec![0.0f32; k];
        for e in 0..n {
            let name = format!("model.layers.{l}.block_sparse_moe.experts.{e}.{w}.weight");
            let weight = planes
                .raw(&name)
                .unwrap_or_else(|err| panic!("{name}: {err}"));
            let scale = planes.raw(&format!("{name}_scale_inv")).unwrap();
            let want = ocp_mxfp4(weight.bytes, scale.bytes, rows, k);
            let got = &written[e * per..(e + 1) * per];
            assert_eq!(got.len(), per, "{}: expert {e}", t.name);
            for (r, w) in want.chunks(k).enumerate() {
                gguf::quant::dequant_row(
                    Ty::MXFP4,
                    &got[r * (k / 32 * 17)..][..k / 32 * 17],
                    &mut v,
                )
                .unwrap();
                assert_eq!(
                    v, w,
                    "{}: expert {e} row {r} does not decode to the OCP value",
                    t.name
                );
            }
            experts += 1;
            values += (rows * k) as u64;
        }
    }
    println!(
        "clause (b) {experts} experts ({values} values) are the loader's bytes and decode to the OCP MXFP4 value: PASS"
    );

    // (c) the Q8_0 tensors: the reference rule over the loader's FP8 values, within the band.
    let mut worst = 0.0f32;
    for (_, t) in split.iter_tensors().filter(|(_, t)| t.ty == Ty::Q8_0) {
        let fp8 = tensor(header, &t.name);
        assert_eq!(
            fp8.ty,
            Ty::FP8_B128,
            "{}: the folder makes it {:?}",
            t.name,
            fp8.ty
        );
        let made = source.whole(&t.name);
        let w = q8_0_within_band(&made, split.data(&t.name).unwrap(), t.dims[0] as usize);
        worst = worst.max(w);
    }
    println!(
        "clause (c) {n_q8} Q8_0 tensors are the reference rule's bytes over the FP8 source, the largest error {worst:.3} of its band: PASS"
    );

    // (d) the description: the same hparams, tensors, spec, KV layout, plan and tokenizer.
    let (from_folder, from_gguf) = (
        Hparams::read(header).unwrap(),
        Hparams::read(split).unwrap(),
    );
    assert_eq!(from_gguf, from_folder, "the hparams the description reads");
    assert_eq!(
        described(split),
        described(header),
        "the tensors the description reads"
    );
    for (_, t) in split.iter_tensors().filter(|(_, t)| {
        quantize::fit::stem_of(&t.name).ends_with("_scale.weight")
            && quantize::fit::stem_of(&t.name).starts_with("hc_")
    }) {
        assert_eq!(
            split.data(&t.name).unwrap(),
            source.whole(&t.name),
            "{}: the joined scalars",
            t.name
        );
    }
    let (a, b) = (
        PlanInputs::describe(header).unwrap(),
        PlanInputs::describe(split).unwrap(),
    );
    assert_eq!(a.spec, b.spec, "the spec");
    assert_eq!(
        format!("{:?}", a.kv),
        format!("{:?}", b.kv),
        "the KV layout"
    );
    let rtx_5090 = CardSpec {
        name: "5090",
        total_bytes: 32_607 * MIB,
        driver_reserve_bytes: 550 * MIB,
        device: None,
        free_bytes: None,
        held_by: None,
        integrated: false,
        margin_bytes: None,
        granule_bytes: workstation::GRANULE,
    };
    let plan = |inputs: &PlanInputs| {
        let machine = workstation::plan_on(rtx_5090, inputs.model.layers);
        let plan = placement::plan_routed_reserving(
            &inputs.model,
            &machine,
            32_768,
            &inputs.kv,
            &PlanLevers::default(),
            routed_rule(Program::Ernie55Body),
            0,
        )
        .unwrap();
        (
            plan.cards[0].experts,
            plan.cards[0].kv_bytes,
            plan.host.expert_bytes,
            plan.cards[0].dense_bytes,
        )
    };
    // The experts and the KV are the same bytes (MXFP4 as the loader makes it, the KV by the keys); the card's dense
    // bytes are the file's types': Q8_0 is 34 bytes a 32 values where the folder's FP8 128-blocks are 132 a 128.
    let (pa, pb) = (plan(&a), plan(&b));
    assert_eq!(
        (pa.0, pa.1, pa.2),
        (pb.0, pb.1, pb.2),
        "the plan on one 32 GB card"
    );
    assert!(
        pb.3 > pa.3 && pb.3 - pa.3 < pa.3 / 16,
        "the GGUF's dense bytes {} beside the folder's {}",
        pb.3,
        pa.3
    );
    // the keys: the folder's header without the split's, a cut folder's fixture keys (what `--hf` reads; the folder
    // opened as a load opens it carries none), then the quantizer's two
    let (fk, gk) = (without_split(header.keys()), without_split(split.keys()));
    let fixture = if std::env::var("E55Q_SOURCE").as_deref() == Ok("folder") {
        Vec::new()
    } else {
        refset::fixture::folder_values(folder).unwrap_or_else(|e| panic!("{e}"))
    };
    assert!(
        gk.len() == fk.len() + fixture.len() + 2
            && gk
                .iter()
                .zip(fk.iter().chain(&fixture))
                .all(|(a, b)| a == b),
        "the GGUF's keys do not begin as the folder's header's and fixture keys"
    );
    let tail: Vec<&str> = gk[fk.len() + fixture.len()..]
        .iter()
        .map(|(k, _)| k.as_str())
        .collect();
    assert_eq!(tail, ["general.quantization_version", "general.file_type"]);
    assert_eq!(
        split.value("general.file_type"),
        Some(&gguf::Value::U32(7)),
        "Q8_0"
    );
    let (tf, tg) = (file.tokenizer().unwrap(), opened.tokenizer.clone());
    let text = "Hello world, 1234567 你好 <|ernie:im_start|>user\nhi<|ernie:im_end|>";
    assert_eq!(
        tg.encode(text, true, false),
        tf.encode(text, true, false),
        "the GGUF's tokenizer"
    );
    println!(
        "clause (d) the description reads the GGUF as the folder (hparams, {} tensors, spec, KV, plan on a 5090: card experts {}, kv {} B, host experts {} B, dense {} B against the folder's {} B, tokenizer): PASS",
        ours.len(),
        pb.0,
        pb.1,
        pb.2,
        pb.3,
        pa.3
    );

    // (e) in place: the experts are the file's mapped bytes where the folder makes them at load.
    let weights = file.weights().unwrap();
    for (_, t) in split.iter_tensors().filter(|(_, t)| t.ty == Ty::MXFP4) {
        assert_eq!(
            header.place(&t.name),
            Some(&Place::Made),
            "{}: the folder",
            t.name
        );
        let part = weights.bytes(&t.name).unwrap();
        let spans = part.spans();
        let mapped = split.data(&t.name).unwrap();
        assert_eq!(spans.len(), 1, "{}: {} spans", t.name, spans.len());
        assert!(
            std::ptr::eq(spans[0].as_ptr(), mapped.as_ptr()),
            "{}: a copy of the file's run",
            t.name
        );
    }
    println!(
        "clause (e) {n_mx} expert stacks are the GGUF's mapped bytes, made at load under the folder: PASS"
    );
    (n_q8, n_mx)
}

/// The toy folder quantized through the plan, the clauses (a) to (e) on its GGUF, and the FP8 kept by a rule.
#[test]
fn a_toy_ernie55_folder_quantizes_to_the_gguf_its_description_opens() {
    let work = work_dir("toy");
    let folder = work.join("ernie55-toy");
    write_toy_folder(&folder);
    let out = work.join("toy.gguf");
    let (gguf, s) = quantize(&folder, &args(&folder, &out, None));
    let (n_q8, n_mx) = prove(&folder, &gguf);
    assert_eq!((s.quantized, s.files.len()), (n_q8, 1));
    let stored: usize = toy_tensors().iter().map(|t| t.3.len()).sum();
    println!(
        "toy: {stored} B stored, {} B out, {n_q8} tensors Q8_0, {n_mx} MXFP4 stacks, the run {:.1} s",
        std::fs::metadata(&gguf).unwrap().len(),
        s.wall_s
    );

    // A rule naming a copied type keeps the FP8 128-blocks as the loader makes them.
    let rules = rule_file(&work);
    let (kept, _) = quantize(
        &folder,
        &args(&folder, &work.join("toy-fp8.gguf"), Some(&rules)),
    );
    assert_eq!(
        lossless(&folder, &kept),
        n_q8,
        "every tensor the file type wrote Q8_0 stays FP8_B128"
    );
    let _ = std::fs::remove_dir_all(&work);
}

/// The fixture cut (`E55Q_CUT`, else the fixture root's `ernie55`, the folder of three layers `fixture cut` writes; the
/// vendor's 43-layer folder is refused by name: that quantize is the lead's run). `E55Q_OUT` names where the GGUF is
/// kept, else a work directory's.
#[test]
#[ignore = "hw: the ernie55 fixture cut (default <fixture root>/ernie55, /models/fixtures/ernie55)"]
fn hw_the_ernie55_cut_quantizes_to_the_gguf_its_description_opens() {
    let folder = cut_folder();
    let work = work_dir("cut");
    let out = std::env::var_os("E55Q_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| work.join("ernie55-cut.gguf"));
    assert!(
        out.parent().is_some_and(Path::is_dir),
        "{}: the output's directory does not exist",
        out.display()
    );
    let (gguf, s) = quantize(&folder, &args(&folder, &out, None));
    let (n_q8, n_mx) = prove(&folder, &gguf);
    let planes = Folder::open(&folder).unwrap();
    let files = planes.headers().len();
    let stored: usize = planes
        .names()
        .map(|n| planes.raw(n).unwrap().bytes.len())
        .sum();
    println!(
        "the cut: {stored} B of tensor data in {files} shards, {n_q8} tensors Q8_0, {n_mx} MXFP4 stacks, {} B out, the run {:.1} s",
        std::fs::metadata(&gguf).unwrap().len(),
        s.wall_s
    );
    let _ = std::fs::remove_dir_all(&work);
}

/// The fixture cut under the rule that keeps every FP8 128-block as the loader makes it: the lossless GGUF, every
/// tensor byte for byte the folder's, for a load that holds the dense weights to the folder's bytes. `E55Q_OUT_FP8`
/// names where it is kept, else a work directory's.
#[test]
#[ignore = "hw: the ernie55 fixture cut (default <fixture root>/ernie55, /models/fixtures/ernie55)"]
fn hw_the_ernie55_cut_keeps_its_fp8_weights_lossless() {
    let folder = cut_folder();
    let work = work_dir("cut-fp8");
    let out = std::env::var_os("E55Q_OUT_FP8")
        .map(PathBuf::from)
        .unwrap_or_else(|| work.join("ernie55-cut-fp8.gguf"));
    assert!(
        out.parent().is_some_and(Path::is_dir),
        "{}: the output's directory does not exist",
        out.display()
    );
    let rules = rule_file(&work);
    let (gguf, s) = quantize(&folder, &args(&folder, &out, Some(&rules)));
    assert_eq!((s.quantized, s.files.len()), (0, 1), "nothing is converted");
    let fp8 = lossless(&folder, &gguf);
    let (file, opened) = (
        ModelFile::open(&gguf).unwrap(),
        ModelFile::open(&folder).unwrap(),
    );
    let (ModelFile::Gguf(ref split), ModelFile::Folder(ref opened)) = (file, opened) else {
        panic!("a GGUF split from a folder");
    };
    assert_eq!(
        (Hparams::read(split).unwrap(), described(split)),
        (
            Hparams::read(&opened.header).unwrap(),
            described(&opened.header)
        ),
        "the description reads it as the folder"
    );
    println!(
        "the cut, lossless: {fp8} tensors FP8_B128 and none Q8_0, every tensor the folder loader's bytes, {} B out, the run {:.1} s",
        std::fs::metadata(&gguf).unwrap().len(),
        s.wall_s
    );
    let _ = std::fs::remove_dir_all(&work);
}

/// The fixture cut's folder: `E55Q_CUT`, else the fixture root's `ernie55`; one of three layers.
fn cut_folder() -> PathBuf {
    let folder = std::env::var_os("E55Q_CUT")
        .filter(|d| !d.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let root = refset::fixture::root().unwrap_or_else(|_| "/models/fixtures".to_string());
            PathBuf::from(format!("{root}/ernie55"))
        });
    let layers = layers_of(&folder);
    assert_eq!(
        layers,
        LAYERS as u64,
        "{}: a folder of {layers} layers; this gate reads the cut of three, the vendor's folder is the lead's run",
        folder.display()
    );
    folder
}

/// A rules file keeping every `.weight` in FP8 128-blocks, written under `work`.
fn rule_file(work: &Path) -> PathBuf {
    let rules = work.join("keep-fp8.rules");
    std::fs::write(&rules, "\\.weight$=fp8_b128\n").unwrap();
    rules
}

/// The GGUF at `gguf`, written under [`rule_file`]'s rule, held to what the rule asks: no Q8_0 tensor, every tensor
/// byte for byte the folder loader's made tensor (the projections and the rest alike). Returns the count of FP8
/// 128-block tensors.
fn lossless(folder: &Path, gguf: &Path) -> usize {
    let file = ModelFile::open(gguf).unwrap();
    let ModelFile::Gguf(ref split) = file else {
        panic!("a GGUF split");
    };
    let source = FileSource::folder(folder);
    let mut fp8 = 0;
    for (_, t) in split.iter_tensors() {
        assert_ne!(t.ty, Ty::Q8_0, "{}: the rule keeps the stored type", t.name);
        assert_eq!(
            split.data(&t.name).unwrap(),
            source.whole(&t.name),
            "{}",
            t.name
        );
        fp8 += usize::from(t.ty == Ty::FP8_B128);
    }
    fp8
}

/// The layer count of the checkpoint folder `dir`'s `config.json`.
fn layers_of(dir: &Path) -> u64 {
    let config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("config.json"))
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display())),
    )
    .unwrap();
    config["num_hidden_layers"]
        .as_u64()
        .expect("num_hidden_layers")
}

/// `--hf` reads an ernie5_5_moe folder as the split `prepare` writes of it ([`model_hf::quantize_source::open`]):
/// the same header keys, tensors and bytes as the opened folder.
#[test]
fn the_hf_source_reads_an_ernie55_folder() {
    let work = work_dir("hf");
    let folder = work.join("ernie55-toy");
    write_toy_folder(&folder);
    let hf = hf_source(&folder);
    let opened = FileSource::folder(&folder);
    assert_eq!(without_split(hf.keys()), without_split(opened.keys()));
    let (a, b) = (hf.tensors(), opened.tensors());
    assert_eq!(a.len(), b.len());
    for ((_, a), (_, b)) in a.iter().zip(&b) {
        assert_eq!(
            (a.name.as_str(), a.dims.as_slice(), a.ty, a.nbytes),
            (b.name.as_str(), b.dims.as_slice(), b.ty, b.nbytes)
        );
        assert_eq!(
            hf.read(&a.name, 0..a.nbytes).unwrap(),
            opened.read(&b.name, 0..b.nbytes).unwrap(),
            "{}",
            a.name
        );
    }
    let _ = std::fs::remove_dir_all(&work);
}

/// The identity keys of the fixture cut `fixture cut` writes into a folder's first shard, as its metadata holds them.
const CUT_KEYS: [(&str, &str); 9] = [
    ("format", "pt"),
    ("bloomery.fixture.card_budget", "8589934592"),
    ("bloomery.fixture.dense_ff", "1024"),
    ("bloomery.fixture.expert_ff", "512"),
    ("bloomery.fixture.layers", "0,1,7"),
    (
        "bloomery.fixture.manifest_sha256",
        "99baab0354879952671fd041070a5b52e7475b6533de8043d2a166cb3942fa6f",
    ),
    (
        "bloomery.fixture.rev",
        "ace1100dcde7bf4eb06add6ae11ce4f43a11ddfe",
    ),
    ("bloomery.fixture.source", "baidu/ERNIE-5.5-Flash-Preview"),
    ("bloomery.fixture.version", "1"),
];

/// The shard at `shard` with `meta` as its `__metadata__`, its tensors untouched.
fn set_metadata(shard: &Path, meta: &[(&str, &str)]) {
    let bytes = std::fs::read(shard).unwrap();
    let n = usize::try_from(u64::from_le_bytes(bytes[..8].try_into().unwrap())).unwrap();
    let mut header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let meta: serde_json::Map<String, serde_json::Value> = meta
        .iter()
        .map(|(k, v)| ((*k).to_string(), serde_json::Value::from(*v)))
        .collect();
    header.insert("__metadata__".to_string(), meta.into());
    let mut text = serde_json::to_string(&header).unwrap();
    while !text.len().is_multiple_of(8) {
        text.push(' ');
    }
    let mut out = (text.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(text.as_bytes());
    out.extend_from_slice(&bytes[8 + n..]);
    std::fs::write(shard, out).unwrap();
}

/// The `bloomery.fixture.` keys of a GGUF's header, in file order.
fn fixture_keys(split: &Split) -> Vec<(String, gguf::Value)> {
    split
        .keys()
        .iter()
        .filter(|(k, _)| k.starts_with("bloomery.fixture."))
        .cloned()
        .collect()
}

/// A GGUF written from a cut folder carries the folder's `bloomery.fixture.*` keys, so the fixture tier opens it as the
/// fixture it is: exactly the folder's keys, the version a u32 and the card budget a u64 as the readers take them, no
/// subset key, the same `# fixture` line as the folder's first shard, and every other key the keyless run's. A folder
/// that holds none writes none.
#[test]
fn a_cut_folders_fixture_keys_land_in_the_gguf() {
    let work = work_dir("fxkeys");
    let folder = work.join("ernie55-toy");
    write_toy_folder(&folder);
    let open = |path: &Path| match ModelFile::open(path).unwrap() {
        ModelFile::Gguf(split) => split,
        ModelFile::Folder(_) => panic!("{}: a GGUF", path.display()),
    };

    let (plain, _) = quantize(&folder, &args(&folder, &work.join("plain.gguf"), None));
    let plain = open(&plain);
    assert!(
        fixture_keys(&plain).is_empty(),
        "a folder with no fixture key writes none"
    );

    let first = folder.join("model-00001-of-00002.safetensors");
    set_metadata(&first, &CUT_KEYS);
    let (cut, _) = quantize(&folder, &args(&folder, &work.join("cut.gguf"), None));
    let split = open(&cut);
    let carried = fixture_keys(&split);
    assert_eq!(
        carried,
        refset::fixture::folder_values(&folder).unwrap(),
        "the folder's keys"
    );
    assert_eq!(
        carried.len(),
        CUT_KEYS.len() - 1,
        "every fixture key but format"
    );
    assert_eq!(
        split.value("bloomery.fixture.version"),
        Some(&gguf::Value::U32(1))
    );
    assert_eq!(
        split
            .value("bloomery.fixture.card_budget")
            .and_then(gguf::Value::as_u64),
        Some(8_589_934_592),
        "the budget a planner reads"
    );
    assert!(
        split.value("bloomery.fixture.subset").is_none(),
        "the fixture tier refuses a file that carries the subset key"
    );
    assert_eq!(
        refset::fixture::keys(&cut).unwrap(),
        refset::fixture::keys(&first).unwrap(),
        "the GGUF's `# fixture` line is the folder's"
    );
    let rest = |s: &Split| -> Vec<(String, gguf::Value)> {
        without_split(s.keys())
            .into_iter()
            .filter(|(k, _)| !k.starts_with("bloomery.fixture."))
            .collect()
    };
    assert_eq!(rest(&split), rest(&plain), "no other key moves");
    let _ = std::fs::remove_dir_all(&work);
}

fn without_split(keys: &[(String, gguf::Value)]) -> Vec<(String, gguf::Value)> {
    keys.iter()
        .filter(|(k, _)| !k.starts_with("split."))
        .cloned()
        .collect()
}
