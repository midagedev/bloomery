//! Plan equality: for each case, `llama-quantize --dry-run` over the input's headers (its own plan of every tensor's
//! type and size) against the plan `bloomery-quantize` resolves from the same rules and options, tensor for tensor in
//! both directions (`Plan::build` refuses a tensor the two type or size differently, a tensor either lacks, and a
//! dimension that differs). The cases are the K3 fixture and the K3.1 type maps over the K3-shaped set. Run by
//! `just gate-quantize-plan` (the box, no card, no data read).

mod common;

use quantize::cli::{Parsed, parse};
use std::path::{Path, PathBuf};
use std::process::Command;

fn env_path(name: &str, default: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| default.to_string()))
}

struct Case<'a> {
    tag: &'a str,
    input: &'a Path,
    rules: &'a str,
    ftype: &'a str,
    token_embedding: &'a str,
    keep_split: bool,
}

/// llama-quantize's dry-run plan as the table `gguf-splice.py plan` writes, then ours against it; the tensor count.
fn equal(lq: &Path, dir: &Path, c: &Case<'_>) -> usize {
    let rules = dir.join(format!("rules-{}.txt", c.tag));
    std::fs::write(&rules, c.rules).unwrap();
    let flags = [
        "--allow-requantize",
        "--output-tensor-type",
        "q6_k",
        "--token-embedding-type",
        c.token_embedding,
    ];
    let out = dir.join(format!("plan-out-{}", c.tag));
    let mut args: Vec<String> = vec!["--dry-run".into()];
    args.extend(flags.map(String::from));
    args.extend(["--tensor-type-file".into(), rules.display().to_string()]);
    if c.keep_split {
        args.push("--keep-split".into());
    }
    args.extend([
        c.input.display().to_string(),
        out.display().to_string(),
        c.ftype.into(),
        "1".into(),
    ]);
    let o = Command::new(lq)
        .args(&args)
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    assert!(
        o.status.success(),
        "{}: llama-quantize --dry-run ended {}",
        c.tag,
        o.status
    );
    let log = dir.join(format!("plan-{}.log", c.tag));
    let mut text = o.stderr;
    text.extend(o.stdout);
    std::fs::write(&log, text).unwrap();
    let tsv = dir.join(format!("plan-{}.tsv", c.tag));
    let st = Command::new("python3")
        .args([
            "-I",
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../tools/ref/gguf-splice.py"
            ),
            "plan",
            "--log",
        ])
        .arg(&log)
        .arg("--out")
        .arg(&tsv)
        .output()
        .unwrap();
    assert!(
        st.status.success(),
        "{}: gguf-splice plan: {}",
        c.tag,
        String::from_utf8_lossy(&st.stderr)
    );
    let rows = std::fs::read_to_string(&tsv).unwrap().lines().count();
    let mut ours: Vec<String> = vec![
        "--plan".into(),
        "--llama-plan".into(),
        tsv.display().to_string(),
        "--tensor-type-file".into(),
        rules.display().to_string(),
    ];
    ours.extend(flags.map(String::from));
    if c.keep_split {
        ours.push("--keep-split".into());
    }
    ours.extend([
        c.input.display().to_string(),
        out.display().to_string(),
        c.ftype.into(),
    ]);
    let Parsed::Run(a) = parse(ours).unwrap() else {
        panic!("not a run");
    };
    let prep = quantize::run::prepare(&a).unwrap_or_else(|e| panic!("{}: {e}", c.tag));
    assert_eq!(prep.plan.items.len(), rows, "{}: tensor counts", c.tag);
    println!(
        "{}: plan equal, {rows} tensors, {:.2} GB out ({} {} --keep-split={})",
        c.tag,
        prep.plan.bytes_out as f64 / 1e9,
        c.ftype,
        c.token_embedding,
        c.keep_split
    );
    rows
}

#[test]
#[ignore = "hw: the box, llama-quantize and the K3 fixture"]
fn hw_plan_equals_llama_quantize_dry_run() {
    let lq = env_path("K3Q_TREE", "").join("build/bin/llama-quantize");
    assert!(lq.is_file(), "no {} (K3Q_TREE)", lq.display());
    let dir = common::scratch("plan");
    let fixture = env_path(
        "K3Q_FIXTURE",
        "/models/fixtures/kimik3/kimi-k3-fixture-00001-of-00001.gguf",
    );
    assert!(fixture.is_file(), "no {} (K3Q_FIXTURE)", fixture.display());
    let src = common::fixture_src(&lq, &fixture, &dir);
    equal(
        &lq,
        &dir,
        &Case {
            tag: "fixture",
            input: &src,
            rules: &common::k31_rules("q4_k"),
            ftype: "Q4_K_M",
            token_embedding: "q4_k",
            keep_split: false,
        },
    );
    let shape = env_path("K3Q_PLAN_IN", "");
    let tight_rules = env_path("K3Q_PLAN_RULES", "");
    if shape.is_file() && tight_rules.is_file() {
        let tight = std::fs::read_to_string(&tight_rules).unwrap();
        let q5 = tight.replace("weight$=q4_k", "weight$=q5_k");
        let mut ud = tight.replace("weight$=q4_k", "weight$=q6_k");
        ud.push_str("^(token_embd|output)\\.weight$=q6_k\n");
        for (tag, rules, ftype, tok) in [
            ("k31-tight", tight.as_str(), "Q4_K_M", "q4_k"),
            ("k31-trunk-q5", q5.as_str(), "Q5_K_M", "q4_k"),
            ("k31-ud-shaped", ud.as_str(), "IQ1_S", "q6_k"),
        ] {
            equal(
                &lq,
                &dir,
                &Case {
                    tag,
                    input: &shape,
                    rules,
                    ftype,
                    token_embedding: tok,
                    keep_split: true,
                },
            );
        }
    } else {
        println!(
            "k31 cases skipped: no {} or {} (K3Q_PLAN_IN names the K3-shaped set's first shard, K3Q_PLAN_RULES the tight map's rules)",
            shape.display(),
            tight_rules.display()
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
