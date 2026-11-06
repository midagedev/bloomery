//! `prepare` on a toy Kimi K3 and K3.1 folder: the split of a whole folder (the BF16 projections
//! and MXFP4 stacks the checkpoint stores, opened through the registry's reader), of a K3.1 folder
//! written in K3's shape (`--arch kimi-k3 --drop 'blk.*.indexer.*'`) and of its indexer alone
//! (`--only`), their tensor counts and sets, and the refusals of the flags.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

#[path = "common/kimi_toy.rs"]
mod kimi_toy;

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("prepare-kimi-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `prepare` run with `args`.
fn prepare(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_prepare"))
        .env_remove("BLOOMERY_PREPARE_Q8")
        .args(args)
        .output()
        .unwrap()
}

fn text(o: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

/// The split written at `prefix`: its architecture word, its tensor names and its keys.
struct Written {
    arch: String,
    names: BTreeSet<String>,
    keys: Vec<String>,
}

fn written(prefix: &Path) -> Written {
    let first = glob_first(prefix);
    let split = gguf::Split::open(&first).unwrap();
    Written {
        arch: split.architecture().unwrap().to_string(),
        names: split.iter_tensors().map(|(_, t)| t.name.clone()).collect(),
        keys: split.iter_kv().map(|(k, _)| k.to_string()).collect(),
    }
}

/// The first shard of the split at `prefix`.
fn glob_first(prefix: &Path) -> PathBuf {
    let dir = prefix.parent().unwrap();
    let stem = prefix.file_name().unwrap().to_str().unwrap();
    let mut firsts: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with(&format!("{stem}-00001-of-"))
        })
        .collect();
    assert_eq!(firsts.len(), 1, "{firsts:?}");
    firsts.remove(0)
}

fn write_split(folder: &Path, out: &Path, flags: &[&str]) -> Output {
    let mut args = vec![folder.to_str().unwrap(), out.to_str().unwrap()];
    args.extend(flags);
    prepare(&args)
}

/// The whole split of a toy folder opened as the server opens it: the description `model::arch::spec`
/// reads from its header.
fn description(prefix: &Path) -> model::arch::Read {
    let split = gguf::Split::open(glob_first(prefix)).unwrap();
    model::arch::spec(&split).unwrap_or_else(|e| panic!("{}: {e}", prefix.display()))
}

/// A K3.1 folder written whole holds its indexer under its own word, and a K3.1 folder written in
/// K3's shape (`--arch kimi-k3 --drop 'blk.*.indexer.*'`) holds exactly the tensors a K3 folder's
/// split holds, with K3's header and no indexer key; its indexer alone is the five tensors it
/// leaves out. Each whole write passes `prepare`'s own open of its split, and the K3-shaped
/// split opens through `model::arch::spec` here: the K3.1 one with the selector on the latent
/// layer alone, the K3 ones with none. Counts: 107 + 5 = 112 at the toy's one latent layer (the
/// real folder has 24, so 120).
#[test]
fn a_k31_folder_split_in_k3s_shape_and_its_indexer_alone_are_complements() {
    let root = scratch("complements");
    let (k3, k31) = (root.join("k3"), root.join("k31"));
    kimi_toy::write_folder(&k3, false, None);
    kimi_toy::write_folder(&k31, true, None);
    let run = |folder: &Path, tag: &str, flags: &[&str]| {
        let out = root.join(tag);
        let o = write_split(folder, &out, flags);
        assert!(o.status.success(), "{tag}: {}", text(&o));
        (written(&out), out, text(&o))
    };
    let (k3_whole, k3_prefix, k3_log) = run(&k3, "k3-whole", &[]);
    let (whole, whole_prefix, whole_log) = run(&k31, "k31-whole", &[]);
    let (shaped, shaped_prefix, _) = run(
        &k31,
        "k31-as-k3",
        &["--arch", "kimi-k3", "--drop", "blk.*.indexer.*"],
    );
    let (indexer, _, _) = run(&k31, "k31-indexer", &["--only", "blk.*.indexer.*"]);

    // A whole write opens its split as the server does, or `prepare` fails.
    assert!(
        k3_log.contains("prepare: the split opens: 4 layers, 107 tensors"),
        "{k3_log}"
    );
    assert!(
        whole_log.contains("prepare: the split opens: 4 layers, 112 tensors"),
        "{whole_log}"
    );
    assert_eq!(k3_whole.names.len(), 107);
    assert_eq!(whole.names.len(), 112);
    assert_eq!(indexer.names.len(), 5);
    assert_eq!(shaped.names, k3_whole.names);
    let latent = kimi_toy::LATENT_LAYER;
    let want: BTreeSet<String> = ["attn_q_b", "attn_k", "proj", "k_norm"]
        .iter()
        .map(|s| format!("blk.{latent}.indexer.{s}.weight"))
        .chain([format!("blk.{latent}.indexer.k_norm.bias")])
        .collect();
    assert_eq!(indexer.names, want);
    let union: BTreeSet<String> = shaped.names.union(&indexer.names).cloned().collect();
    assert_eq!(union, whole.names);
    assert!(shaped.names.is_disjoint(&indexer.names));

    assert_eq!(
        (k3_whole.arch.as_str(), shaped.arch.as_str()),
        ("kimi-k3", "kimi-k3")
    );
    assert_eq!(whole.arch, "kimi-k31");
    let has_indexer_key = |w: &Written| {
        w.keys.iter().any(|k| {
            k.contains(".attention.indexer.") || k.ends_with(".attention.layer_norm_epsilon")
        })
    };
    assert!(!has_indexer_key(&k3_whole) && !has_indexer_key(&shaped));
    assert!(has_indexer_key(&whole));

    // The three splits open through the registry's reader: one architecture, the selector on the
    // K3.1 file's latent layer alone.
    let selects = |prefix: &Path| -> Vec<bool> {
        description(prefix)
            .spec
            .layers
            .iter()
            .map(
                |l| matches!(&l.mixer, model::arch::models::Mixer::Latent(a) if a.select.is_some()),
            )
            .collect()
    };
    let at_latent = |l: usize| (0..4).map(|i| i == l).collect::<Vec<bool>>();
    assert_eq!(selects(&whole_prefix), at_latent(latent));
    assert_eq!(selects(&k3_prefix), [false; 4]);
    assert_eq!(selects(&shaped_prefix), [false; 4]);
    let arch = |prefix: &Path| description(prefix).spec.arch;
    assert_eq!(arch(&whole_prefix), model::arch::models::Arch::KimiK3);
    assert_eq!(arch(&shaped_prefix), model::arch::models::Arch::KimiK3);
    std::fs::remove_dir_all(&root).unwrap();
}

/// The flags' refusals, each by name and each before a shard is written: a pattern that matches
/// nothing, both selections at once, an architecture word the map has none of or a K3 folder
/// has no indexer for, a K3.1 folder under K3's word with its indexer tensors kept, and `--plan`
/// with a flag that writes.
#[test]
fn the_selection_flags_are_refused_by_name() {
    let root = scratch("refusals");
    let (k3, k31) = (root.join("k3"), root.join("k31"));
    kimi_toy::write_folder(&k3, false, None);
    kimi_toy::write_folder(&k31, true, None);
    let refused = |folder: &Path, flags: &[&str]| -> String {
        let out = root.join("out");
        let o = write_split(folder, &out, flags);
        assert!(!o.status.success(), "{flags:?}: {}", text(&o));
        assert!(
            std::fs::read_dir(&root).unwrap().all(|e| {
                let n = e.unwrap().file_name().to_string_lossy().into_owned();
                !n.starts_with("out-")
            }),
            "{flags:?} left a shard"
        );
        text(&o)
    };
    let e = refused(&k31, &["--drop", "blk.*.nothing.*"]);
    assert!(e.contains("--drop") && e.contains("blk.*.nothing.*"), "{e}");
    let e = refused(&k31, &["--only", "token_embd.weight", "--only", "no.such"]);
    assert!(e.contains("--only") && e.contains("no.such"), "{e}");
    let e = refused(&k31, &["--drop", "blk.*", "--only", "blk.*"]);
    assert!(e.contains("both"), "{e}");
    let e = refused(&k31, &["--arch", "llama"]);
    assert!(e.contains("--arch") && e.contains("kimi-k31"), "{e}");
    let e = refused(&k3, &["--arch", "kimi-k31"]);
    assert!(e.contains("a K3 folder holds no indexer"), "{e}");
    let e = refused(&k31, &["--arch", "kimi-k3"]);
    assert!(
        e.contains(&format!("blk.{}.indexer.", kimi_toy::LATENT_LAYER)),
        "{e}"
    );
    assert!(e.contains("--drop 'blk.*.indexer.*'"), "{e}");
    let o = prepare(&[k31.to_str().unwrap(), "--plan", "--drop", "blk.*"]);
    assert!(
        !o.status.success() && text(&o).contains("takes no --shard-gib, --drop"),
        "{}",
        text(&o)
    );
    std::fs::remove_dir_all(&root).unwrap();
}
