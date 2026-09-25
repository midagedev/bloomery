//! The stage-1 oracle set's manifest, `$BLOOMERY_DATA/ref/MANIFEST.tsv`, which
//! `tools/ref/dump.sh` wrote: found, complete, and made for the model the gates open. The
//! one reader of the file itself; `oracle` and `prompt` read their parts from it.
//!
//! The reference set is produced by the lead and only read here. If it is absent this
//! stops with an error naming the command — it never falls back to computing something,
//! because a gate that quietly measures nothing stays green.

use std::path::PathBuf;

/// The manifest's directory and text, after the checks every reader needs.
pub fn read() -> (PathBuf, String) {
    let base = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into());
    let dir = PathBuf::from(base).join("ref");
    let manifest = dir.join("MANIFEST.tsv");
    let text = std::fs::read_to_string(&manifest).unwrap_or_else(|e| {
        panic!(
            "no oracle at {} ({e}). The lead produces it: `just dump-ref`. \
             Do not run it yourself and do not skip this test.",
            manifest.display()
        )
    });
    // The dumper's last line is its completion proof. A manifest without it is from a
    // run that died, and the .f32 files beside it are then a mixture of two runs.
    // File count cannot detect that; the trailer can.
    if !text.lines().any(|l| l.starts_with("# complete\t")) {
        panic!(
            "the oracle at {} has no completion trailer — the dump that wrote it did \
             not finish, so the tensors beside it may be from two different runs. \
             The lead regenerates it: `just dump-ref`. Do not run dump_ref yourself \
             and do not skip this test.",
            manifest.display()
        );
    }
    // The gates must open the file the oracle came from.
    let model = text
        .lines()
        .find_map(|l| l.strip_prefix("# model\t"))
        .map(str::trim)
        .unwrap_or_else(|| panic!("the oracle at {} names no model", manifest.display()));
    let gated = super::model_path::model_path();
    let same = |a: &str, b: &str| match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    };
    if !same(model, &gated) {
        panic!(
            "the oracle at {} was dumped from {model}, the gates open {gated} \
             ($BLOOMERY_MODEL): a reference of another file is not a reference.",
            manifest.display()
        );
    }
    (dir, text)
}
