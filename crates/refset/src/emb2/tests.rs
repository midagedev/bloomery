use super::{Kind, LCPP_BUILD, Profile};
use crate::RefError;
use crate::family::Family;
use std::path::{Path, PathBuf};

/// A fresh directory for one test's set; tests of one process run in
/// parallel, so each passes its own `what`.
pub(crate) fn set_dir(what: &str) -> Result<PathBuf, RefError> {
    Ok(crate::clefvis::testkit::scratch(&format!(
        "embedding-{what}"
    )))
}

pub(crate) fn remove(dir: &Path) -> Result<(), RefError> {
    std::fs::remove_dir_all(dir).map_err(|e| RefError::missing(dir, e.to_string()))
}

/// The header of a `kind` set as tools/ref/emb2 writes it, one line a
/// `(key, value)` pair; the arguments of `with` replace or (with an empty
/// value) drop the line of their key.
pub(crate) fn header(
    profile: &Profile,
    arch: &str,
    kind: Kind,
    with: &[(&str, &str)],
) -> Vec<(String, String)> {
    let device = format!(
        "{}\tthreads\t{}\tcard\t{}",
        kind.device(),
        if matches!(kind, Kind::Server | Kind::ServerCpu) {
            "-"
        } else {
            "16"
        },
        if kind.device() == "cpu" {
            "cpu"
        } else {
            "NVIDIA GB10"
        }
    );
    let tower = matches!(kind, Kind::Tower | Kind::TowerCpu);
    let mut h: Vec<(&str, String)> = vec![
        ("build", LCPP_BUILD.into()),
        (
            "arch",
            if tower {
                crate::clefvis::TOWER_ARCH
            } else {
                arch
            }
            .into(),
        ),
        ("emb2", kind.set_kind().into()),
        ("device", device),
    ];
    if tower {
        let (mmproj, _) = profile.mmproj.expect("a tower fixture has a projector");
        h.push(("model", mmproj.into()));
        h.push((
            "text_model",
            format!("{}\tsha256\t{}", profile.model, profile.model_sha256),
        ));
    } else {
        h.push(("model", profile.model.into()));
        h.push(("model_sha256", profile.model_sha256.into()));
    }
    if !matches!(kind, Kind::Text | Kind::TextCpu)
        && let Some((path, digest)) = profile.mmproj
    {
        h.push(("mmproj", format!("{path}\tsha256\t{digest}")));
    }
    for &(k, v) in with {
        h.retain(|&(key, _)| key != k);
        if !v.is_empty() {
            h.push((k, v.to_string()));
        }
    }
    h.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// `family`'s check of a set at `dir` with these header lines, one tensor row
/// and the trailer when `complete`.
pub(crate) fn check(
    dir: &Path,
    family: &Family,
    header: &[(String, String)],
    complete: bool,
) -> Result<String, RefError> {
    let mut lines =
        vec!["# dump_emb2: llama.cpp mainline EmbeddingGemma, raw f32, little-endian".to_string()];
    lines.extend(header.iter().map(|(k, v)| format!("# {k}\t{v}")));
    lines.push(
        "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1"
            .to_string(),
    );
    let name = if matches!(
        family.identity,
        crate::family::Identity::Emb2(Kind::Tower | Kind::TowerCpu, _)
    ) {
        "small/layer_out-0"
    } else {
        "en_short/pooled"
    };
    lines.push(format!(
        "tensor\t{name}\t0\tf32\t768\t1\t1\t1\t3072\t0\tMEAN\t1\t0\t-\t-"
    ));
    if complete {
        lines.push("# complete\t1\t0".to_string());
    }
    let path = dir.join("MANIFEST.tsv");
    std::fs::write(&path, {
        let mut out = lines.join("\n");
        out.push('\n');
        out
    })
    .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    family.check_set(dir).map(|p| p.dumped_from)
}
