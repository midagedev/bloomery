//! The EmbeddingGemma image gates' pinned inputs and CPU-twin comparisons.

use crate::e2e::rel;
use bloomery_gpu_gates::{GateError, RefManifest, sha256, verdict};
use refset::arch::gemma_embedding2::{MMPROJ_BF16, MMPROJ_BF16_SHA256, TWIN};
use refset::emb2::Emb2Set;
use refset::family::Family;
use refset::ik::ref_tensor_of_in;
use std::path::{Path, PathBuf};
use vision::Rgb8;

pub fn open(family: &Family) -> Result<RefManifest, GateError> {
    Ok(Emb2Set::open(&family.path(family.sets[0]), family)?.man)
}

pub fn row(man: &RefManifest, name: &str, shape: [u64; 4]) -> Result<Vec<f32>, GateError> {
    let r = man.tensor(name, 0)?;
    if r.ty != "f32" || r.ne != shape {
        return Err(format!("{name}: {} {:?}, expected f32 {shape:?}", r.ty, r.ne).into());
    }
    let data = ref_tensor_of_in(&man.dir, r)?;
    if data.iter().any(|v| !v.is_finite()) {
        return Err(format!("{name}: non-finite oracle value").into());
    }
    Ok(data)
}

pub fn compare(name: &str, ours: &[f32], card: &[f32], cpu: &[f32]) -> bool {
    let distance = rel(ours, card);
    let twin = rel(cpu, card);
    let band = TWIN * twin;
    let ok = !card.is_empty() && distance.is_finite() && twin.is_finite() && distance <= band;
    println!(
        "{name}: rel {distance:.9e}, cpu-twin {twin:.9e}, band {band:.9e} ({TWIN} x twin) {}",
        verdict(ok)
    );
    ok
}

pub fn projector(model: &Path) -> Result<PathBuf, GateError> {
    let path = model.with_file_name(
        Path::new(MMPROJ_BF16)
            .file_name()
            .ok_or("no projector filename")?,
    );
    let digest = sha256(&path)?;
    if digest != MMPROJ_BF16_SHA256 {
        return Err(format!(
            "{}: projector sha256 {digest}, expected {MMPROJ_BF16_SHA256}",
            path.display()
        )
        .into());
    }
    Ok(path)
}

/// Both sets must name the same image bytes; the file is checked before decoding.
pub fn image(
    card: &RefManifest,
    cpu: &RefManifest,
    name: &str,
) -> Result<(Rgb8, String), GateError> {
    let header = |man: &RefManifest| -> Result<(PathBuf, String), GateError> {
        let prefix = format!("# image\t{name}\t");
        let mut lines = man
            .header
            .other
            .iter()
            .filter_map(|l| l.strip_prefix(&prefix));
        let line = lines
            .next()
            .ok_or_else(|| format!("{name}: missing image identity"))?;
        if lines.next().is_some() {
            return Err(format!("{name}: duplicate image identity").into());
        }
        let f: Vec<_> = line.split('\t').collect();
        if f.len() < 2 {
            return Err(format!("{name}: malformed image identity").into());
        }
        Ok((PathBuf::from(f[0]), f[1].to_string()))
    };
    let (path, digest) = header(card)?;
    if header(cpu)?.1 != digest || sha256(&path)? != digest {
        return Err(format!("{name}: CPU/card image sha256 mismatch").into());
    }
    Ok((Rgb8::from_png(&std::fs::read(path)?)?, digest))
}
