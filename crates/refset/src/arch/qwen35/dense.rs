//! The dense Qwen3.8-27B family (`logits-qwen38d`). The file is `qwen35` in the GGUF, the dense architecture Clef's
//! backbone shares, so the family is one more row of that architecture's table with a model file of its own. A set is
//! llama.cpp mainline's logits after a prompt and after each id forced behind it, written from [`MODEL`] by the mainline
//! tree [`super::LCPP_BUILD`] names: `tools/ref/dense_logits_ref.cpp` through `tools/ref/dense-logits.py`
//! (`just dump-logits-qwen38d`, `tools/ref/models/qwen38d.sh`). The prompt is the first ids of the qwen35 prose corpus
//! and the forced ids the ones after them. A set is a node dump's `MANIFEST.tsv` ([`RefManifest`]) with one tensor row:
//!
//! ```text
//! # model    /models/Qwen3.8-27B/Qwen3.8-27B-Q4_K_M.gguf
//! # build    53ed051ce
//! # arch     qwen35
//! # rows     32            # vocab  248320
//! # prompt_sha256  …       # forced_sha256  …       # raw_sha256  …
//! # kind  name  occurrence  type  ne0  ne1  ne2  ne3  bytes  sum  op  contig  logical  src0  src1
//! tensor  logits  0  f32  248320  32  1  1  31784960  <sum>  FORCED_LOGITS  1  0  -  -
//! # complete  1  0
//! ```
//!
//! `logits.0.f32` is `rows` rows of `vocab` f32, little-endian: row 0 follows the whole prompt and row `i` follows forced
//! id `i - 1` (the last forced id is not fed). `prompt.ids` and `forced.ids` hold the ids, one decimal a line.
//! [`ForcedLogits::open`] checks the set against the family, then each file against the manifest's digests and shape. The
//! set's CPU twin (`<set>.cpu`, mainline with no card in view) is no family's set: it is the oracle's own floor.
//! [`ForcedLogits::open_cpu`] reads it with the same checks, and each opener refuses the other's `# backend`.

use super::{ARCH, LCPP_BUILD};
use crate::RefError;
use crate::family::{Build, Family, Identity};
use crate::ik::{RefManifest, ref_tensor_of_in};
use sha2::{Digest, Sha256};
use std::path::Path;

/// The file every `logits-qwen38d` set is dumped from: lmstudio-community's Qwen3.8-27B Q4_K_M
/// (`tools/ref/models/qwen38d.sh`).
pub const MODEL: &str = "/models/Qwen3.8-27B/Qwen3.8-27B-Q4_K_M.gguf";

/// The first 64 and 4,096 ids of the prose corpus as the prompt, each with the 32 ids after them forced.
pub const P64: &str = "ref_qwen38d_logits_p64";
pub const P4096: &str = "ref_qwen38d_logits_p4096";

/// The sets, shortest first.
pub const SETS: &[&str] = &[P64, P4096];

/// The `op` of a set's one tensor row: the rows are no node of any engine's graph.
pub const OP: &str = "FORCED_LOGITS";

/// [`MODEL`], as a family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// llama.cpp mainline's logits rows of the 27B Q4_K_M file.
pub static LOGITS: Family = Family {
    name: "logits-qwen38d",
    sets: SETS,
    resolve: None,
    recipe: "just dump-logits-qwen38d",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &[],
};

/// A set of [`LOGITS`], opened and checked: the ids it was fed and its rows.
#[derive(Debug)]
pub struct ForcedLogits {
    /// The prompt's ids.
    pub prompt: Vec<u32>,
    /// The forced ids; row `i` follows the prompt and `forced[..i]`, and there is one row a forced id.
    pub forced: Vec<u32>,
    /// The width of a row.
    pub vocab: usize,
    rows: Vec<f32>,
}

impl ForcedLogits {
    /// The set at `dir` read and checked: the family's check ([`RefManifest::open`]: the trailer, the model file, the
    /// architecture and the build), the reference engine's CUDA rows (`# engine`, `# backend`: the CPU twin is refused),
    /// the one tensor row against `# rows` and `# vocab`, each ids file and the rows file against the digest the manifest
    /// states, every id inside the vocabulary and one forced id a row.
    pub fn open(dir: &Path) -> Result<ForcedLogits, RefError> {
        Self::open_backend(dir, "cuda")
    }

    /// The CPU twin at `dir` (`<set>.cpu`) read with [`ForcedLogits::open`]'s checks, the `# backend` being `cpu`: a
    /// CUDA set is refused.
    pub fn open_cpu(dir: &Path) -> Result<ForcedLogits, RefError> {
        Self::open_backend(dir, "cpu")
    }

    /// The set at `dir` whose manifest states the reference engine on `backend`, else as [`ForcedLogits::open`].
    fn open_backend(dir: &Path, backend: &str) -> Result<ForcedLogits, RefError> {
        let man = RefManifest::open(dir, &LOGITS)?;
        let at = dir.join("MANIFEST.tsv");
        for (key, want) in [("engine", "reference"), ("backend", backend)] {
            let got = stated::<String>(&man, key)?;
            if got != want {
                return Err(RefError::malformed(
                    at.display(),
                    format!("# {key} {got:?}, want {want:?}"),
                ));
            }
        }
        let (rows, vocab): (usize, usize) = (stated(&man, "rows")?, stated(&man, "vocab")?);
        let row = man.tensor("logits", 0)?;
        row.expect("logits rows", "f32", [vocab as u64, rows as u64, 1, 1], OP)?;
        let raw = dir.join(row.file_name());
        let raw_sha: String = stated(&man, "raw_sha256")?;
        let got = digest(&raw)?;
        if got != raw_sha {
            return Err(RefError::malformed(
                raw.display(),
                format!("sha256 {got}, the manifest says {raw_sha}"),
            ));
        }
        let prompt = ids(
            dir,
            "prompt.ids",
            &stated::<String>(&man, "prompt_sha256")?,
            vocab,
        )?;
        let forced = ids(
            dir,
            "forced.ids",
            &stated::<String>(&man, "forced_sha256")?,
            vocab,
        )?;
        if forced.len() != rows {
            return Err(RefError::malformed(
                at.display(),
                format!("{} forced ids for {rows} rows", forced.len()),
            ));
        }
        Ok(ForcedLogits {
            prompt,
            forced,
            vocab,
            rows: ref_tensor_of_in(dir, row)?,
        })
    }

    /// How many rows the set holds.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.forced.len()
    }

    /// Row `i`: the logits after the prompt and `forced[..i]`.
    #[must_use]
    pub fn row(&self, i: usize) -> &[f32] {
        &self.rows[i * self.vocab..(i + 1) * self.vocab]
    }
}

/// The `# key` line of `man`, parsed as `T`; the line missing or not a `T` is `Malformed`.
fn stated<T: std::str::FromStr>(man: &RefManifest, key: &str) -> Result<T, RefError> {
    let at = man.dir.join("MANIFEST.tsv");
    let prefix = format!("# {key}\t");
    let value = man
        .header
        .other
        .iter()
        .find_map(|l| l.strip_prefix(prefix.as_str()))
        .ok_or_else(|| RefError::malformed(at.display(), format!("no `# {key}` line")))?;
    value.parse().map_err(|_| {
        RefError::malformed(at.display(), format!("`# {key}` {value:?} does not parse"))
    })
}

/// The sha256 of `bytes`, as lowercase hex.
fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The sha256 of the file at `path`, as lowercase hex.
fn digest(path: &Path) -> Result<String, RefError> {
    let bytes = std::fs::read(path)
        .map_err(|e| RefError::missing(path, format!("{}: {e}", path.display())))?;
    Ok(sha(&bytes))
}

/// `name`, one decimal id a line, against the digest the manifest states; every id inside `vocab`.
fn ids(dir: &Path, name: &str, sha256: &str, vocab: usize) -> Result<Vec<u32>, RefError> {
    let path = dir.join(name);
    let got = digest(&path)?;
    if got != sha256 {
        return Err(RefError::malformed(
            path.display(),
            format!("sha256 {got}, the manifest says {sha256}"),
        ));
    }
    let text = std::fs::read_to_string(&path)
        .map_err(|e| RefError::missing(&path, format!("{}: {e}", path.display())))?;
    let ids = text
        .lines()
        .enumerate()
        .map(|(i, line)| {
            line.parse::<u32>()
                .ok()
                .filter(|&id| (id as usize) < vocab)
                .ok_or_else(|| {
                    RefError::malformed(
                        format!("{}:{}", path.display(), i + 1),
                        format!("{line:?} is not an id inside the vocabulary of {vocab}"),
                    )
                })
        })
        .collect::<Result<Vec<u32>, RefError>>()?;
    if ids.is_empty() {
        return Err(RefError::malformed(path.display(), "no ids"));
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::{ForcedLogits, LOGITS, MODEL, OP, sha};
    use crate::RefError;
    use crate::arch::qwen35::{ARCH, HIDDEN, LCPP_BUILD};
    use std::path::{Path, PathBuf};

    const VOCAB: usize = 4;
    const PROMPT: [u32; 3] = [1, 2, 3];
    const FORCED: [u32; 2] = [0, 1];
    const ROWS: [f32; 8] = [0.5, 3.0, -1.0, 0.25, 1.0, 0.0, 2.0, -2.0];

    fn ids_text(ids: &[u32]) -> String {
        ids.iter().map(|i| format!("{i}\n")).collect()
    }

    fn rows_bytes() -> Vec<u8> {
        ROWS.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// A set of dense-logits.py's shape at a fresh directory: the writer's header lines for these values, its
    /// one tensor row and the three files, the trailer last when `complete`.
    fn write_set(name: &str, model: &str, build: &str, arch: &str, complete: bool) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "bloomery-refset-qwen38d-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        let raw = rows_bytes();
        std::fs::write(dir.join("logits.0.f32"), &raw).expect("rows");
        std::fs::write(dir.join("prompt.ids"), ids_text(&PROMPT)).expect("prompt");
        std::fs::write(dir.join("forced.ids"), ids_text(&FORCED)).expect("forced");
        let sum: f64 = ROWS.iter().map(|&v| f64::from(v)).sum();
        let mut lines = vec![
            "# dense_logits_ref - llama.cpp mainline logits after the prompt and after each forced id".to_string(),
            "# format\tbloomery-forced-logits-v2".to_string(),
            format!("# model\t{model}"),
            format!("# build\t{build}"),
            format!("# arch\t{arch}"),
            "# engine\treference".to_string(),
            "# backend\tcuda".to_string(),
            "# schedule\tbatch".to_string(),
            format!("# prompt_sha256\t{}", sha(ids_text(&PROMPT).as_bytes())),
            format!("# forced_sha256\t{}", sha(ids_text(&FORCED).as_bytes())),
            "# rows\t2".to_string(),
            format!("# vocab\t{VOCAB}"),
            format!("# raw_sha256\t{}", sha(&raw)),
            "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1"
                .to_string(),
            format!("tensor\tlogits\t0\tf32\t{VOCAB}\t2\t1\t1\t{}\t{sum:?}\t{OP}\t1\t0\t-\t-", raw.len()),
        ];
        if complete {
            lines.push("# complete\t1\t0".to_string());
        }
        std::fs::write(dir.join("MANIFEST.tsv"), lines.join("\n") + "\n").expect("manifest");
        dir
    }

    fn refused(dir: &Path) -> String {
        ForcedLogits::open(dir).expect_err("refused").to_string()
    }

    /// The family takes a complete set of the 27B file, the pinned mainline commit and this architecture, and refuses
    /// by name another model file (Clef's, which shares the architecture and the build), an ik build, the MoE
    /// architecture, and a set without its trailer.
    #[test]
    fn the_family_refuses_a_set_of_another_file_build_or_arch() -> Result<(), RefError> {
        let dir = write_set("family", MODEL, LCPP_BUILD, ARCH, true);
        assert_eq!(LOGITS.check_set(&dir)?.dumped_from, MODEL);
        let clef = crate::arch::qwen35::MODEL;
        let cases = [
            (clef, LCPP_BUILD, ARCH, true, "dumped from"),
            (MODEL, "db517b69", ARCH, true, "build"),
            (MODEL, LCPP_BUILD, "qwen35moe", true, "arch"),
            (MODEL, LCPP_BUILD, ARCH, false, "no `# complete` trailer"),
        ];
        for (model, build, arch, complete, want) in cases {
            let dir = write_set("family", model, build, arch, complete);
            let e = LOGITS.check_set(&dir).expect_err("refused").to_string();
            assert!(e.contains(want), "{model} {build} {arch} {complete}: {e}");
        }
        // One family a file: the 27B's hidden-state family refuses this set, and this family Clef's.
        let dir = write_set("family", MODEL, LCPP_BUILD, ARCH, true);
        let e = HIDDEN.check_set(&dir).expect_err("refused").to_string();
        assert!(e.contains("dumped from"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    /// The opened set holds its ids and rows, and open refuses a set whose files are not the manifest's: a rows or ids
    /// file of another digest, a row of the wrong shape or op, an id outside the vocabulary, and a forced id too many.
    #[test]
    fn a_set_opens_as_its_ids_and_rows_and_a_file_of_another_run_is_refused() -> Result<(), RefError>
    {
        let dir = write_set("open", MODEL, LCPP_BUILD, ARCH, true);
        let set = ForcedLogits::open(&dir)?;
        assert_eq!(
            (set.prompt.as_slice(), set.forced.as_slice()),
            (&PROMPT[..], &FORCED[..])
        );
        assert_eq!((set.vocab, set.rows()), (VOCAB, 2));
        assert_eq!(set.row(0), &ROWS[..VOCAB]);
        assert_eq!(set.row(1), &ROWS[VOCAB..]);

        // The rows' digest: a value moved, the file as long as before and every value finite.
        let mut raw = rows_bytes();
        raw[0] ^= 1;
        std::fs::write(dir.join("logits.0.f32"), raw).expect("rows");
        assert!(refused(&dir).contains("sha256"), "rows");
        // The ids' digests.
        let dir = write_set("open", MODEL, LCPP_BUILD, ARCH, true);
        std::fs::write(dir.join("prompt.ids"), ids_text(&[1, 2, 2])).expect("prompt");
        assert!(refused(&dir).contains("prompt.ids"), "prompt");
        // The row's own columns against the header: an op that is no logits row's, a width that is not `# vocab`.
        let dir = write_set("open", MODEL, LCPP_BUILD, ARCH, true);
        let text = std::fs::read_to_string(dir.join("MANIFEST.tsv")).expect("manifest");
        std::fs::write(dir.join("MANIFEST.tsv"), text.replace(OP, "DIAGNOSTIC")).expect("manifest");
        assert!(refused(&dir).contains("DIAGNOSTIC"), "op");
        std::fs::write(
            dir.join("MANIFEST.tsv"),
            text.replace("# vocab\t4", "# vocab\t5"),
        )
        .expect("manifest");
        assert!(refused(&dir).contains("expect: logits rows"), "vocab");
        // One forced id too many for the rows, digests following the file as a writer's would.
        let dir = write_set("open", MODEL, LCPP_BUILD, ARCH, true);
        let long = ids_text(&[0, 1, 2]);
        std::fs::write(dir.join("forced.ids"), &long).expect("forced");
        let text = std::fs::read_to_string(dir.join("MANIFEST.tsv")).expect("manifest");
        let was = sha(ids_text(&FORCED).as_bytes());
        std::fs::write(
            dir.join("MANIFEST.tsv"),
            text.replace(&was, &sha(long.as_bytes())),
        )
        .expect("manifest");
        assert!(
            refused(&dir).contains("3 forced ids for 2 rows"),
            "forced count"
        );
        // An id outside the vocabulary.
        let bad = ids_text(&[0, 4]);
        std::fs::write(dir.join("forced.ids"), &bad).expect("forced");
        let text = std::fs::read_to_string(dir.join("MANIFEST.tsv")).expect("manifest");
        std::fs::write(
            dir.join("MANIFEST.tsv"),
            text.replace(&sha(long.as_bytes()), &sha(bad.as_bytes())),
        )
        .expect("manifest");
        assert!(refused(&dir).contains("inside the vocabulary of 4"), "id");
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    fn refused_cpu(dir: &Path) -> String {
        ForcedLogits::open_cpu(dir)
            .expect_err("refused")
            .to_string()
    }

    /// The manifest of the set at `dir` with `was` replaced by `now`.
    fn restate(dir: &Path, was: &str, now: &str) {
        let text = std::fs::read_to_string(dir.join("MANIFEST.tsv")).expect("manifest");
        assert!(text.contains(was), "the writer's shape: {was}");
        std::fs::write(dir.join("MANIFEST.tsv"), text.replace(was, now)).expect("manifest");
    }

    /// A CUDA set opens as one and is refused as the CPU twin, by its `# backend`.
    #[test]
    fn a_cuda_set_is_refused_as_the_cpu_twin() {
        let dir = write_set("cuda-as-cpu", MODEL, LCPP_BUILD, ARCH, true);
        ForcedLogits::open(&dir).expect("the CUDA set opens as one");
        let e = refused_cpu(&dir);
        assert!(e.contains("# backend \"cuda\", want \"cpu\""), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The CPU twin passes the family's check and the digest and shape checks of `open`, so `open_cpu` reads it as its
    /// ids and rows; `open` still refuses it, by its `# backend`.
    #[test]
    fn the_cpu_twin_opens_as_the_twin_and_is_refused_as_a_cuda_set() {
        let dir = write_set("twin-as-cuda", MODEL, LCPP_BUILD, ARCH, true);
        restate(&dir, "# backend\tcuda", "# backend\tcpu");
        let twin = ForcedLogits::open_cpu(&dir).expect("the CPU twin opens as one");
        assert_eq!(
            (twin.prompt.as_slice(), twin.forced.as_slice()),
            (&PROMPT[..], &FORCED[..])
        );
        assert_eq!((twin.vocab, twin.rows()), (VOCAB, 2));
        assert_eq!(twin.row(1), &ROWS[VOCAB..]);
        let e = refused(&dir);
        assert!(e.contains("# backend \"cpu\", want \"cuda\""), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A run that is not the reference engine's is refused by both openers, whichever backend it states.
    #[test]
    fn a_native_run_is_refused_by_both_openers() {
        for (backend, name) in [("cuda", "native-cuda"), ("cpu", "native-cpu")] {
            let dir = write_set(name, MODEL, LCPP_BUILD, ARCH, true);
            restate(&dir, "# backend\tcuda", &format!("# backend\t{backend}"));
            restate(&dir, "# engine\treference", "# engine\tnative");
            for e in [refused(&dir), refused_cpu(&dir)] {
                assert!(e.contains("# engine \"native\""), "{backend}: {e}");
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
