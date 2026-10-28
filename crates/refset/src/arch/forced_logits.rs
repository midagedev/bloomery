//! The forced-logits sets of the dense models and their one reader. A set is llama.cpp mainline's logits after a prompt and
//! after each id forced behind it, written from a model file by the mainline tree `tools/ref/dense_logits_ref.cpp` links
//! through `tools/ref/dense-logits.py` (`tools/ref/dense-logits.sh`, one profile a file). A set is a node dump's
//! `MANIFEST.tsv` ([`RefManifest`]) with one tensor row:
//!
//! ```text
//! # model    <the model file>
//! # build    <the mainline commit's first nine digits>
//! # arch     <general.architecture>
//! # rows     32            # vocab  <the width of a row>
//! # prompt_sha256  …       # forced_sha256  …       # raw_sha256  …
//! # kind  name  occurrence  type  ne0  ne1  ne2  ne3  bytes  sum  op  contig  logical  src0  src1
//! tensor  logits  0  f32  <vocab>  32  1  1  <rows * vocab * 4>  <sum>  FORCED_LOGITS  1  0  -  -
//! # complete  1  0
//! ```
//!
//! `logits.0.f32` is `rows` rows of `vocab` f32, little-endian: row 0 follows the whole prompt and row `i` follows forced
//! id `i - 1` (the last forced id is not fed). `prompt.ids` and `forced.ids` hold the ids, one decimal a line.
//! [`ForcedLogits::open`] checks the set against its [`Kind`]'s family, then each file against the manifest's digests and
//! shape. The set's CPU twin (`<set>.cpu`, mainline with no card in view) is no family's set: it is the oracle's own
//! floor. [`ForcedLogits::open_cpu`] reads it with the same checks, and each opener refuses the other's `# backend`.
//!
//! What differs between models is the family alone (the file, the architecture, the sets' names); a model's module names
//! its family in a [`Kind`] and aliases [`ForcedLogits`] at it.

use crate::RefError;
use crate::family::Family;
use crate::ik::{RefManifest, ref_tensor_of_in};
use gguf::digest::sha256_file;
use std::marker::PhantomData;
use std::path::Path;

/// The `op` of a set's one tensor row: the rows are no node of any engine's graph.
pub const OP: &str = "FORCED_LOGITS";

/// The family a [`ForcedLogits`] reads its sets through: the model file, the mainline build and the architecture they
/// must state.
pub trait Kind {
    /// The family whose sets this reader opens.
    fn family() -> &'static Family;
}

/// A set of `K`'s family, opened and checked: the ids it was fed and its rows.
#[derive(Debug)]
pub struct ForcedLogits<K> {
    /// The prompt's ids.
    pub prompt: Vec<u32>,
    /// The forced ids; row `i` follows the prompt and `forced[..i]`, and there is one row a forced id.
    pub forced: Vec<u32>,
    /// The width of a row.
    pub vocab: usize,
    rows: Vec<f32>,
    family: PhantomData<K>,
}

impl<K: Kind> ForcedLogits<K> {
    /// The set at `dir` read and checked: the family's check ([`RefManifest::open`]: the trailer, the model file, the
    /// architecture and the build), the reference engine's CUDA rows (`# engine`, `# backend`: the CPU twin is refused),
    /// the one tensor row against `# rows` and `# vocab`, each ids file and the rows file against the digest the manifest
    /// states, every id inside the vocabulary and one forced id a row.
    pub fn open(dir: &Path) -> Result<Self, RefError> {
        Self::open_backend(dir, "cuda")
    }

    /// The CPU twin at `dir` (`<set>.cpu`) read with [`ForcedLogits::open`]'s checks, the `# backend` being `cpu`: a
    /// CUDA set is refused.
    pub fn open_cpu(dir: &Path) -> Result<Self, RefError> {
        Self::open_backend(dir, "cpu")
    }

    /// The set at `dir` whose manifest states the reference engine on `backend`, else as [`ForcedLogits::open`].
    fn open_backend(dir: &Path, backend: &str) -> Result<Self, RefError> {
        let man = RefManifest::open(dir, K::family())?;
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
            family: PhantomData,
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

/// The sha256 of the file at `path`, as lowercase hex.
fn digest(path: &Path) -> Result<String, RefError> {
    sha256_file(path).map_err(|e| RefError::missing(path, e.to_string()))
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

/// The reader's contracts, against a family of its own, and the writer of a set of the shape `dense-logits.py` leaves,
/// which each model's family test reuses.
#[cfg(test)]
pub(crate) mod tests {
    use super::{ForcedLogits, Kind, OP};
    use crate::RefError;
    use crate::family::{Build, Family, Identity};
    use gguf::digest::sha256_hex;
    use std::path::{Path, PathBuf};

    const VOCAB: usize = 4;
    const PROMPT: [u32; 3] = [1, 2, 3];
    const FORCED: [u32; 2] = [0, 1];
    const ROWS: [f32; 8] = [0.5, 3.0, -1.0, 0.25, 1.0, 0.0, 2.0, -2.0];

    const MODEL: &str = "/models/probe/probe-Q4_K_M.gguf";
    const BUILD: &str = "0123456ab";
    const ARCH: &str = "probe";

    fn model() -> Result<String, RefError> {
        Ok(MODEL.to_string())
    }

    /// A family of the shape every model's is, over a file no model has.
    static PROBE: Family = Family {
        name: "logits-probe",
        sets: &[],
        resolve: None,
        recipe: "just dump-logits-probe",
        identity: Identity::Manifest,
        arch: Some(ARCH),
        build: Some(Build::Is(BUILD)),
        runs: Some(model),
        draft_runs: None,
        consumers: &[],
    };

    #[derive(Debug)]
    enum Probe {}

    impl Kind for Probe {
        fn family() -> &'static Family {
            &PROBE
        }
    }

    type Set = ForcedLogits<Probe>;

    fn ids_text(ids: &[u32]) -> String {
        ids.iter().map(|i| format!("{i}\n")).collect()
    }

    fn rows_bytes() -> Vec<u8> {
        ROWS.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// A set of dense-logits.py's shape at a fresh directory: the writer's header lines for these values, its
    /// one tensor row and the three files, the trailer last when `complete`. `name` is unique to the test that
    /// asks, across every module that uses this writer: the tests of a process share its id.
    pub(crate) fn write_set(
        name: &str,
        model: &str,
        build: &str,
        arch: &str,
        complete: bool,
    ) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "bloomery-refset-forced-logits-{}-{name}",
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
            format!("# prompt_sha256\t{}", sha256_hex(ids_text(&PROMPT).as_bytes())),
            format!("# forced_sha256\t{}", sha256_hex(ids_text(&FORCED).as_bytes())),
            "# rows\t2".to_string(),
            format!("# vocab\t{VOCAB}"),
            format!("# raw_sha256\t{}", sha256_hex(&raw)),
            "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1"
                .to_string(),
            format!("tensor\tlogits\t0\tf32\t{VOCAB}\t2\t1\t1\t{}\t{sum:?}\t{OP}\t1\t0\t-\t-", raw.len()),
        ];
        if complete {
            lines.push("# complete\t1\t0".to_string());
        }
        std::fs::write(dir.join("MANIFEST.tsv"), {
            let mut out = lines.join("\n");
            out.push('\n');
            out
        })
        .expect("manifest");
        dir
    }

    fn written(name: &str) -> PathBuf {
        write_set(name, MODEL, BUILD, ARCH, true)
    }

    fn refused(dir: &Path) -> String {
        Set::open(dir).expect_err("refused").to_string()
    }

    fn refused_cpu(dir: &Path) -> String {
        Set::open_cpu(dir).expect_err("refused").to_string()
    }

    /// The manifest of the set at `dir` with `was` replaced by `now`.
    fn restate(dir: &Path, was: &str, now: &str) {
        let text = std::fs::read_to_string(dir.join("MANIFEST.tsv")).expect("manifest");
        assert!(text.contains(was), "the writer's shape: {was}");
        std::fs::write(dir.join("MANIFEST.tsv"), text.replace(was, now)).expect("manifest");
    }

    /// The opened set holds its ids and rows, and open refuses a set whose files are not the manifest's: a rows or ids
    /// file of another digest, a row of the wrong shape or op, an id outside the vocabulary, and a forced id too many.
    #[test]
    fn a_set_opens_as_its_ids_and_rows_and_a_file_of_another_run_is_refused() -> Result<(), RefError>
    {
        let dir = written("open");
        let set = Set::open(&dir)?;
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
        let dir = written("open");
        std::fs::write(dir.join("prompt.ids"), ids_text(&[1, 2, 2])).expect("prompt");
        assert!(refused(&dir).contains("prompt.ids"), "prompt");
        // The row's own columns against the header: an op that is no logits row's, a width that is not `# vocab`.
        let dir = written("open");
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
        let dir = written("open");
        let long = ids_text(&[0, 1, 2]);
        std::fs::write(dir.join("forced.ids"), &long).expect("forced");
        let text = std::fs::read_to_string(dir.join("MANIFEST.tsv")).expect("manifest");
        let was = sha256_hex(ids_text(&FORCED).as_bytes());
        std::fs::write(
            dir.join("MANIFEST.tsv"),
            text.replace(&was, &sha256_hex(long.as_bytes())),
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
            text.replace(&sha256_hex(long.as_bytes()), &sha256_hex(bad.as_bytes())),
        )
        .expect("manifest");
        assert!(refused(&dir).contains("inside the vocabulary of 4"), "id");
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    /// A rows or ids file that is gone is refused as missing, naming the file.
    #[test]
    fn missing_raw_and_id_files_are_missing_naming_the_file() {
        for file in ["logits.0.f32", "prompt.ids", "forced.ids"] {
            let dir = written(&format!("missing-{file}"));
            let path = dir.join(file);
            std::fs::remove_file(&path).expect("remove fixture file");
            match Set::open(&dir) {
                Err(RefError::Missing {
                    path: missing,
                    what,
                }) => {
                    assert_eq!(missing, path);
                    assert!(what.contains(&path.display().to_string()), "{what}");
                }
                result => panic!("expected Missing for {}, got {result:?}", path.display()),
            }
            std::fs::remove_dir_all(dir).expect("remove fixture");
        }
    }

    /// A rows or ids file whose bytes moved is refused as malformed at that file.
    #[test]
    fn altered_raw_and_id_files_are_malformed_naming_the_file() {
        for file in ["logits.0.f32", "prompt.ids", "forced.ids"] {
            let dir = written(&format!("altered-{file}"));
            let path = dir.join(file);
            let mut bytes = std::fs::read(&path).expect("fixture file");
            bytes[0] ^= 1;
            std::fs::write(&path, bytes).expect("alter fixture file");
            match Set::open(&dir) {
                Err(RefError::Malformed { at, .. }) => {
                    assert_eq!(at, path.display().to_string());
                }
                result => panic!("expected Malformed for {}, got {result:?}", path.display()),
            }
            std::fs::remove_dir_all(dir).expect("remove fixture");
        }
    }

    /// A CUDA set opens as one and is refused as the CPU twin, by its `# backend`.
    #[test]
    fn a_cuda_set_is_refused_as_the_cpu_twin() {
        let dir = written("cuda-as-cpu");
        Set::open(&dir).expect("the CUDA set opens as one");
        let e = refused_cpu(&dir);
        assert!(e.contains("# backend \"cuda\", want \"cpu\""), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The CPU twin passes the family's check and the digest and shape checks of `open`, so `open_cpu` reads it as its
    /// ids and rows; `open` still refuses it, by its `# backend`.
    #[test]
    fn the_cpu_twin_opens_as_the_twin_and_is_refused_as_a_cuda_set() {
        let dir = written("twin-as-cuda");
        restate(&dir, "# backend\tcuda", "# backend\tcpu");
        let twin = Set::open_cpu(&dir).expect("the CPU twin opens as one");
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
            let dir = written(name);
            restate(&dir, "# backend\tcuda", &format!("# backend\t{backend}"));
            restate(&dir, "# engine\treference", "# engine\tnative");
            for e in [refused(&dir), refused_cpu(&dir)] {
                assert!(e.contains("# engine \"native\""), "{backend}: {e}");
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
