//! Raw diagnostic logits, independent of how a generator chooses its next token.

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::GateError;

/// Write f32 bits in vocabulary order, little-endian, without changing the row.
pub(crate) fn write_f32s(writer: &mut impl Write, values: &[f32]) -> io::Result<usize> {
    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    writer.write_all(&bytes)?;
    Ok(bytes.len())
}

/// The input tokens and schedule behind a teacher-forced row dump.
pub struct ForcedFacts<'a> {
    pub prompt_ids: &'a [u32],
    pub forced_ids: &'a [u32],
    pub mode: &'a str,
    pub prefill: &'a str,
}

/// A new raw f32 file whose sidecar is completed only after every expected row.
/// Row zero follows the full prompt; row i follows forced token i - 1.
pub struct ForcedRows<'a> {
    raw: BufWriter<File>,
    meta: PathBuf,
    vocab: usize,
    rows: usize,
    facts: ForcedFacts<'a>,
}

impl<'a> ForcedRows<'a> {
    /// Create a new dump, refusing existing output, empty inputs and out-of-vocabulary ids.
    /// The caller must feed the prompt's last token as a step before reading row zero.
    pub fn create(path: &Path, vocab: usize, facts: ForcedFacts<'a>) -> Result<Self, GateError> {
        if vocab == 0 || facts.prompt_ids.is_empty() || facts.forced_ids.is_empty() {
            return Err(
                "--logit-rows needs a vocabulary, a prompt and forced output tokens".into(),
            );
        }
        if !matches!(facts.mode, "eager" | "graph")
            || !matches!(facts.prefill, "auto" | "pass" | "gemm")
        {
            return Err("--logit-rows received an unsupported execution schedule".into());
        }
        for (kind, ids) in [("prompt", facts.prompt_ids), ("forced", facts.forced_ids)] {
            if let Some((i, id)) = ids
                .iter()
                .enumerate()
                .find(|(_, id)| **id as usize >= vocab)
            {
                return Err(format!(
                    "--logit-rows: {kind} id {i} is {id}, outside vocabulary {vocab}"
                )
                .into());
            }
        }
        let mut meta = path.as_os_str().to_os_string();
        meta.push(".meta");
        let meta = PathBuf::from(meta);
        if meta.try_exists()? {
            return Err(format!("--logit-rows: {} already exists", meta.display()).into());
        }
        let raw = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| format!("--logit-rows {}: {e}", path.display()))?;
        Ok(Self {
            raw: BufWriter::new(raw),
            meta,
            vocab,
            rows: 0,
            facts,
        })
    }

    /// Append one finite vocabulary row; the forced token need not be its argmax.
    pub fn read(&mut self, row: &[f32]) -> Result<(), GateError> {
        if self.rows >= self.facts.forced_ids.len() || row.len() != self.vocab {
            return Err(format!(
                "--logit-rows: row {} has {} values for {} rows of vocabulary {}",
                self.rows,
                row.len(),
                self.facts.forced_ids.len(),
                self.vocab
            )
            .into());
        }
        if let Some((id, value)) = row.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(format!(
                "--logit-rows: row {} id {id} is non-finite ({value})",
                self.rows
            )
            .into());
        }
        write_f32s(&mut self.raw, row)?;
        self.rows += 1;
        Ok(())
    }

    /// Flush all rows and write the dimensions, exact ids and completion trailer to FILE.meta.
    pub fn finish(mut self) -> Result<(), GateError> {
        if self.rows != self.facts.forced_ids.len() {
            return Err(format!(
                "--logit-rows: wrote {} of {} forced rows",
                self.rows,
                self.facts.forced_ids.len()
            )
            .into());
        }
        self.raw.flush()?;
        let mut meta = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&self.meta)?,
        );
        writeln!(
            meta,
            "format\tbloomery-forced-logits-v1\ndtype\tfloat32-le\nrows\t{}\nvocab\t{}",
            self.rows, self.vocab
        )?;
        for (kind, ids) in [
            ("prompt_ids", self.facts.prompt_ids),
            ("forced_ids", self.facts.forced_ids),
        ] {
            write!(meta, "{kind}\t")?;
            for (i, id) in ids.iter().enumerate() {
                if i > 0 {
                    write!(meta, ",")?;
                }
                write!(meta, "{id}")?;
            }
            writeln!(meta)?;
        }
        writeln!(
            meta,
            "mode\t{}\nprefill\t{}\nlast_step\ttrue\ncomplete\ttrue",
            self.facts.mode, self.facts.prefill
        )?;
        meta.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT: AtomicUsize = AtomicUsize::new(0);

    fn path() -> PathBuf {
        std::env::temp_dir().join(format!(
            "bloomery-forced-rows-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn facts() -> ForcedFacts<'static> {
        ForcedFacts {
            prompt_ids: &[2, 1],
            forced_ids: &[0, 1],
            mode: "graph",
            prefill: "pass",
        }
    }

    #[test]
    fn forced_rows_keep_bits_and_token_alignment_without_argmax_picking() {
        let path = path();
        let mut dump = ForcedRows::create(&path, 3, facts()).unwrap();
        let meta = dump.meta.clone();
        assert!(!meta.exists());
        dump.read(&[-0.0, 4.0, 9.0]).unwrap();
        dump.read(&[8.0, 2.0, 3.0]).unwrap();
        assert!(dump.read(&[1.0, 2.0, 3.0]).is_err());
        dump.finish().unwrap();
        let expected: Vec<_> = [-0.0f32, 4.0, 9.0, 8.0, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(std::fs::read(&path).unwrap(), expected);
        let text = std::fs::read_to_string(&meta).unwrap();
        assert!(text.contains("rows\t2\nvocab\t3\nprompt_ids\t2,1\nforced_ids\t0,1\n"));
        assert!(text.ends_with("last_step\ttrue\ncomplete\ttrue\n"));
        assert!(ForcedRows::create(&path, 3, facts()).is_err());
        std::fs::remove_file(path).unwrap();
        std::fs::remove_file(meta).unwrap();
    }

    #[test]
    fn refused_or_incomplete_rows_never_complete_a_dump() {
        let path = path();
        assert!(ForcedRows::create(&path, 2, facts()).is_err());
        assert!(!path.exists());
        let mut dump = ForcedRows::create(&path, 3, facts()).unwrap();
        let meta = dump.meta.clone();
        for row in [
            &[1.0, 2.0][..],
            &[1.0, f32::NAN, 2.0],
            &[f32::INFINITY, 1.0, 2.0],
        ] {
            assert!(dump.read(row).is_err());
        }
        dump.read(&[1.0, 2.0, 3.0]).unwrap();
        assert!(dump.finish().is_err());
        assert!(!meta.exists());
        assert!(ForcedRows::create(&path, 3, facts()).is_err());
        std::fs::remove_file(path).unwrap();
    }
}
