//! The token-embedding row walk the host side of every model shares: find
//! the tensor in the file, check that it is whole rows of the model's
//! width, slice one row's file bytes, dequantize it. A body holds
//! [`EmbedTable`] beside what it does with the rows — a body that repeats
//! a row into its streams copies what `row_into` fills, one that
//! quantizes on the card reads `row`'s bytes alone.

use std::sync::Arc;

use gguf::quant::dequant_row;
use gguf::{GgmlType, Split, TensorInfo};

use crate::ModelError;

/// A row table of an open [`Split`]: the token embedding — `n_vocab` rows
/// of `n_embd` values of one ggml type — opened once at load and read a
/// row at a time by token.
///
/// The file is held by its `Arc`, not borrowed: a body holds its table for
/// the model's life inside holders that carry no lifetime (its model, a
/// gate, a server), and the load shares the file with its other tiers
/// anyway, so one more reference costs nothing a borrow would save.
pub struct EmbedTable {
    file: Arc<Split>,
    shard: usize,
    info: TensorInfo,
    n_embd: usize,
    row_bytes: usize,
    n_vocab: usize,
}

impl EmbedTable {
    /// The tensor `name` of `file` as a table of rows `n_embd` values wide.
    /// Refused by name: a tensor not in the file, a first dimension other
    /// than `n_embd`, a type ggml sizes no row of `n_embd` values for, and
    /// bytes that are not a whole number of rows.
    pub fn new(file: Arc<Split>, name: &str, n_embd: usize) -> Result<EmbedTable, ModelError> {
        let (shard, info) = file
            .find(name)
            .map(|(s, t)| (s, t.clone()))
            .ok_or_else(|| ModelError::MissingTensor(name.to_string()))?;
        let refuse = |detail: String| ModelError::EmbedRows {
            tensor: name.to_string(),
            detail,
        };
        if info.dims.first() != Some(&(n_embd as u64)) {
            return Err(refuse(format!(
                "is {:?}, want rows of {n_embd} values",
                info.dims
            )));
        }
        let row_bytes = match (info.ty.blck_size(), info.ty.type_size()) {
            (Some(blck), Some(tsz)) if n_embd.is_multiple_of(blck as usize) => {
                tsz as usize * (n_embd / blck as usize)
            }
            _ => {
                return Err(refuse(format!(
                    "is {ty}, which sizes no row of {n_embd} values",
                    ty = info.ty
                )));
            }
        };
        let n_vocab = info.dims.get(1).copied().unwrap_or(0) as usize;
        if n_vocab == 0 {
            return Err(refuse("holds no rows".to_string()));
        }
        let Some(want) = (n_vocab as u64).checked_mul(row_bytes as u64) else {
            return Err(refuse(format!(
                "holds {n_vocab} rows, more than any file does"
            )));
        };
        if info.nbytes != want {
            return Err(refuse(format!(
                "holds {} bytes, not the {want} of its {n_vocab} rows",
                info.nbytes
            )));
        }
        Ok(EmbedTable {
            file,
            shard,
            info,
            n_embd,
            row_bytes,
            n_vocab,
        })
    }

    /// The table's ggml type, for a body whose load runs one type only.
    pub fn ty(&self) -> GgmlType {
        self.info.ty
    }

    /// The vocabulary: how many rows the table holds.
    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    /// Row `token`'s file bytes. A token at or past the vocabulary is
    /// refused by name, naming the token and the vocabulary.
    pub fn row(&self, token: u32) -> Result<&[u8], ModelError> {
        let t = token as usize;
        if t >= self.n_vocab {
            return Err(self.refuse(format!(
                "token {token} is past the {n_vocab} rows",
                n_vocab = self.n_vocab
            )));
        }
        let data = self
            .file
            .shard(self.shard)
            // `Split::find` proved the shard at open, and a `Split` never
            // changes after.
            .expect("the shard Split::find named at open")
            .data(&self.info)?;
        Ok(&data[t * self.row_bytes..][..self.row_bytes])
    }

    /// Row `token` dequantized into `out`, which holds the table's width.
    /// A token at or past the vocabulary is refused as [`EmbedTable::row`]
    /// refuses it.
    pub fn row_into(&self, token: u32, out: &mut [f32]) -> Result<(), ModelError> {
        if out.len() != self.n_embd {
            return Err(self.refuse(format!(
                "fills {n} values, out holds {m}",
                n = self.n_embd,
                m = out.len()
            )));
        }
        let src = self.row(token)?;
        dequant_row(self.info.ty, src, out)?;
        Ok(())
    }

    /// The table's refusal of `detail`, naming the tensor.
    fn refuse(&self, detail: String) -> ModelError {
        ModelError::EmbedRows {
            tensor: self.info.name.clone(),
            detail,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gguf::write::{Layout, TensorDecl, Writer};
    use gguf::{GENERAL_ARCHITECTURE, Value};

    const NAME: &str = "token_embd";
    /// The tables under test hold 32-value q8_0 rows, 34 bytes each.
    const N_EMBD: usize = 32;
    const ROW: u64 = 34;

    /// The bytes of row `t` the files under test hold: the f16 scale 1,
    /// then codes `t * 32 + i`.
    fn row_of(t: usize) -> Vec<u8> {
        let mut b = vec![0x00, 0x3c];
        b.extend((0..32u16).map(|i| (t as u16 * 32 + i) as u8));
        b
    }

    /// A one-tensor q8_0 file of `dims` (`dims[0]` a whole q8_0 block
    /// count, so the writer and the reader take it), every 34 bytes a
    /// `row_of` row, its tensor named [`NAME`], opened as a table of rows
    /// `n_embd` values wide under the name `want` — a name the file does
    /// not hold is the missing-tensor refusal. The header's shape need not
    /// be whole `dims[1]` rows, so the table's own refusal has a case to
    /// refuse.
    fn opened(want: &str, dims: &[u64], n_embd: usize) -> Result<EmbedTable, ModelError> {
        use std::sync::atomic::{AtomicU32, Ordering};
        // One directory a call: the tests run in parallel, and two calls of
        // one shape would otherwise empty each other's directory.
        static CALL: AtomicU32 = AtomicU32::new(0);
        let nbytes = ROW * (dims[0] / 32) * dims[1..].iter().product::<u64>();
        let mut bytes = Vec::with_capacity(nbytes as usize);
        for t in 0..nbytes / ROW {
            bytes.extend(row_of(t as usize));
        }
        let dir = std::env::temp_dir().join(format!(
            "model-embed-{}-{}",
            std::process::id(),
            CALL.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.gguf");
        let decl = TensorDecl {
            name: NAME.to_string(),
            dims: dims.to_vec(),
            type_id: 8, // q8_0
            nbytes,
        };
        let layout = Layout::new(
            &[(
                GENERAL_ARCHITECTURE.to_string(),
                Value::String("embed-test".to_string()),
            )],
            vec![decl],
        )
        .unwrap();
        let file = std::fs::File::create(&path).unwrap();
        let mut w = Writer::new(std::io::BufWriter::new(file), layout).unwrap();
        w.tensor(NAME, &bytes).unwrap();
        w.finish().unwrap();
        let split = Arc::new(Split::open(&path).unwrap());
        std::fs::remove_dir_all(&dir).unwrap();
        EmbedTable::new(split, want, n_embd)
    }

    fn table(n_vocab: u64) -> EmbedTable {
        opened(NAME, &[N_EMBD as u64, n_vocab], N_EMBD).unwrap()
    }

    /// Row `t` is row `t`'s file bytes, and `row_into` their dequant.
    #[test]
    fn row_t_returns_row_ts_bytes() {
        let t = table(4);
        for token in [0u32, 1, 3] {
            assert_eq!(t.row(token).unwrap(), row_of(token as usize));
        }
        let mut out = vec![0.0f32; N_EMBD];
        t.row_into(2, &mut out).unwrap();
        let mut want = vec![0.0f32; N_EMBD];
        dequant_row(GgmlType::Q8_0, &row_of(2), &mut want).unwrap();
        assert_eq!(out, want);
    }

    /// A token at the vocabulary is refused by name, both numbers named.
    #[test]
    fn a_token_at_the_vocabulary_is_refused_by_name() {
        let t = table(3);
        for token in [3u32, 9] {
            let e = t.row(token).unwrap_err().to_string();
            assert!(
                e.contains(&format!("{NAME}: token {token} is past the 3 rows")),
                "{e:?}"
            );
            let mut out = vec![0.0f32; N_EMBD];
            assert!(t.row_into(token, &mut out).is_err());
        }
    }

    /// A three-dimensional table, whose bytes are not a whole number of
    /// its second dimension's rows, is refused by name.
    #[test]
    fn a_table_whose_bytes_are_not_whole_rows_is_refused_by_name() {
        let Err(e) = opened(NAME, &[N_EMBD as u64, 2, 2], N_EMBD) else {
            panic!("a 3-dimensional table opens as rows of its second dimension");
        };
        let e = e.to_string();
        // dims [32, 2, 2] holds 136 bytes; the table reads 2 rows of 34.
        assert!(
            e.contains(NAME) && e.contains("136") && e.contains("68"),
            "{e:?}"
        );
    }

    /// A tensor not in the file is refused by name.
    #[test]
    fn a_missing_tensor_is_refused_by_name() {
        let Err(e) = opened("output.weight", &[N_EMBD as u64, 3], N_EMBD) else {
            panic!("a tensor the file does not hold opens");
        };
        let e = e.to_string();
        assert!(
            e.contains("output.weight") && e.contains("not in the file"),
            "{e:?}"
        );
    }

    /// A first dimension other than the width is refused by name.
    #[test]
    fn a_first_dimension_other_than_the_width_is_refused_by_name() {
        let Err(e) = opened(NAME, &[2 * N_EMBD as u64, 3], N_EMBD) else {
            panic!("a table of 64-value rows opens at 32 values");
        };
        let e = e.to_string();
        assert!(
            e.contains(NAME) && e.contains("want rows of 32 values"),
            "{e:?}"
        );
    }
}
