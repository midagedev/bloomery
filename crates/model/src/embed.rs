//! The token-embedding row walk the host side of every model shares: find
//! the tensor in the file, check that it is whole rows of the model's
//! width, slice one row's file bytes, dequantize it. [`RowLayout`] is the
//! walk; a body that owns its file holds [`EmbedTable`] (the layout and the
//! file's `Arc`), one that borrows the file keeps the layout beside its
//! borrow. A body that repeats a row into its streams copies what
//! `row_into` fills; one that dequantizes on the card reads `row`'s bytes
//! alone.

use std::sync::Arc;

use gguf::quant::dequant_row;
use gguf::{GgmlType, Split, TensorInfo};

use crate::ModelError;

/// The geometry of a token-embedding table of an open [`Split`]: where the
/// tensor sits and how its rows are laid out, checked once at open, then
/// reads rows of whatever file it is handed. A holder of a borrowed
/// [`Split`] keeps this beside its borrow; a holder that owns the file takes
/// [`EmbedTable`].
#[derive(Clone)]
pub struct RowLayout {
    shard: usize,
    info: TensorInfo,
    n_embd: usize,
    row_bytes: usize,
    n_vocab: usize,
}

impl RowLayout {
    /// The tensor `name` of `file` as a table of rows `n_embd` values wide.
    /// Refused by name: a tensor not in the file, a first dimension other
    /// than `n_embd`, a type ggml sizes no row of `n_embd` values for, and
    /// bytes that are not a whole number of rows.
    pub fn open(file: &Split, name: &str, n_embd: usize) -> Result<RowLayout, ModelError> {
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
        Ok(RowLayout {
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

    /// The file bytes of one row.
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Row `token`'s bytes of `file`, the file this layout was opened on. A
    /// token at or past the vocabulary is refused by name, naming the token
    /// and the vocabulary.
    pub fn row<'a>(&self, file: &'a Split, token: u32) -> Result<&'a [u8], ModelError> {
        let t = token as usize;
        if t >= self.n_vocab {
            return Err(self.refuse(format!(
                "token {token} is past the {n_vocab} rows",
                n_vocab = self.n_vocab
            )));
        }
        let shard = file.shard(self.shard).ok_or_else(|| {
            self.refuse(format!("the file has no shard {shard}", shard = self.shard))
        })?;
        let data = shard.data(&self.info)?;
        Ok(&data[t * self.row_bytes..][..self.row_bytes])
    }

    /// The layout's refusal of `detail`, naming the tensor.
    fn refuse(&self, detail: String) -> ModelError {
        ModelError::EmbedRows {
            tensor: self.info.name.clone(),
            detail,
        }
    }
}

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
    layout: RowLayout,
}

impl EmbedTable {
    /// The tensor `name` of `file` as a table of rows `n_embd` values wide,
    /// refused as [`RowLayout::open`] refuses it, and by name when its type
    /// is one [`dequant_row`] does not decode (`GgmlType::has_dequant`, the
    /// set a coverage pin of a body that reads this table holds too).
    pub fn new(file: Arc<Split>, name: &str, n_embd: usize) -> Result<EmbedTable, ModelError> {
        let layout = RowLayout::open(&file, name, n_embd)?;
        if !layout.ty().has_dequant() {
            return Err(layout.refuse(format!(
                "is {ty}, whose rows the host reader does not decode",
                ty = layout.ty()
            )));
        }
        Ok(EmbedTable { file, layout })
    }

    /// The table's ggml type, for a body whose load runs one type only.
    pub fn ty(&self) -> GgmlType {
        self.layout.ty()
    }

    /// The vocabulary: how many rows the table holds.
    pub fn n_vocab(&self) -> usize {
        self.layout.n_vocab()
    }

    /// Row `token`'s file bytes. A token at or past the vocabulary is
    /// refused by name, naming the token and the vocabulary.
    pub fn row(&self, token: u32) -> Result<&[u8], ModelError> {
        self.layout.row(&self.file, token)
    }

    /// Row `token` dequantized into `out`, which holds the table's width.
    /// A token at or past the vocabulary is refused as [`EmbedTable::row`]
    /// refuses it.
    pub fn row_into(&self, token: u32, out: &mut [f32]) -> Result<(), ModelError> {
        let layout = &self.layout;
        if out.len() != layout.n_embd {
            return Err(layout.refuse(format!(
                "fills {n} values, out holds {m}",
                n = layout.n_embd,
                m = out.len()
            )));
        }
        let src = self.row(token)?;
        dequant_row(layout.ty(), src, out)?;
        Ok(())
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
        opened_as(8, want, dims, n_embd)
    }

    /// [`opened`] for a file whose tensor is of ggml type `type_id`: its
    /// bytes are `row_of` rows for q8_0 and zeros for any other type, whose
    /// blocks the test never decodes.
    fn opened_as(
        type_id: u32,
        want: &str,
        dims: &[u64],
        n_embd: usize,
    ) -> Result<EmbedTable, ModelError> {
        use std::sync::atomic::{AtomicU32, Ordering};
        // One directory a call: the tests run in parallel, and two calls of
        // one shape would otherwise empty each other's directory.
        static CALL: AtomicU32 = AtomicU32::new(0);
        let ty = GgmlType::from_u32(type_id);
        let (blck, tsz) = (ty.blck_size().unwrap(), ty.type_size().unwrap());
        let nbytes = tsz * (dims[0] / blck) * dims[1..].iter().product::<u64>();
        let mut bytes = Vec::with_capacity(nbytes as usize);
        if ty == GgmlType::Q8_0 {
            for t in 0..nbytes / ROW {
                bytes.extend(row_of(t as usize));
            }
        } else {
            bytes.resize(nbytes as usize, 0);
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
            type_id,
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

    /// The borrowed layout reads the bytes the owning table reads, and
    /// refuses a token past the vocabulary the same way.
    #[test]
    fn the_layout_reads_the_tables_rows_of_a_borrowed_file() {
        let t = table(4);
        let l = RowLayout::open(&t.file, NAME, N_EMBD).unwrap();
        assert_eq!((l.ty(), l.n_vocab(), l.row_bytes()), (t.ty(), 4, 34));
        for token in 0..4u32 {
            assert_eq!(l.row(&t.file, token).unwrap(), t.row(token).unwrap());
        }
        let want = t.row(4).unwrap_err().to_string();
        assert_eq!(l.row(&t.file, 4).unwrap_err().to_string(), want);
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

    /// The table opens the types `dequant_row` decodes, named here, and
    /// refuses a file whose type it does not decode by name, tensor and type.
    #[test]
    fn a_type_the_host_reader_does_not_decode_is_refused_by_name() {
        for ty in [
            GgmlType::Q8_0,
            GgmlType::Q3_K,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::F16,
            GgmlType::IQ4_XS,
        ] {
            assert!(ty.has_dequant(), "{ty}");
        }
        for type_id in [16u32, 19, 22] {
            let ty = GgmlType::from_u32(type_id);
            assert!(!ty.has_dequant(), "{ty}");
            let Err(e) = opened_as(type_id, NAME, &[256, 2], 256) else {
                panic!("a {ty} table opens");
            };
            let e = e.to_string();
            assert!(
                e.contains(NAME) && e.contains(&format!("is {ty}, whose rows the host reader")),
                "{e:?}"
            );
        }
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
