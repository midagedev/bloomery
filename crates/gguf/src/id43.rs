//! Type id 43, the one id two files mean differently.
//!
//! ggml's table gives 43 to F8_E4M3 (one byte a value); llama.cpp's MXFP8 writers, Unsloth's DeepSeek
//! V4.1 engram table among them, tag their tensors 43 too, as 33 bytes a block of 32 values (an E8M0
//! scale byte, then 32 E4M3 codes: [`GgmlType::MXFP8`]). The header does not say which, but the bytes
//! do. A tensor's data runs from its offset to the next tensor's offset, or to the end of the file for
//! the last one, and the writer pads each tensor to the file's alignment, so a meaning holds when the
//! tensor's bytes in it are no more than that extent and fewer than one alignment short of it. The
//! two sizes differ by a thirty-second of the tensor, so a tensor of any size past a few alignments
//! fits one and not the other. A tensor that fits neither or both is refused by name, never guessed.

use crate::{GgmlType, LoadError, RawTensorInfo, tensor_bytes};

/// The ambiguous ggml type id.
pub(crate) const ID: u32 = 43;

/// The two meanings of [`ID`] as bloomery's own value types: ggml's F8_E4M3 rows and llama.cpp's MXFP8.
const MEANINGS: [GgmlType; 2] = [GgmlType::F8E4M3_ROWS, GgmlType::MXFP8];

/// The bytes a tensor of `dims` takes in `ty`, `None` when its first dim is not whole blocks or the
/// product passes a u64.
fn bytes_in(ty: GgmlType, dims: &[u64]) -> Option<u64> {
    let (blck, size) = (ty.blck_size()?, ty.type_size()?);
    dims[0]
        .is_multiple_of(blck)
        .then(|| tensor_bytes(blck, size, dims))
        .flatten()
}

/// Give every tensor of `tensors` tagged [`ID`] the type its bytes fit, as that type's own id and
/// size. `data_base` and `len` place the data in the file; `alignment` is its padding.
///
/// # Errors
/// [`LoadError::AmbiguousType`] for a tensor whose extent fits neither meaning or both.
pub(crate) fn resolve(
    tensors: &mut [RawTensorInfo],
    data_base: u64,
    len: u64,
    alignment: u64,
) -> Result<(), LoadError> {
    if tensors.iter().all(|t| t.type_id != ID) {
        return Ok(());
    }
    let mut starts: Vec<u64> = tensors.iter().map(|t| t.offset).collect();
    starts.sort_unstable();
    let end = len.saturating_sub(data_base);
    for t in tensors.iter_mut().filter(|t| t.type_id == ID) {
        let next = starts
            .iter()
            .copied()
            .find(|&o| o > t.offset)
            .unwrap_or(end);
        let extent = next.saturating_sub(t.offset);
        let sizes = MEANINGS.map(|ty| bytes_in(ty, &t.dims));
        let fits = sizes.map(|b| b.is_some_and(|b| b <= extent && extent - b < alignment));
        let ty = match fits {
            [true, false] => MEANINGS[0],
            [false, true] => MEANINGS[1],
            [f8, mx] => {
                let size =
                    |b: Option<u64>| b.map_or("not whole blocks".to_string(), |b| b.to_string());
                return Err(LoadError::AmbiguousType {
                    name: std::mem::take(&mut t.name),
                    extent,
                    detail: format!(
                        "fit {} (F8_E4M3 {} bytes, MXFP8 {} bytes, each short of the extent by less than the alignment {alignment})",
                        if f8 && mx { "both" } else { "neither" },
                        size(sizes[0]),
                        size(sizes[1]),
                    ),
                });
            }
        };
        t.type_id = ty.as_u32();
        t.nbytes = bytes_in(ty, &t.dims);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::write::{TensorDecl, write_file};
    use crate::{GENERAL_ALIGNMENT, GgmlType, Gguf, LoadError, Split, Value};
    use std::path::{Path, PathBuf};

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("gguf-id43-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn decl(name: &str, dims: &[u64], type_id: u32, nbytes: u64) -> TensorDecl {
        TensorDecl {
            name: name.to_string(),
            dims: dims.to_vec(),
            type_id,
            nbytes,
        }
    }

    /// A file of an F32 tensor, then the tensor `t` tagged id 43 with `nbytes` bytes, then an F32
    /// tensor: `t` is neither the first nor the last tensor's bytes in the file.
    fn tagged(path: &Path, kvs: &[(String, Value)], dims: &[u64], nbytes: u64) {
        let bytes: Vec<u8> = (0..nbytes).map(|i| (i * 7 + 3) as u8).collect();
        write_file(
            path,
            kvs,
            &[
                (decl("a", &[4], 0, 16), vec![1; 16]),
                (decl("t", dims, 43, nbytes), bytes),
                (decl("z", &[4], 0, 16), vec![2; 16]),
            ],
        );
    }

    fn refusal(e: &LoadError) -> String {
        let text = e.to_string();
        assert!(
            text.contains("\"t\"") && text.contains("type id 43"),
            "the refusal names the tensor and the id: {text}"
        );
        text
    }

    /// A tensor of id 43 whose bytes are 33 a 32 values opens as MXFP8 and one whose bytes are one a
    /// value as F8 rows, in the middle of a file and at its end; the strict open of one file, which
    /// holds no own type, refuses each by name.
    #[test]
    fn the_extent_picks_the_meaning() {
        let d = dir("pick");
        for (dims, nbytes, want) in [
            (vec![64u64, 4], 264u64, GgmlType::MXFP8),
            (vec![64, 4], 256, GgmlType::F8E4M3_ROWS),
            (vec![96, 3], 297, GgmlType::MXFP8),
            (vec![96, 3], 288, GgmlType::F8E4M3_ROWS),
        ] {
            let p = d.join(format!("{nbytes}.gguf"));
            tagged(&p, &[], &dims, nbytes);
            let split = Split::open(&p).unwrap();
            let (s, t) = split.find("t").unwrap();
            assert_eq!(
                (t.ty, t.nbytes, t.dims.as_slice()),
                (want, nbytes, &dims[..])
            );
            assert_eq!(
                split.shard(s).unwrap().data(t).unwrap().len() as u64,
                nbytes,
                "{want}: the bytes the tensor reads are its own, not its padding"
            );
            let Err(e) = Gguf::open(&p) else {
                panic!("{want} opens through a split alone")
            };
            assert!(
                matches!(e, LoadError::UnsupportedType { ref name, ty } if name == "t" && ty == want),
                "{e}"
            );
        }
        // The last tensor of the file: its extent runs to the file's end.
        let p = d.join("last.gguf");
        let bytes: Vec<u8> = (0..264u32).map(|i| (i * 5) as u8).collect();
        write_file(&p, &[], &[(decl("t", &[64, 4], 43, 264), bytes)]);
        let split = Split::open(&p).unwrap();
        assert_eq!(split.find("t").unwrap().1.ty, GgmlType::MXFP8);
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// Bytes that fit neither meaning, and a tensor too small for the padding to tell the two apart,
    /// are refused by name, whatever the file's last tensor does.
    #[test]
    fn bytes_that_cannot_tell_are_refused() {
        let d = dir("refuse");
        let p = d.join("neither.gguf");
        tagged(&p, &[], &[64, 4], 300);
        let Err(e) = Split::open(&p) else {
            panic!("300 bytes of 256 values fit neither meaning")
        };
        let text = refusal(&e);
        assert!(
            text.contains("neither") && text.contains("256") && text.contains("264"),
            "{text}"
        );

        // 32 values are 32 bytes as F8 and 33 as MXFP8; a 4096-byte alignment pads both to the same
        // extent.
        let wide = [(GENERAL_ALIGNMENT.to_string(), Value::U32(4096))];
        let p = d.join("both.gguf");
        tagged(&p, &wide, &[32, 1], 33);
        let Err(e) = Split::open(&p) else {
            panic!("32 values padded to 4096 bytes fit both meanings")
        };
        let text = refusal(&e);
        assert!(text.contains("both"), "{text}");

        // A row that is not whole MXFP8 blocks can only be F8 rows.
        let p = d.join("ragged.gguf");
        tagged(&p, &[], &[40, 2], 80);
        let split = Split::open(&p).unwrap();
        assert_eq!(split.find("t").unwrap().1.ty, GgmlType::F8E4M3_ROWS);
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// The header-only inventory of a file reads id 43 the same way: the resolved type's size, and
    /// a file whose bytes cannot tell is refused there too.
    #[test]
    fn the_inventory_resolves_too() {
        let d = dir("inventory");
        let p = d.join("inv.gguf");
        tagged(&p, &[], &[64, 4], 264);
        let inv = crate::inventory_of(&p).unwrap();
        let t = inv.tensors.iter().find(|t| t.name == "t").unwrap();
        assert_eq!(
            (t.type_id, t.nbytes),
            (GgmlType::MXFP8.as_u32(), Some(264)),
            "an inventory row carries the resolved type"
        );
        let p = d.join("bad.gguf");
        tagged(&p, &[], &[64, 4], 300);
        let Err(e) = crate::inventory_of(&p) else {
            panic!("300 bytes of 256 values fit neither meaning")
        };
        assert!(refusal(&e).contains("neither"));
        std::fs::remove_dir_all(&d).unwrap();
    }
}
