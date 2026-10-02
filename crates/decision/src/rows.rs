//! The output embedding rows the head's lexical means read, from a GGUF file.
//!
//! The rows are the output head's: the `output` tensor, else `token_embd` (the rule llama.cpp uses
//! for a tied head); a file with neither is refused by name. Each row is dequantized through
//! `crates/gguf`.

use gguf::{Split, TensorInfo};

use crate::Error;

/// The output head's tensor: its name, its shard and its header entry.
pub fn output_head(split: &Split) -> Result<(&'static str, usize, &TensorInfo), Error> {
    ["output.weight", "token_embd.weight"]
        .into_iter()
        .find_map(|n| split.find(n).map(|(shard, t)| (n, shard, t)))
        .ok_or(Error::NoOutputHead)
}

/// The output head's rows of `ids`, `[ids.len(), width]` row-major f32.
pub fn output_rows(split: &Split, ids: &[u32]) -> Result<Vec<f32>, Error> {
    let (name, shard, t) = output_head(split)?;
    let gguf = split.shard(shard).ok_or(Error::NoOutputHead)?;
    let bad = |what: String| Error::OutputHead { name, what };
    let &[width, vocab] = t.dims.as_slice() else {
        return Err(bad(format!("dims {:?} are not [width, vocab]", t.dims)));
    };
    let width = usize::try_from(width).map_err(|_| bad("the width does not fit usize".into()))?;
    let (blck, size) =
        t.ty.blck_size()
            .zip(t.ty.type_size())
            .ok_or_else(|| bad(format!("{} has no block table", t.ty)))?;
    let blck =
        usize::try_from(blck).map_err(|_| bad("the block size does not fit usize".into()))?;
    let size = usize::try_from(size).map_err(|_| bad("the block bytes do not fit usize".into()))?;
    if blck == 0 || !width.is_multiple_of(blck) {
        return Err(bad(format!("width {width} is not whole {} blocks", t.ty)));
    }
    let row_bytes = width / blck * size;
    let data = gguf.data(t).map_err(|e| bad(e.to_string()))?;
    let mut out = vec![0f32; ids.len() * width];
    for (&id, dst) in ids.iter().zip(out.chunks_exact_mut(width)) {
        if u64::from(id) >= vocab {
            return Err(bad(format!("id {id} is past the vocabulary of {vocab}")));
        }
        let at = id as usize * row_bytes;
        let src = data
            .get(at..at + row_bytes)
            .ok_or_else(|| bad(format!("row {id} runs past the tensor's bytes")))?;
        gguf::quant::dequant_row(t.ty, src, dst).map_err(|e| bad(e.to_string()))?;
    }
    Ok(out)
}
