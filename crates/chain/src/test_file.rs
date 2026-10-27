//! A synthetic file header for the reader tests: tensor declarations and no weight bytes, all a
//! plan reads.

use gguf::Split;
use gguf::quant::GgmlType;
use gguf::write::{Layout, TensorDecl};
use std::path::PathBuf;

/// `name`, `dims` (`dims[0]` the contiguous axis) at `ty`, as a file header declares it.
pub(crate) fn decl(name: String, dims: &[u64], ty: GgmlType) -> TensorDecl {
    let id = ty.as_u32();
    let (_, blck, size) = gguf::ggml_type_info(id).expect("a type the table sizes");
    TensorDecl {
        name,
        dims: dims.to_vec(),
        type_id: id,
        nbytes: gguf::tensor_bytes(blck, size, dims).expect("a size that fits"),
    }
}

/// A file's header of `decls` and no weight bytes.
pub(crate) fn header(decls: Vec<TensorDecl>) -> Split {
    let layout = Layout::new(&[], decls).unwrap_or_else(|e| panic!("{e}"));
    Split::header_only(PathBuf::from("synthetic.gguf"), &layout).unwrap_or_else(|e| panic!("{e}"))
}
