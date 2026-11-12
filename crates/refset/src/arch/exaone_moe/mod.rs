//! The K-EXAONE-2.0 family (`exaone-moe`): the tokenizer only so far. [`VOCABULARY`] is the
//! vocab-only file `convert_hf_to_gguf.py --vocab-only` writes from the hub folder (the
//! metadata and the tokenizer, no tensors), the one file the tokenizer gate
//! (`crates/tokenizer/tests/exaone.rs`) and its oracle
//! (`crates/tokenizer/tools/exaone-oracle.sh`) read; the node-dump sets and the fixture the
//! later rounds add will live here with it.

/// The vocab-only GGUF the converter writes from the hub folder: the file the `exaone` tokenizer
/// gate opens.
pub const VOCABULARY: &str = "/data/models/K-EXAONE-2.0-vocab/exaone-moe-vocab.gguf";
