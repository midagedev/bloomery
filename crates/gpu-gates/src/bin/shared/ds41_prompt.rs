//! The V4.1 gates' own prompt: row 0 of `tools/ref/prompts.tsv`, "The capital of France is", as the
//! V4.1 file's tokenizer reads it. The V4.1 gates that need only a prompt to run read it here, and
//! no reference set.

/// Five ids and no BOS (the file sets `tokenizer.ggml.add_bos_token` to false). The V4.1 profile's
/// `REF_TOKENS` (`tools/ref/models/deepseek41.sh`) is their owner.
pub const DS41_PROMPT0: [u32; 5] = [671, 6102, 294, 8760, 344];
