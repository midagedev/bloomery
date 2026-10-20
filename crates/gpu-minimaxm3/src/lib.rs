//! MiniMax-M3 on the card. Every layer is grouped-query attention (64 query heads over 4 key heads of
//! 128 values, a partial NEOX rope); a sparse layer attends the blocks its indexer picks.
//!
//! [`attn`] is the sparse attention (MSA): the indexer's head kernel and key cache, the block-max
//! selector and the flash over the selected blocks. The dense layers attend through the common GQA
//! owners with the same head kernel.

pub mod attn;
