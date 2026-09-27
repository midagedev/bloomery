//! What a tensor does in a decode step.

use std::fmt;

/// What a tensor does in a decode step; the role decides its device.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Role {
    /// Attention, its compressor and its indexer: on the layer's card.
    Attention,
    /// Hyper-connection mixing: on the layer's card.
    HyperConnection,
    /// The router weight and its selection bias: on the layer's card.
    Router,
    /// The pre-FFN norm: on the layer's card.
    FfnNorm,
    /// The shared expert: on the layer's card.
    SharedExpert,
    /// A dense FFN's gate, up and down (a layer without a router): on the
    /// layer's card.
    DenseFfn,
    /// Engram weights every token reads in full: on the layer's card.
    EngramDense,
    /// An engram gate's gain vectors (`engram_k`, `engram_q`), which the card
    /// multiplies by and no gemv reads: on the layer's card as f32 values,
    /// whatever type the file stores them in.
    EngramGain,
    /// An engram table: on NVMe, a few rows gathered per token.
    EngramTable,
    /// A hash router's token table (`ffn_gate_tid2eid`): on the host, the one
    /// row of expert ids a token's id selects gathered per token.
    HashTable,
    /// A routed expert stack: split by the expert rule between the layer's card and the host.
    RoutedExperts,
    /// The token embedding: on the host, one row gathered per token — or
    /// whole on the first card when the plan's [`Card::token_embedding`] says so.
    TokenEmbedding,
    /// The output head and its norm: on the head card.
    Head,
    /// In the file, never read by text decode: a tensor of the decode graph's
    /// own layers that no step reads.
    Unread,
    /// In the file, never loaded: weights of a graph this engine does not run
    /// (a multi-token-prediction head's `nextn.*`, `*.mtp.*`). Its layer, if
    /// any, may lie past the model's layers; no stage is asked for it.
    Unused,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Role::Attention => "attn",
            Role::HyperConnection => "hc",
            Role::Router => "router",
            Role::FfnNorm => "ffn_norm",
            Role::SharedExpert => "shexp",
            Role::DenseFfn => "dense_ffn",
            Role::EngramDense => "engram_dense",
            Role::EngramGain => "engram_gain",
            Role::EngramTable => "engram_table",
            Role::HashTable => "hash_table",
            Role::RoutedExperts => "routed",
            Role::TokenEmbedding => "token_embd",
            Role::Head => "head",
            Role::Unread => "unread",
            Role::Unused => "unused",
        })
    }
}
