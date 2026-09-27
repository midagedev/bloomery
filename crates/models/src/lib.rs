//! What a model is, as data: the architecture's layers, mixers, feed-forward
//! blocks, residual scheme and chat surface, read once from a file's headers
//! by its family's reader (`model::arch::<family>::spec`). A layer program
//! runs a [`ModelSpec`]; the coverage check ([`needs`], and the available
//! table that joins it) says which of its parts no program here runs.
//!
//! This crate is host-only and depends on the GGUF reader alone: no reader
//! and no kernel lives here, only the vocabulary the readers write and the
//! programs read. A fact the file does not carry is a field the reader fills
//! from a cited constant, never a default here.
//!
//! Counts are `u32`, the files' key type. A field with a `None` or an empty
//! list names what it means in its own comment.

mod need;
mod role;

pub use need::{Need, Unimplemented, needs};
pub use role::Role;

/// A trunk layer's index, 0-based.
pub type LayerIdx = u32;

/// One model: its trunk, the draft layers it carries, and the model-wide parts.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelSpec {
    /// `general.architecture`.
    pub arch: Arch,
    /// `embedding_length`: values per residual row.
    pub hidden: u32,
    /// `vocab_size`, else the length of `tokenizer.ggml.tokens`.
    pub vocab: u32,
    /// `context_length`: the positions the model was trained to.
    pub ctx_train: u32,
    /// `attention.layer_norm_rms_epsilon`: every block norm's.
    pub rms_eps: f32,
    /// The trunk: `block_count` less the next-token layers, in order.
    pub layers: Vec<LayerSpec>,
    /// The next-token (MTP) layers the file carries; empty: none.
    pub mtp: Vec<LayerSpec>,
    /// The hyper-connections; `None`: no layer is [`Residual::Hc`].
    pub hc: Option<HcSpec>,
    /// The engram dimensions; `None`: no layer carries [`Extra::Engram`].
    pub engram: Option<EngramSpec>,
    pub chat: ChatSpec,
}

/// The architectures a reader exists for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Arch {
    Deepseek41,
    Deepseek4,
    Qwen3Moe,
    Qwen35Moe,
    Glm5Next,
}

impl Arch {
    /// The `general.architecture` string.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Arch::Deepseek41 => "deepseek41",
            Arch::Deepseek4 => "deepseek4",
            Arch::Qwen3Moe => "qwen3moe",
            Arch::Qwen35Moe => "qwen35moe",
            Arch::Glm5Next => "glm5next",
        }
    }
}

/// One layer: its token mixer, its feed-forward block, how its output joins
/// the residual, and what else it runs.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerSpec {
    pub mixer: Mixer,
    pub ffn: Ffn,
    pub residual: Residual,
    /// Empty but for the layers that carry one.
    pub extras: Vec<Extra>,
}

impl LayerSpec {
    /// Whether this layer writes the compressed rows it (and later layers) read.
    #[must_use]
    pub fn owns_rows(&self) -> bool {
        self.latent()
            .and_then(|a| a.compress.as_ref())
            .is_some_and(|c| matches!(c.rows, Source::Own(_)))
    }

    /// Whether this layer writes index keys.
    #[must_use]
    pub fn owns_keys(&self) -> bool {
        matches!(
            self.latent().and_then(|a| a.select.as_ref()),
            Some(Selector::StreamTopK {
                keys: Source::Own(_),
                ..
            })
        )
    }

    /// The layers whose compressed rows, index keys and top-k list layer `l`
    /// (this one) reads, `Own` resolved to `l`; `None` for a layer that reads
    /// no selected stream.
    #[must_use]
    pub fn sources(&self, l: LayerIdx) -> Option<[LayerIdx; 3]> {
        let a = self.latent()?;
        let c = a.compress.as_ref()?;
        let Some(Selector::StreamTopK { keys, list, .. }) = a.select.as_ref() else {
            return None;
        };
        Some([c.rows.layer(l), keys.layer(l), list.layer(l)])
    }

    /// The latent mixer, if this layer's mixer is one.
    #[must_use]
    pub fn latent(&self) -> Option<&Latent> {
        match &self.mixer {
            Mixer::Latent(a) => Some(a),
            _ => None,
        }
    }

    /// The routed mixture, if this layer's feed-forward block is one.
    #[must_use]
    pub fn moe(&self) -> Option<&Moe> {
        match &self.ffn {
            Ffn::Moe(m) => Some(m),
            Ffn::Dense { .. } => None,
        }
    }
}

/// How a layer mixes positions.
#[derive(Clone, Debug, PartialEq)]
pub enum Mixer {
    Gqa(Gqa),
    Latent(Latent),
    DeltaRule(DeltaRule),
}

/// Grouped-query attention.
#[derive(Clone, Debug, PartialEq)]
pub struct Gqa {
    /// `attention.head_count`: query heads.
    pub heads: u32,
    /// `attention.head_count_kv`; `heads` is a multiple of it.
    pub kv_heads: u32,
    /// `attention.key_length`, which the value length equals: values per head.
    pub head_dim: u32,
    pub rope: Rope,
    /// A per-head RMS gain on q and k before the rope.
    pub qk_norm: bool,
    /// `attn_q` also writes a per-head gate; the output is multiplied by its sigmoid.
    pub out_gate: bool,
}

impl Gqa {
    /// Query heads per key and value head.
    #[must_use]
    pub fn group(&self) -> u32 {
        self.heads / self.kv_heads
    }
}

/// Latent attention: one latent row per position, shared by every head.
#[derive(Clone, Debug, PartialEq)]
pub struct Latent {
    /// `attention.head_count`.
    pub heads: u32,
    /// `attention.q_lora_rank`: the query latent's values.
    pub q_lora: u32,
    /// The cached row's values.
    pub latent: u32,
    pub up: LatentUp,
    /// `None`: no rope.
    pub rope: Option<Rope>,
    /// A per-head RMS norm on the query, without a gain.
    pub q_head_norm: bool,
    pub out: LatentOut,
    /// The raw positions attended; `None`: every position.
    pub window: Option<u32>,
    /// A per-head softmax sink (`attn_sinks`).
    pub sinks: bool,
    /// `None`: no compressed stream.
    pub compress: Option<Compress>,
    /// `None`: no top-k — the whole compressed stream, or every position.
    pub select: Option<Selector>,
}

/// How the latent becomes each head's key and value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatentUp {
    /// The latent is each head's key and value.
    KeqV,
    /// Absorbed up-projections: `qk` and `v` values per head.
    Absorbed { qk: u32, v: u32 },
}

/// How the heads' outputs become the layer's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatentOut {
    /// `groups` blocks of the head outputs, each to `rank` values, then to `hidden`.
    Grouped { groups: u32, rank: u32 },
    /// One projection of every head's output to `hidden`.
    Plain,
}

/// A compressed stream a latent layer attends.
#[derive(Clone, Debug, PartialEq)]
pub struct Compress {
    /// Tokens pooled into one row, above 0.
    pub ratio: u32,
    /// Who writes the rows.
    pub rows: Source<Compressor>,
}

/// A compressor a layer owns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Compressor {
    /// It pools with scores (`…_gate`).
    pub gated: bool,
    /// It adds a position table to the scores (`…_ape`).
    pub ape: bool,
    /// Its groups overlap: it projects to two rows' width.
    pub overlap: bool,
}

/// Who writes what a layer reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source<T> {
    /// This layer writes it.
    Own(T),
    /// An earlier layer's `Own`.
    From(LayerIdx),
}

impl<T> Source<T> {
    /// The writing layer, `l` being the reader.
    #[must_use]
    pub fn layer(&self, l: LayerIdx) -> LayerIdx {
        match self {
            Source::Own(_) => l,
            Source::From(s) => *s,
        }
    }
}

/// How a latent layer picks the compressed rows it attends.
#[derive(Clone, Debug, PartialEq)]
pub enum Selector {
    /// A top-k over index keys, one per compressed row.
    StreamTopK {
        /// `attention.indexer.head_count`.
        heads: u32,
        /// `attention.indexer.key_length`: values per key.
        d: u32,
        /// `attention.indexer.top_k`: rows kept per query.
        k: u32,
        /// Who writes the index keys.
        keys: Source<IndexKeys>,
        /// Who scores and keeps the list.
        list: Source<()>,
        /// `None`: every row is a candidate.
        candidates: Option<Candidates>,
    },
    /// A top-k over pooled keys of the raw positions.
    TokenPool {
        heads: u32,
        d: u32,
        /// Tokens kept: `top_k / pool` pools and a `pool − 1` tail.
        top_k: u32,
        /// Tokens per pooled key.
        pool: u32,
        /// The key LayerNorm's epsilon.
        key_eps: f32,
    },
}

/// Where a layer's index keys come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexKeys {
    /// Projected from the pooled latent rows.
    FromRows,
    /// Pooled from the layer input by the indexer's own compressor.
    Compressor(Compressor),
}

/// A two-level candidate pool: the blocks another layer's index scores rank first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidates {
    /// The layer whose index scores pick the blocks.
    pub source: LayerIdx,
    /// Blocks kept.
    pub blocks: u32,
    /// Compressed rows per block.
    pub block: u32,
}

/// A delta-rule linear-attention layer.
#[derive(Clone, Debug, PartialEq)]
pub struct DeltaRule {
    pub kind: DeltaKind,
    pub k_heads: u32,
    pub v_heads: u32,
    /// Values per key and value head.
    pub d: u32,
    /// Causal depthwise conv taps.
    pub conv: u32,
}

/// Which delta rule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DeltaKind {
    /// A scalar decay per value head, one conv over q|k|v, the output gated by z.
    Gdn { khead_map: KHeadMap },
    /// A per-channel decay bounded below, a conv per q, k and v, the output gated.
    Kda { gate_lower_bound: f32 },
}

/// Which key head a value head reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KHeadMap {
    /// Value head `j` reads key head `j mod k_heads`.
    Tiled,
}

/// A rotary position embedding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rope {
    pub mode: RopeMode,
    /// Values rotated per head.
    pub dims: u32,
    /// The base of θ.
    pub base: f32,
    /// `None`: plain rope.
    pub yarn: Option<Yarn>,
}

/// How a rope pairs a head's values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeMode {
    /// GPT-J pairs over the last `dims` values.
    NormTail,
    /// NeoX halves over the first `dims` values.
    Neox,
    /// NeoX halves, pair `i`'s position taken from its section's axis.
    Imrope { sections: [u32; 4] },
}

/// YaRN's parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Yarn {
    pub factor: f32,
    /// The original context, positions.
    pub orig_ctx: u32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

/// A layer's feed-forward block.
#[derive(Clone, Debug, PartialEq)]
pub enum Ffn {
    /// `feed_forward_length` values.
    Dense {
        ff: u32,
        act: Act,
    },
    Moe(Moe),
}

/// A routed mixture of experts.
#[derive(Clone, Debug, PartialEq)]
pub struct Moe {
    /// `expert_count`.
    pub experts: u32,
    /// `expert_used_count`.
    pub top_k: u32,
    /// `expert_feed_forward_length`: one expert's values.
    pub expert_ff: u32,
    /// The routed experts' activation.
    pub act: Act,
    pub router: Router,
    /// `None`: no shared expert.
    pub shared: Option<Shared>,
}

/// How a token picks and weighs its experts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Router {
    pub score: Score,
    /// A selection bias (`exp_probs_b`) that steers the choice only.
    pub bias: bool,
    /// The kept weights are renormalized to sum to one.
    pub norm: bool,
    /// The factor on the kept weights.
    pub scale: f32,
    /// The experts come from a token table (`ffn_gate_tid2eid`).
    pub hash: bool,
}

/// The router's score function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Score {
    SqrtSoftplus,
    Softmax,
    Sigmoid,
}

/// A shared expert every token runs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shared {
    /// Hidden values.
    pub ff: u32,
    pub act: Act,
    /// Its output is multiplied by `σ(x · ffn_gate_inp_shexp)`.
    pub sigmoid_gate: bool,
}

/// An activation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Act {
    /// `None`: `silu(g)·u`. `Some(L)`: `min(silu(g), L)·clamp(u, ±L)`; an `L`
    /// of at most 1e-6 clamps nothing.
    SwiGlu { limit: Option<f32> },
}

/// How a layer's outputs join the residual.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residual {
    Plain,
    /// The owning spec's [`HcSpec`].
    Hc,
}

/// Hyper-connections: the residual carried in several streams.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HcSpec {
    /// `hyper_connection.count`.
    pub streams: u32,
    /// `hyper_connection.sinkhorn_iterations`.
    pub sinkhorn: u32,
    /// `hyper_connection.epsilon`: the floor on the mixes.
    pub eps: f32,
    pub mix: HcMix,
    pub collapse: Collapse,
}

/// Which mix a sublayer folds its input by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HcMix {
    /// The previous sublayer's.
    Lagged,
    /// Its own.
    Own,
}

/// How the streams become one before the output norm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Collapse {
    /// With the last FFN's unused mix.
    LastMix,
    /// Their unweighted mean.
    Mean,
    /// With a trained head (`output_hc_*`).
    Head,
}

/// What a layer runs besides its mixer and feed-forward block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extra {
    Engram,
}

/// The engram dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EngramSpec {
    /// `engram.head_count`.
    pub heads: u32,
    /// `engram.max_ngram_size`: n-grams of 2 to this many tokens.
    pub max_ngram: u32,
    /// `engram.key_length`: values per gathered row.
    pub key_length: u32,
}

/// What a chat front end needs of the model.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatSpec {
    /// `tokenizer.ggml.pre`.
    pub pre: String,
    /// `tokenizer.chat_template`; `None`: the file has none.
    pub template: Option<String>,
    /// `None`: no parser; a request with tools is refused by name.
    pub tools: Option<ToolFormat>,
    /// `None`: the output is not split.
    pub reasoning: Option<ReasoningFormat>,
}

/// A tool-call syntax.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolFormat {
    Dsml,
}

/// A reasoning span syntax.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningFormat {
    /// `<think>…</think>`.
    ThinkSpan,
}

/// A draft file, read to be paired with a target.
#[derive(Clone, Debug, PartialEq)]
pub enum DraftSpec {
    Block(BlockDraft),
}

/// A block draft: blocks of positions proposed in one pass from the
/// target's features.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockDraft {
    /// `embedding_length`; the target's too.
    pub hidden: u32,
    /// The target's too.
    pub vocab: u32,
    pub rms_eps: f32,
    pub hc: HcSpec,
    pub layers: Vec<LayerSpec>,
    /// `block_size`: positions of one pass, the seed included.
    pub width: u32,
    /// The feature is the mean of the hyper-connection streams entering each.
    pub target_layers: Vec<LayerIdx>,
    /// `tokenizer.ggml.mask_token_id`.
    pub mask_token: u32,
    /// `markov_w1`'s rows; no key carries it.
    pub markov_rank: u32,
}

impl ModelSpec {
    /// The layers before the first routed one.
    #[must_use]
    pub fn dense_lead(&self) -> u32 {
        count_u32(self.layers.iter().take_while(|l| l.moe().is_none()).count())
    }

    /// The layers that route by a token table, in order.
    #[must_use]
    pub fn hash_layers(&self) -> Vec<LayerIdx> {
        self.indices(|l| l.moe().is_some_and(|m| m.router.hash))
    }

    /// The distinct compression ratios in layer order: the first and the next
    /// one, at most two in a file the readers accept.
    #[must_use]
    pub fn ratio_segments(&self) -> Vec<u32> {
        let mut seen = Vec::new();
        for c in self
            .layers
            .iter()
            .filter_map(|l| l.latent()?.compress.as_ref())
        {
            if !seen.contains(&c.ratio) {
                seen.push(c.ratio);
            }
        }
        seen
    }

    /// The layers carrying an engram site, in order.
    #[must_use]
    pub fn engram_sites(&self) -> Vec<LayerIdx> {
        self.indices(|l| l.extras.contains(&Extra::Engram))
    }

    fn indices(&self, pick: impl Fn(&LayerSpec) -> bool) -> Vec<LayerIdx> {
        (0..)
            .zip(&self.layers)
            .filter(|(_, l)| pick(l))
            .map(|(i, _)| i)
            .collect()
    }
}

/// A count this crate holds as `u32`; the readers refuse a file past it.
fn count_u32(n: usize) -> u32 {
    u32::try_from(n).expect("a layer count read from a u32 key")
}
