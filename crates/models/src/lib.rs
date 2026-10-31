//! What a model is, as data: the architecture's layers, mixers, feed-forward
//! blocks, residual scheme and chat surface, read once from a file's headers
//! by its family's reader (`model::arch::<family>::spec`). A layer program
//! runs a [`ModelSpec`]; the coverage check ([`needs`], and the available
//! table that joins it) says which of its parts no program here runs.
//!
//! This crate is host-only and depends on the GGUF reader alone among the
//! workspace's crates: no family reader and no kernel lives here, only the
//! vocabulary the readers write and the programs read, and the one reader of
//! a row-list head's file that every draft shares ([`headpick`]). A fact the
//! file does not carry is a field the reader fills from a cited constant,
//! never a default here.
//!
//! Counts are `u32`, the files' key type. A field with a `None` or an empty
//! list names what it means in its own comment.

mod encoder;
mod formats;
pub mod headpick;
mod need;
mod role;
pub mod shape;

pub use encoder::EncoderSpec;
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
    /// The n-gram embedding dimensions; `None`: no layer carries an
    /// [`Extra::Engram`] or [`Extra::Ple`] site.
    pub engram: Option<EngramSpec>,
    /// What an encoder carries past its trunk; `None`: the model generates
    /// tokens.
    pub encoder: Option<EncoderSpec>,
    /// The head's logit epilogue; `None`: the output projection's values are
    /// the logits.
    pub logit_epilogue: Option<LogitEpilogue>,
    pub chat: ChatSpec,
}

/// What the head runs on its logits after the output projection, in the
/// graph's order: the scale, then the softcap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogitEpilogue {
    /// `logit_scale`: every logit's factor.
    pub scale: f32,
    /// `final_logit_softcapping`: `cap·tanh(x/cap)`; 0: no cap.
    pub softcap: f32,
}

/// The architectures a reader exists for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Arch {
    Deepseek41,
    Deepseek4,
    Qwen3Moe,
    Qwen35Moe,
    /// Qwen3.5's dense trunk: the qwen35moe layers with a dense FFN.
    Qwen35,
    Qwen4Exp,
    Glm5Next,
    MiMo2,
    /// MiniMax-M3: GQA layers with a block-max sparse selector over a dense
    /// lead, routed experts with a shared one, `swigluoai` everywhere.
    MiniMaxM3,
    NemotronH,
    /// Nemotron-H with latent mixture-of-experts layers in place of the
    /// dense MLP ones.
    NemotronHMoe,
    /// Solar-Open2: Kimi delta attention layers with a NoPE GQA layer every fourth, routed
    /// experts with a shared one in every layer.
    SolarOpen2,
    /// Raon-1: Raon delta attention layers (decoupled erase and write gates) with a gated NoPE
    /// latent-attention layer every few, `situ` experts with a shared one, Raon_mHC streams.
    Raon1,
    GemmaEmbedding,
    GemmaEmbedding2,
    /// Flint: dense lead, sigmoid MoE with one shared MLP, and independently
    /// masked sliding attention and RoPE. No executable program is registered.
    Flint,
    /// Muse Spark: GQA layers whose sliding half ropes and full half does not, all of them routed
    /// in a latent, no dense layer.
    MuseSpark,
    /// A dense Llama: GQA layers with the pairwise (NORM) rope and a SwiGLU
    /// feed-forward block of `feed_forward_length`, on an untied head.
    Llama,
    /// LFM3: gated short-conv layers and GQA layers (sliding or full) in one trunk, a dense
    /// SwiGLU block in every layer, sandwich norms, a tied head.
    Lfm3,
    /// Mistral Large 4: MLA with no query latent, SSMax, a lightning indexer over keys from
    /// the layer input, KV sharing and NoPE layers the file's arrays name, over softmax-routed
    /// experts with one shared. No program runs it yet.
    MistralLarge4,
}

impl Arch {
    /// The `general.architecture` string the architecture is named by. A file
    /// may spell it otherwise: a `glm5-next` file reads as [`Arch::Glm5Next`],
    /// whose name is `glm5next`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Arch::Deepseek41 => "deepseek41",
            Arch::Deepseek4 => "deepseek4",
            Arch::Qwen3Moe => "qwen3moe",
            Arch::Qwen35Moe => "qwen35moe",
            Arch::Qwen35 => "qwen35",
            Arch::Qwen4Exp => "qwen4exp",
            Arch::Glm5Next => "glm5next",
            Arch::MiMo2 => "mimo2",
            Arch::MiniMaxM3 => "minimax-m3",
            Arch::NemotronH => "nemotron_h",
            Arch::NemotronHMoe => "nemotron_h_moe",
            Arch::SolarOpen2 => "solar-open2",
            Arch::Raon1 => "raon1",
            Arch::GemmaEmbedding => "gemma-embedding",
            Arch::GemmaEmbedding2 => "gemma-embedding2",
            Arch::Flint => "flint",
            Arch::MuseSpark => "muse-spark",
            Arch::Llama => "llama",
            Arch::Lfm3 => "lfm3",
            Arch::MistralLarge4 => "mistral-large4",
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
    /// A Mamba-2 mixer: a short conv, then a selective scan of [`Mamba2::n_heads`]
    /// heads of [`Mamba2::head_dim`] values in [`Mamba2::n_groups`] groups.
    Mamba2(Mamba2),
    /// A gated short convolution: `C ⊙ conv(B ⊙ x)` over the projection's three chunks.
    ShortConv(ShortConv),
    /// This layer has no token mixer. Its feed-forward block is the layer.
    None,
}

impl Mixer {
    /// The mixer's kind as a refusal names it.
    #[must_use]
    pub fn what(&self) -> &'static str {
        match self {
            Mixer::Gqa(_) => "a GQA mixer",
            Mixer::Latent(_) => "a latent-attention mixer",
            Mixer::DeltaRule(_) => "a delta-rule mixer",
            Mixer::Mamba2(_) => "a Mamba-2 mixer",
            Mixer::ShortConv(_) => "a short-conv mixer",
            Mixer::None => "no token mixer",
        }
    }
}

/// A Mamba-2 mixer. The widths are the header's `ssm.*` keys, and the two
/// products the input projection and the conv are built at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mamba2 {
    /// `ssm.inner_size`.
    pub d_inner: u32,
    /// `ssm.state_size`.
    pub d_state: u32,
    /// `ssm.time_step_rank`: the scan's heads.
    pub n_heads: u32,
    /// `d_inner / n_heads`.
    pub head_dim: u32,
    /// `ssm.group_count`.
    pub n_groups: u32,
    /// `ssm.conv_kernel`.
    pub d_conv: u32,
    /// Input-projection columns: `2 * d_inner + 2 * n_groups * d_state + n_heads`.
    pub d_in_proj: u32,
}

/// A gated short-convolution mixer. The input projection writes three chunks of `hidden` values,
/// `B | C | x` in that order; the mixer's output is `C ⊙ conv(B ⊙ x)`, the conv causal and
/// depthwise (one filter of `taps` values per channel, the oldest tap first), with no bias and
/// no activation. The output projection follows outside the mixer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShortConv {
    /// `shortconv.l_cache`: the filter's taps, the current position's among them.
    pub taps: u32,
}

/// Grouped-query attention.
#[derive(Clone, Debug, PartialEq)]
pub struct Gqa {
    /// `attention.head_count`: query heads.
    pub heads: u32,
    /// `attention.head_count_kv`; `heads` is a multiple of it.
    pub kv_heads: u32,
    /// `attention.key_length`: key and score values per head.
    pub head_dim: u32,
    /// `attention.value_length`: value values per head; equal to `head_dim`
    /// in every architecture but MiMo-V2's, whose value projection writes a
    /// narrower head.
    pub value_dim: u32,
    pub rope: Rope,
    /// A per-head RMS gain on q and k before the rope.
    pub qk_norm: bool,
    /// A per-head gainless RMS norm on the heads' outputs, at the block
    /// norm's epsilon, before the output gate.
    pub o_norm: bool,
    /// The gate on the attention output; [`OutGate::None`]: the output is not gated.
    pub out_gate: OutGate,
    /// `None`: every position is attended.
    pub select: Option<Selector>,
    /// The raw positions attended; `None`: every position. An encoder's
    /// layers keep `None` and carry the reach in [`crate::EncoderSpec::reach`].
    pub window: Option<u32>,
    /// A per-head softmax sink (`attn_sinks`).
    pub sinks: bool,
    /// A multiplier on the attention's value rows; `None`: 1.
    pub value_scale: Option<f32>,
}

/// Where a GQA layer's output gate comes from; the attention output is multiplied by its sigmoid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutGate {
    /// No gate.
    None,
    /// `attn_q` also writes a per-head gate beside the query, interleaved with it.
    Packed,
    /// A projection of the layer input of its own (`attn_gate`), one value per query-head
    /// channel.
    Proj,
    /// A projection of the layer input of its own (`attn_gate`) with one value per query head:
    /// a head's gate multiplies every value of that head alike.
    ProjHead,
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
    /// `attention.q_lora_rank`: the query latent's values. 0: the query is the
    /// hidden row's own projection, no query latent on the path.
    pub q_lora: u32,
    /// The cached row's values.
    pub latent: u32,
    pub up: LatentUp,
    /// `None`: no rope.
    pub rope: Option<Rope>,
    /// A per-head RMS norm on the query, without a gain.
    pub q_head_norm: bool,
    /// SSMax: every query value of head `h` is scaled by
    /// `s_h·ln(pos + 1) + b_h` (`attn_ssmax`), the whole cached-width row after
    /// its parts are joined, and the indexer's query by its own pair
    /// (`indexer.ssmax`, one per index head) — the same op, one code path.
    pub ssmax: bool,
    /// Key values beside the latent in the cached row, which no rope turns: shared by every head
    /// (`qk_rope_head_dim` of a model that rotates none). 0: the row is the latent alone.
    pub pe: u32,
    /// Who writes the cached rows this layer reads ([`Source::From`]:
    /// `attention.kv_source_map`, the layer stores nothing); the source's
    /// rope-ness is the reader's to hold equal.
    pub cache: Source<()>,
    /// The gate on the heads' outputs before the output projection; [`OutGate::Proj`] is a
    /// projection of the layer input of its own (`attn_gate`). [`OutGate::Packed`] and
    /// [`OutGate::ProjHead`] are not a latent layer's.
    pub out_gate: OutGate,
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
    /// A top-k over index keys, one per compressed row of the layer's stream,
    /// or one per raw position where the layer compresses nothing
    /// ([`Latent::compress`] `None`).
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
        rule: PoolRule,
    },
    /// A top-k of blocks of the raw positions: every position's key is scored
    /// against each index head, a block scores the largest of its positions'
    /// scores, and the query's own block is kept inside the `top_k`. Each key
    /// head selects on its own, and the keys are RMS-normed and rotated.
    BlockMax {
        /// `attention.indexer.head_count`: index heads, one for each key head.
        heads: u32,
        /// `attention.indexer.key_length`: values per index head and key.
        d: u32,
        /// `attention.indexer.top_k`: blocks kept, the query's own among them.
        top_k: u32,
        /// `attention.indexer.block_size`: positions a block holds.
        block: u32,
        /// `attention.indexer.local_blocks`: blocks, the query's own first,
        /// kept whatever they score.
        local: u32,
        /// The rotation of the index query and key.
        rope: Rope,
    },
}

/// How a token-pool selector makes a pool's key and scores it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PoolRule {
    /// The pool's key a learned convex mix of its members' keys, a LayerNorm
    /// with a gain and a bias (epsilon `key_eps`) on it, no rope, the heads'
    /// scores weighed by a learned projection.
    Learned { key_eps: f32 },
    /// The pool's key the mean of its members' raw keys, an RMS gain norm on
    /// it and on each query, both rotated by `rope` (the key at its pool's
    /// first position), the heads' rectified scores summed.
    Mean { rope: Rope },
}

/// Where a layer's index keys come from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IndexKeys {
    /// Projected from the pooled latent rows.
    FromRows,
    /// Pooled from the layer input by the indexer's own compressor.
    Compressor(Compressor),
    /// Projected from the layer input over the raw positions: the projection
    /// of the input, a LayerNorm with a gain and a bias on it (at `norm_eps`,
    /// `attention.layer_norm_epsilon`), the layer's own [`Latent::rope`]
    /// turning `dims` values of each head (the file's first `dims`; V4.1's
    /// index keys turn theirs at the head's tail, so a body moves the file's
    /// leading rope rows there before use; none on a layer whose rope is
    /// `None`), then the Hadamard rotation, the
    /// keys stored f16 — DeepSeek's
    /// lightning index keys, one per position. The scoring query is the same
    /// form: the LayerNorm-normed input projected by the indexer's own
    /// `attn_q_b`.
    FromInput {
        /// The LayerNorm's epsilon on the key.
        norm_eps: f32,
    },
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
    Gdn { khead_map: KHeadMap, gate: GdnGate },
    /// A per-channel decay, a conv per q, k and v, the output gated. `gate` is
    /// how the decay is made, `beta_scale` the scale on `sigmoid(b_proj x)`.
    Kda { gate: KdaGate, beta_scale: f32 },
    /// A Kda decay and conv with an erase gate `sigmoid(b)` apart from the write gate
    /// `sigmoid(b + w)` (a value scale `exp(logsigmoid(b + w) - logsigmoid(b))`, `w` the
    /// per-head `ssm_rda` projection), the output gated by a full-width projection of its own
    /// (`ssm_g`) after a per-head RMS norm. `gate` is how the decay is made.
    Rda { gate: KdaGate },
}

/// How a KDA layer's decay gate is made.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KdaGate {
    /// `g = lower_bound * sigmoid(exp(A_log) * (x + dt_bias))`, `lower_bound`
    /// below 0 (GLM-5.3-Flash).
    Sigmoid { lower_bound: f32 },
    /// `g = -exp(A_log) * softplus(x + dt_bias)`, unbounded (Solar-Open2).
    Softplus,
}

/// The activation of z that gates a GDN layer's normed output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnGate {
    Silu,
    Sigmoid,
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
    /// `rope_freqs.weight`, `dims / 2` factors: pair `i`'s frequency is divided by factor `i`
    /// (ggml's `freq_factors`), so a pair with a very large factor turns nothing; `false`: no
    /// factors.
    pub factors: bool,
}

/// How a rope pairs a head's values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeMode {
    /// GPT-J pairs over the last `dims` values; with `dims == head_dim` it is
    /// llama's NORM rope, pairs `(2i, 2i + 1)` over the whole head.
    NormTail,
    /// NeoX halves over the first `dims` values.
    Neox,
    /// NeoX halves, pair `i`'s position taken from its section's axis.
    Imrope { sections: [u32; 4] },
    /// No value is rotated.
    None,
}

/// YaRN's parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Yarn {
    pub factor: f32,
    /// The original context, positions.
    pub orig_ctx: u32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    /// The net factor on the rotated values — ggml's `mscale`, the product of
    /// the `attn_factor` the caller passes and YaRN's own `1 + 0.1·ln factor`:
    /// the reader states the value the whole recipe applies, so no consumer
    /// works an `attn_factor` out again. A model that runs YaRN without
    /// magnitude scaling states 1.
    pub magnitude: f32,
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
    /// `moe_latent_size`: the routed experts read and write this many values,
    /// the layer's input projected down to it before them (`ffn_latent_down`)
    /// and their weighted sum projected back up after (`ffn_latent_up`);
    /// `None`: they read and write the hidden row.
    pub latent: Option<u32>,
    /// `ffn_pre_exp_norm`: a weighted RMS norm on the experts' input — the
    /// latent, when they run in one — at the block norm's epsilon; `false`:
    /// the projection's output is their input.
    pub pre_exp_norm: bool,
    /// `ffn_post_exp_norm`: the experts' aggregated output scaled by its
    /// weight, then normalized at this epsilon, before the router's weights
    /// apply; `None`: no norm.
    pub post_exp_norm: Option<f32>,
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
    /// The divisor policy on the kept, unbiased scores before scaling.
    pub norm: RouterNorm,
    /// The factor on the kept weights.
    pub scale: f32,
    /// The experts come from a token table (`ffn_gate_tid2eid`).
    pub hash: bool,
}

/// The normalization of the selected, unbiased router scores before scaling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouterNorm {
    /// Keep each selected score unchanged.
    None,
    /// Divide by the selected-score sum, guarded by adding `1e-20`.
    SelectedSum,
    /// Flint's fixed policy: divide by `max(sum_selected, 2^-14)`, not an epsilon-added sum.
    SelectedSumFloor,
}

impl RouterNorm {
    /// Map the legacy normalization flag to its divisor policy.
    #[must_use]
    pub const fn from_legacy_bool(norm: bool) -> Self {
        if norm { Self::SelectedSum } else { Self::None }
    }
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
    /// `relu(x)^2`, with no gate.
    ReluSqr,
    /// `gelu(g)·u`, GELU in its tanh form.
    GeGlu,
    /// `swigluoai`: with `x = min(g, limit)` and `y = clamp(u, ±limit)`,
    /// `x·sigmoid(alpha·x)·(y + 1)`. The gate is clamped from above only,
    /// before the sigmoid, and the up path carries a unit offset.
    SwigluOai { alpha: f32, limit: f32 },
    /// `situ`: `gate_beta·tanh(g/gate_beta)·sigmoid(g) · value_beta·tanh(u/value_beta)`; both
    /// betas are above 0 and finite.
    Situ { gate_beta: f32, value_beta: f32 },
}

impl Act {
    /// The activation as a refusal names it.
    #[must_use]
    pub fn what(self) -> &'static str {
        match self {
            Act::SwiGlu { .. } => "SwiGLU",
            Act::ReluSqr => "squared ReLU",
            Act::GeGlu => "GELU GLU",
            Act::SwigluOai { .. } => "swigluoai",
            Act::Situ { .. } => "situ",
        }
    }

    /// The clamp the combine kernels take: 0, the plain combine, when the
    /// activation has no limit, else the limit. Only SwiGLU has a clamp; a
    /// caller matches on [`Act`] and refuses the other arms by name.
    #[must_use]
    pub fn swiglu_limit(self) -> f32 {
        match self {
            Act::SwiGlu { limit } => limit.unwrap_or(0.0),
            other => panic!("{} has no SwiGLU clamp", other.what()),
        }
    }
}

/// How a layer's outputs join the residual.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Residual {
    Plain,
    /// The owning spec's [`HcSpec`].
    Hc,
    /// Per-channel α·stream + β·branch (`attn_resid_alpha`/`beta`,
    /// `ffn_resid_alpha`/`beta`): each sublayer's output is normalized at
    /// `eps` before it joins — the attention's weightlessly, the
    /// feed-forward's scaled by its `post_ffw_norm` weight then normalized —
    /// and α² + β² = 1 per channel, resolved at conversion.
    AlphaBeta {
        eps: f32,
    },
    /// Each sublayer's output goes through its own weighted RMS norm at `eps` before it joins
    /// the stream: `x += add[i] · gain ⊙ rms_norm(out)`, `i` 0 for the mixer (`post_attention_norm`)
    /// and 1 for the feed-forward block (`post_ffw_norm`).
    Sandwich {
        eps: f32,
        add: [f32; 2],
    },
}

/// Hyper-connections: the residual carried in several streams.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HcSpec {
    /// `hyper_connection.count`.
    pub streams: u32,
    pub kind: HcKind,
}

/// How a sublayer reads the streams and writes its output back.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HcKind {
    /// Manifold-constrained: pre, post and comb mixes from one projection of
    /// the streams, the comb made doubly stochastic by Sinkhorn rounds.
    Mhc {
        /// `hyper_connection.sinkhorn_iterations`.
        sinkhorn: u32,
        /// `hyper_connection.epsilon`: the floor on the mixes.
        eps: f32,
        mix: HcMix,
        collapse: Collapse,
    },
    /// Chart-constrained (Raon_mHC): pre, post and a Helmert-chart residual mix from one
    /// projection of the streams scaled by their RMS, every sublayer its own mix. The residual
    /// mix is `(1 - g) I + g Phi` with `g = sigmoid(eta)` and `Phi` the doubly stochastic point
    /// of the chart; the post mix is `magnitude · sigmoid(..)`.
    Chart {
        /// `hyper_connection.epsilon`: added to the streams' mean square.
        eps: f32,
        /// `hyper_connection.magnitude`: the post mix's factor.
        magnitude: f32,
        collapse: Collapse,
    },
    /// Gated-residual: the sublayer's input the mean of the normed streams,
    /// each gated elementwise by a sigmoid through a `rank`-value bottleneck,
    /// its output added to each stream by a scalar sigmoid gate; no mix
    /// between streams. A trained head of the same form (`output_hc_*`)
    /// collapses them.
    Gated {
        /// `hyper_connection.low_rank`.
        rank: u32,
    },
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
    /// A per-layer n-gram embedding, before the layer's attention mix.
    Ple,
}

/// The n-gram embedding dimensions: every site's rows are gathered by a hash
/// of the token's last n-grams, `heads` rows per n-gram size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EngramSpec {
    /// `engram.head_count`, `ple.heads_per_ngram`: rows per n-gram size.
    pub heads: u32,
    /// `engram.max_ngram_size`, `ple.ngram_size`: n-grams of 2 to this many tokens.
    pub max_ngram: u32,
    /// `engram.key_length`, `embedding_length_per_layer_input`: values per gathered row.
    pub key_length: u32,
    pub rule: NgramRule,
}

/// What a site hashes and what it does with the rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NgramRule {
    /// Token ids through the file's token map, the window before the start
    /// padded with the file's pad id; the gated value added to every stream.
    Engram,
    /// Raw token ids; an `eos` in the window, or a position before the start,
    /// turns it and every older slot into `eos`. Each stream's gated value,
    /// normed, goes through a causal depthwise conv of `conv` taps `dilation`
    /// positions apart and a SiLU; the gated value and the conv's output are
    /// both added to that stream.
    Ple {
        /// `ple.eos_token_id`.
        eos: u32,
        /// `ple.image_token_id`: the id an image position hashes as; `None`
        /// when the file has none.
        image: Option<u32>,
        /// `ple.conv_kernel`.
        conv: u32,
        /// The n-gram size (the architecture's rule, no key).
        dilation: u32,
    },
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
    /// V4.1's `<｜DSML｜tool_calls>` blocks.
    Dsml,
    /// GLM's `<tool_call>NAME<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>`.
    GlmXml,
    /// Qwen's `<tool_call><function=NAME><parameter=KEY>…</parameter>
    /// </function></tool_call>`, which MiMo-V2's template teaches.
    QwenXml,
    /// A call on tagged lines: `<|tool_call:start|>NAME`, one `<|tool_arg:start|>KEY<|tool_arg:value|>VALUE
    /// <|tool_arg:end|>` per argument, `<|tool_call:end|>`, which Solar-Open's template teaches.
    TagKv,
    /// Muse Spark's recipient protocol: a turn is `to=<recipient><|message|>` messages ended by
    /// `<|eom|>` (more follow) or `<|eot|>` (the turn's end), `to=self` the reasoning span and
    /// `to=<tool>` a call in ATEM markup (`<atem:function_calls>`).
    MuseSpark,
    /// Mistral's `[TOOL_CALLS]NAME[ARGS]{json}`, one markup per call with no
    /// call id and nothing between calls (Mistral Large 4's template).
    Mistral,
}

impl ToolFormat {
    /// The format a chat template's call markup spells, for the formats a
    /// description names from the template alone (`formats`' table). `None`:
    /// no such markup, so no parser is named for the file.
    #[must_use]
    pub fn of_template(template: &str) -> Option<ToolFormat> {
        formats::tool_of(template)
    }
}

/// The markup of [`ToolFormat::QwenXml`], spelled once: the template table
/// names a file's format by the markers its template teaches, and the
/// server's scanner reads a reply by the same strings.
pub mod function_xml {
    /// Opens a call.
    pub const CALL_OPEN: &str = "<tool_call>";
    /// Closes a call.
    pub const CALL_CLOSE: &str = "</tool_call>";
    /// Opens the call's function, its name to the `>` (`<function=NAME>`).
    pub const FUNCTION_OPEN: &str = "<function=";
    /// Closes the function.
    pub const FUNCTION_CLOSE: &str = "</function>";
    /// Opens a parameter, its name to the `>` (`<parameter=KEY>`).
    pub const PARAM_OPEN: &str = "<parameter=";
    /// Closes a parameter.
    pub const PARAM_CLOSE: &str = "</parameter>";
    /// Ends a value a template frames on lines of its own
    /// (`<parameter=KEY>\nVALUE\n</parameter>`): the newline before the tag
    /// and the one after it are the markup's own.
    pub const PARAM_VALUE_CLOSE: &str = "\n</parameter>\n";
}

/// A reasoning span syntax.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningFormat {
    /// The reasoning between two tags the model writes: `open` starts it and
    /// `close` ends it. The tags are in the model's vocabulary, so the
    /// description names them and the output parser reads them from it.
    ThinkSpan {
        open: &'static str,
        close: &'static str,
    },
}

impl ReasoningFormat {
    /// `<think>…</think>`: V4.1's, GLM's, Qwen's, MiMo's and Nemotron's.
    pub const THINK: ReasoningFormat = ReasoningFormat::ThinkSpan {
        open: "<think>",
        close: "</think>",
    };

    /// The span a chat template opens (`formats`' table). `None`: the
    /// template opens no span, so the output is not split.
    #[must_use]
    pub fn of_template(template: &str) -> Option<ReasoningFormat> {
        formats::span_of(template)
    }
}

/// A draft, read to be paired with a target.
#[derive(Clone, Debug, PartialEq)]
pub enum DraftSpec {
    Block(BlockDraft),
    /// Boxed: the layer's description is several times a block draft's size.
    Mtp(Box<MtpDraft>),
}

/// A next-token (MTP) draft: one layer that reads the target's last hidden
/// state and the embedding of the token the target just emitted, and
/// proposes the token after it through the output matrix; its own output
/// state is the next step's hidden input.
#[derive(Clone, Debug, PartialEq)]
pub struct MtpDraft {
    pub source: MtpSource,
    /// The layer, described as a trunk layer is.
    pub layer: LayerSpec,
    /// The layer's `blk.` index in its file: `block_count` less
    /// `nextn_predict_layers`, the target trunk's length.
    pub index: LayerIdx,
    /// `embedding_length`; the target's too.
    pub hidden: u32,
    /// The target's too.
    pub vocab: u32,
    pub rms_eps: f32,
    /// The layer's hyper-connections; `None`: the layer is
    /// [`Residual::Plain`].
    pub hc: Option<HcSpec>,
    pub input: MtpInput,
    pub head_norm: MtpHeadNorm,
    pub head_rows: headpick::HeadSet,
}

/// Where an MTP layer's tensors are.
#[derive(Clone, Debug, PartialEq)]
pub enum MtpSource {
    /// In the target's own file.
    InFile {
        /// Its `blk.` index there.
        layer: LayerIdx,
    },
    /// In a draft file of its own.
    File {
        /// The first shard's full path: the file's identity.
        first_shard: std::path::PathBuf,
        /// The file's tensor bytes, the header excluded.
        bytes: u64,
        borrows: Borrows,
    },
}

/// Which of the target's tensors a draft file uses in place of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Borrows {
    /// `token_embd`.
    pub embedding: bool,
    /// `output`.
    pub head: bool,
}

/// Why a draft file is not of its target's checkpoint ([`same_checkpoint`]).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CheckpointMismatch {
    /// Both files name their checkpoint, and the names differ.
    #[error(
        "the draft file is of checkpoint {draft:?} and the target of {target:?}: a draft head \
         reads the hidden state of the checkpoint it was made from"
    )]
    Differs { target: String, draft: String },
    /// A file names no checkpoint, so nothing says whose it is.
    #[error("the {file} file has no `general.name`: its checkpoint is unknown")]
    Unnamed { file: &'static str },
}

/// Whether a draft file was made from the checkpoint its target is: the
/// `general.name` of each file (`None`: the key is absent), the names equal.
/// Two checkpoints of one architecture share every key and tensor shape but
/// this name, so no other check tells them apart; a file with no name is
/// refused, never taken for a match.
pub fn same_checkpoint(
    target: Option<&str>,
    draft: Option<&str>,
) -> Result<(), CheckpointMismatch> {
    let target = target.ok_or(CheckpointMismatch::Unnamed { file: "target" })?;
    let draft = draft.ok_or(CheckpointMismatch::Unnamed { file: "draft" })?;
    if target == draft {
        Ok(())
    } else {
        Err(CheckpointMismatch::Differs {
            target: target.to_string(),
            draft: draft.to_string(),
        })
    }
}

/// How an MTP layer's input is made. Every form norms the token's embedding
/// by `nextn.enorm` and the target's row by `nextn.hnorm`, joins them with
/// the embedding's half first, and projects the join by `nextn.eh_proj`
/// (`2 × hidden` values to `hidden`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtpInput {
    /// The target's hyper-connection streams entering its output head, not
    /// collapsed: `hnorm` over all `streams × hidden` values as one row, then
    /// each stream joined and projected, one input stream per stream.
    Streams,
    /// The row the target's head reads, its streams collapsed and normed by
    /// the output norm: one row joined and projected.
    HeadRow,
    /// The target's last-layer row before its output norm, and the draft's
    /// own row before its head norm: the row the next step reads is taken
    /// ahead of the norm, and `hnorm` is the only norm it meets
    /// (MiMo-V2's graph).
    PreNorm,
}

/// What the MTP layer's row goes through before the output matrix, in place
/// of the target's own head norm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtpHeadNorm {
    /// The layer's own gated-residual head (`nextn.hc_head_*`), of the
    /// target's `output_hc_*` form: it norms and collapses the streams.
    HcHead,
    /// The layer's own RMS gain: `nextn.shared_head_norm`, or MiMo-V2's
    /// `blk.N.layer_output_norm`.
    Rms,
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
    /// The first routed mixture: the one every routed layer of a file shares
    /// (the readers hold them to one count and width). `None`: no layer
    /// routes, the whole feed-forward side is dense.
    #[must_use]
    pub fn routed(&self) -> Option<&Moe> {
        self.layers.iter().find_map(LayerSpec::moe)
    }

    /// The first GQA layer's mixer: the head shape and rope the trunk's
    /// attention layers share. `None`: no layer attends.
    #[must_use]
    pub fn gqa(&self) -> Option<&Gqa> {
        self.layers.iter().find_map(|l| match &l.mixer {
            Mixer::Gqa(g) => Some(g),
            _ => None,
        })
    }

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

#[cfg(test)]
mod tests {
    use super::{Act, CheckpointMismatch, same_checkpoint};

    const MOPD: &str = "MiMo-V2.6-Flash-MOPD";
    const RL: &str = "MiMo-V2.6-Flash-RL";

    /// A draft of the target's checkpoint is accepted; a draft of the other
    /// checkpoint is refused naming both.
    #[test]
    fn a_draft_of_another_checkpoint_is_refused_naming_both() {
        assert_eq!(same_checkpoint(Some(MOPD), Some(MOPD)), Ok(()));
        assert_eq!(same_checkpoint(Some(RL), Some(RL)), Ok(()));
        let err = same_checkpoint(Some(MOPD), Some(RL)).expect_err("RL beside MOPD is refused");
        assert_eq!(
            err,
            CheckpointMismatch::Differs {
                target: MOPD.to_string(),
                draft: RL.to_string(),
            }
        );
        let text = err.to_string();
        assert!(text.contains(MOPD) && text.contains(RL), "{text}");
        assert!(same_checkpoint(Some(RL), Some(MOPD)).is_err());
    }

    /// A file with no name proves no checkpoint, on either side.
    #[test]
    fn a_file_without_a_name_is_refused() {
        assert_eq!(
            same_checkpoint(None, Some(RL)),
            Err(CheckpointMismatch::Unnamed { file: "target" })
        );
        assert_eq!(
            same_checkpoint(Some(RL), None),
            Err(CheckpointMismatch::Unnamed { file: "draft" })
        );
        assert!(same_checkpoint(None, None).is_err());
    }

    /// No limit is the plain combine's 0; a limit is itself.
    #[test]
    fn swiglu_limit_is_zero_without_a_limit_and_the_limit_with_one() {
        assert_eq!(Act::SwiGlu { limit: None }.swiglu_limit(), 0.0);
        assert_eq!(Act::SwiGlu { limit: Some(7.0) }.swiglu_limit(), 7.0);
    }

    /// `swigluoai` is not the SwiGLU clamp: its limit clamps the gate from above and the up path on
    /// both sides, so a caller asking it for the SwiGLU limit is refused by the activation's name.
    #[test]
    #[should_panic(expected = "swigluoai has no SwiGLU clamp")]
    fn swiglu_limit_on_swigluoai_names_the_activation() {
        let _ = Act::SwigluOai {
            alpha: 1.702,
            limit: 7.0,
        }
        .swiglu_limit();
    }

    /// A squared ReLU has no SwiGLU clamp. Calling the clamp on it is the
    /// caller's miss: the message names the activation.
    #[test]
    #[should_panic(expected = "squared ReLU has no SwiGLU clamp")]
    fn swiglu_limit_on_squared_relu_names_the_activation() {
        let _ = Act::ReluSqr.swiglu_limit();
    }
}
