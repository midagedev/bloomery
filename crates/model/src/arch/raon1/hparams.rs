//! Every `raon1.*` key the description reads — the two dialects' files through one reading —
//! each layer's mixer from `attention.head_count_kv`, and the tensors the file is held to. A key
//! the loader requires is required here, and a value this reader does not run is refused by the
//! key's name. A layer is an RDA layer when its key-head count is 0 and a latent layer when it
//! is 1; the tensors a layer carries must be exactly its kind's and its dialect's
//! ([`Hparams::read`]). The dialects' authority is llama.cpp's `src/models/raon1.cpp` (the
//! released Mini) and `src/models/raon1nano.cpp` (Nano): the Mini file's split stems, `situ`
//! keys and routed blocks, the Nano file's fused stems, per-layer inputs, dense blocks and tied
//! head, each a value the file's own keys and tensors state, the facts none of them states a row
//! of [`crate::arch::rules`]`.

use gguf::Split;
use models::{Act, KdaGate};

use super::names;
use super::roles::{Dialect, block_groups, every_stem};
use crate::arch::delta_mla::{self, Mla, Situ};
use crate::arch::noaux::{self, Experts};
use crate::arch::rules;
use crate::arch::{
    finite_positive, kda, metadata, n_vocab, nextn_layers, optional_f32, optional_usize,
    per_layer_count, positive, refuse_tensor_set, unread_keys_of,
};
use crate::placement::PlacementError;

/// What a layer mixes with, from `attention.head_count_kv`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// 0 key heads: a Raon delta-attention layer.
    Rda,
    /// 1 key head: a gated latent-attention layer without a rope.
    Mla,
}

/// The hyper-connection (Raon_mHC) keys.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hc {
    /// `hyper_connection.count`: residual streams.
    pub streams: usize,
    /// `hyper_connection.epsilon`: added to the streams' mean square before the selector's scale.
    pub eps: f32,
    /// `hyper_connection.magnitude`: the post mix's factor.
    pub magnitude: f32,
}

impl Hc {
    /// A sublayer's selector width: `streams` pre mixes, `streams` post mixes and the residual
    /// mix's `(streams - 1)^2` chart coordinates.
    #[must_use]
    pub fn mix_dim(&self) -> usize {
        2 * self.streams + (self.streams - 1) * (self.streams - 1)
    }

    /// A sublayer's folded chart width: the `mix_dim · streams²` Helmert coordinates over the
    /// whole mix vector, then the `streams²` rho denominators and the `streams²` combination
    /// constants (raon1nano.cpp:73-74).
    #[must_use]
    pub fn chart_dim(&self) -> usize {
        let streams = self.streams;
        self.mix_dim() * streams * streams + 2 * streams * streams
    }
}

/// The hyperparameters of one raon1 file, either dialect's.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// The file's own architecture: which of the reader's two dialects it names.
    pub arch: models::Arch,
    /// `block_count`: the trunk's blocks and the next-token ones.
    pub n_layer: usize,
    /// The trunk: the blocks no next-token layer takes.
    pub n_trunk: usize,
    /// `embedding_length`.
    pub n_embd: usize,
    /// `attention.head_count`, one value a block (a scalar names every block's): the latent
    /// layers' query heads and the delta layers' key and value heads.
    pub heads: Vec<usize>,
    /// The delta layers' head count: every delta layer's is one, the recurrent state's size
    /// (raon1nano.cpp:38-39).
    pub n_head: usize,
    /// `ssm.conv_kernel` and `kda.head_dim`: the delta layers' conv taps and head width.
    pub kda: kda::Keys,
    /// `ssm.time_step_rank`: the delta layers' decay's low-rank width.
    pub decay_rank: usize,
    /// How the delta layers' decay is made: `kda.gate_lower_bound` when the file has it (the
    /// sigmoid form bounded below by it), else the softplus form.
    pub gate: KdaGate,
    pub mla: Mla,
    /// `attention.layer_norm_rms_epsilon`: every block norm's.
    pub rms_eps: f32,
    /// `attention.latent_norm_rms_epsilon`: the two latents' norms'.
    pub latent_eps: f32,
    pub hc: Hc,
    /// The `situ` activation's keys, when the file carries them; the Nano file carries none, and
    /// takes its blocks' activation from the description row.
    pub situ: Option<Situ>,
    /// What the feed-forward blocks take: the `situ` keys' or the row's.
    pub act: Act,
    /// `embedding_length_per_layer_input`: the per-layer input's width; `None`: the file carries
    /// no per-layer input.
    pub ple: Option<usize>,
    /// Whether the file's head is the token embedding (no `output.weight`): the description row
    /// states the tie.
    pub head_tied: bool,
    /// `vocab_size`, else the token list's length.
    pub n_vocab: usize,
    /// `context_length`.
    pub n_ctx_train: usize,
    /// The layers before the first routed one; the trunk's own count when the file routes none
    /// (raon1nano.cpp:23-24: a dense FFN in every layer).
    pub dense_lead: usize,
    /// `feed_forward_length`, the dense layers' width.
    pub dense_ff: usize,
    /// The routed layers' router and experts; `None`: the file routes none.
    pub experts: Option<Experts>,
    /// The tensor dialect the file speaks, from its delta layers' projections.
    pub dialect: Dialect,
    /// Every block's kind, the next-token ones included, from `attention.head_count_kv`.
    pub kinds: Vec<Kind>,
    /// The keys this read took a default for: the key, the value, the line.
    pub defaults: Vec<String>,
}

/// Every `raon1.` key [`Hparams::read`] reads but the routed block's ([`noaux::KEYS`]) and the
/// per-layer input's, by suffix. The delta rule's `ssm.conv_kernel` and `kda.head_dim` are
/// [`kda::read`]'s.
const READ: &[&str] = &[
    "block_count",
    "nextn_predict_layers",
    "context_length",
    "embedding_length",
    "vocab_size",
    "feed_forward_length",
    "leading_dense_block_count",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.key_length_mla",
    "attention.value_length_mla",
    "attention.q_lora_rank",
    "attention.kv_lora_rank",
    "attention.layer_norm_rms_epsilon",
    "attention.latent_norm_rms_epsilon",
    "rope.dimension_count",
    "ssm.conv_kernel",
    "ssm.state_size",
    "ssm.group_count",
    "ssm.time_step_rank",
    "kda.head_dim",
    "kda.gate_lower_bound",
    "hyper_connection.count",
    "hyper_connection.epsilon",
    "hyper_connection.magnitude",
    "activation.situ_beta",
    "activation.situ_linear_beta",
    "embedding_length_per_layer_input",
];

/// The architecture keys and `general.sampling.*` keys of `split` that [`Hparams::read`] does not
/// read: listed, not refused.
#[must_use]
pub fn unread_keys(split: &Split) -> Vec<String> {
    unread_keys_of(split, &[READ, noaux::KEYS].concat())
}

impl Hparams {
    /// Every value from `split`'s headers; a missing required key, a value this reader does not
    /// run, a next-token layer its tensors do not carry, or a layer whose tensors disagree with
    /// its kind, its dialect or the keys is an error naming the key or the tensor.
    pub fn read(split: &Split) -> Result<Hparams, PlacementError> {
        let mut defaults = Vec::new();
        let name = split.architecture().unwrap_or("<missing>");
        let arch = match name {
            "raon1" => models::Arch::Raon1,
            "raon1nano" => models::Arch::Raon1Nano,
            other => {
                return Err(PlacementError::Metadata {
                    key: "general.architecture".to_string(),
                    detail: format!("is {other:?}; the raon1 reader reads raon1 and raon1nano"),
                });
            }
        };
        // The description row: the file's own facts its keys do not state. The Mini dialect's
        // keys state all of them, so its string has no row.
        let row = rules::row(name);
        let n_layer = positive(split, "block_count")?;
        let nextn = nextn_of(split, n_layer, &mut defaults)?;
        let n_trunk = n_layer - nextn;
        let n_embd = positive(split, "embedding_length")?;
        let heads = head_counts(split, n_layer)?;
        let kinds = delta_mla::kinds(split, n_layer, (Kind::Rda, Kind::Mla))?;
        if let Some(l) = (n_trunk..n_layer).find(|&l| matches!(kinds[l], Kind::Rda)) {
            return Err(metadata(
                split,
                "attention.head_count_kv",
                format!(
                    "is 0 at layer {l}: the next-token block is a latent layer, not a delta one \
                     (raon1nano.cpp:461)"
                ),
            ));
        }
        let mla = Mla::read(split)?;
        value_length_agrees(split, &mla)?;
        let dialect = Dialect::of(split, &kinds)?;
        let n_head = delta_head(split, &heads, &kinds)?;
        let kda = kda::read(split, n_head, &mut defaults)?;
        let gate = kda::gate(split, "src/models/raon1.cpp", &mut defaults)?;
        let situ = situ_of(split)?;
        let act = match (&situ, row.and_then(|r| r.ffn_act)) {
            (Some(situ), _) => situ.act(),
            (None, Some(act)) => {
                defaults.push(
                    row.expect("the row whose activation this is")
                        .ffn_act_line()
                        .expect("the row states its activation"),
                );
                act
            }
            (None, None) => {
                return Err(metadata(
                    split,
                    "activation.situ_beta",
                    format!(
                        "is absent, and no description row of {name} states the feed-forward \
                         blocks' activation"
                    ),
                ));
            }
        };
        let ple = per_layer_of(split, row, &mut defaults)?;
        let (experts, dense_lead) = ffn_of(split, n_trunk, &mut defaults)?;
        let head_tied = split.find("output.weight").is_none()
            && row.is_some_and(|r| r.head == rules::HeadWeight::Tied);
        if head_tied {
            defaults.push(
                row.expect("the row that states the tie")
                    .tied_head_line()
                    .expect("the row states its tie"),
            );
        }
        let hp = Hparams {
            arch,
            n_layer,
            n_trunk,
            n_embd,
            heads,
            n_head,
            kda,
            decay_rank: positive(split, "ssm.time_step_rank")?,
            gate,
            mla,
            rms_eps: finite_positive(split, "attention.layer_norm_rms_epsilon")?,
            latent_eps: finite_positive(split, "attention.latent_norm_rms_epsilon")?,
            hc: Hc::read(split)?,
            situ,
            act,
            ple,
            head_tied,
            n_vocab: n_vocab(split)?,
            n_ctx_train: positive(split, "context_length")?,
            dense_lead,
            dense_ff: positive(split, "feed_forward_length")?,
            experts,
            dialect,
            kinds,
            defaults,
        };
        hp.tensors_agree(split)?;
        Ok(hp)
    }

    /// Layer `l`'s kind.
    #[must_use]
    pub fn kind(&self, l: usize) -> Kind {
        self.kinds[l]
    }

    /// Layer `l`'s head count: its latent layer's query heads or its delta layer's key and value
    /// heads.
    #[must_use]
    pub fn head(&self, l: usize) -> usize {
        self.heads[l]
    }

    /// Whether layer `l`'s feed-forward side is dense: one of the trunk's dense lead, or a
    /// next-token layer (raon1nano.cpp:146-154: the blocks past the trunk take the dense block).
    #[must_use]
    pub fn dense(&self, l: usize) -> bool {
        l < self.dense_lead || l >= self.n_trunk
    }

    /// The delta layers' inner width: the heads' channels.
    #[must_use]
    pub fn inner(&self) -> usize {
        self.n_head * self.kda.head_dim
    }

    /// Every tensor the file is held to is there, at the dims its keys and dialect give it, and
    /// nothing else is: the first missing or misplaced name, with how many more there are.
    fn tensors_agree(&self, split: &Split) -> Result<(), PlacementError> {
        let mut want: Vec<(String, Vec<u64>)> = vec![
            (
                names::token_embd(),
                vec![self.n_embd as u64, self.n_vocab as u64],
            ),
            (names::output_norm(), vec![self.n_embd as u64]),
        ];
        if !self.head_tied {
            want.push((
                names::output(),
                vec![self.n_embd as u64, self.n_vocab as u64],
            ));
        }
        if let Some(ple) = self.ple {
            let (e, p) = (self.n_embd as u64, ple as u64);
            want.extend([
                (
                    names::PER_LAYER_TOKEN_EMBD.to_string(),
                    vec![p * self.n_trunk as u64, self.n_vocab as u64],
                ),
                (
                    names::PER_LAYER_MODEL_PROJ.to_string(),
                    vec![e, p * self.n_trunk as u64],
                ),
                (names::PER_LAYER_PROJ_NORM.to_string(), vec![p]),
            ]);
        }
        for l in 0..self.n_layer {
            want.extend(
                self.layer_dims(l)
                    .into_iter()
                    .map(|(stem, dims)| (format!("blk.{l}.{stem}"), dims)),
            );
        }
        crate::arch::tensors_agree(split, want)?;
        // A stem of a group the block does not carry — the other dialect's, the other kind's, a
        // routed block's on a dense layer, a per-layer input's on a next-token block — is in no
        // group of the block that holds it.
        let pool = every_stem();
        let mut misplaced = Vec::new();
        for l in 0..self.n_layer {
            let owns = |stem: &str| {
                block_groups(self, l)
                    .iter()
                    .flat_map(|g| g.iter())
                    .any(|s| s.name == stem)
            };
            misplaced.extend(
                pool.iter()
                    .filter(|stem| !owns(stem))
                    .filter(|stem| split.find(&format!("blk.{l}.{stem}")).is_some())
                    .map(|stem| format!("blk.{l}.{stem}")),
            );
        }
        refuse_tensor_set(&[], &misplaced, "", "its layer's kind carries none")
    }

    /// Layer `l`'s tensors by stem and the dims llama.cpp creates them with, ggml order: its
    /// kind's mixer in the file's dialect, its feed-forward block, its hyper-connection mix, its
    /// per-layer input when it is a trunk layer of a file that carries one, and its next-token
    /// stems when it is a next-token layer.
    #[must_use]
    pub fn layer_dims(&self, l: usize) -> Vec<(&'static str, Vec<u64>)> {
        let head = self.head(l) as u64;
        let e = self.n_embd as u64;
        let mut dims: Vec<(&'static str, Vec<u64>)> = match self.kind(l) {
            Kind::Rda if self.dialect == Dialect::Fused => {
                let inner = head * self.kda.head_dim as u64;
                let gates = 2 * head;
                vec![
                    ("attn_norm.weight", vec![e]),
                    (names::ATTN_QKV, vec![e, 4 * inner]),
                    (names::SSM_CONV1D, vec![self.kda.conv as u64, 3 * inner]),
                    ("ssm_f_a.weight", vec![e, self.decay_rank as u64 + gates]),
                    ("ssm_f_b.weight", vec![self.decay_rank as u64, inner]),
                    ("ssm_dt.bias", vec![inner]),
                    ("ssm_norm.weight", vec![self.kda.head_dim as u64]),
                    ("attn_output.weight", vec![inner, e]),
                ]
            }
            Kind::Rda => {
                let inner = self.inner() as u64;
                let mut dims = kda::head_dims(
                    self.n_embd,
                    self.n_head,
                    self.kda.head_dim,
                    self.decay_rank,
                    self.kda.conv,
                )
                .to_vec();
                dims.extend([
                    ("ssm_g.weight", vec![e, inner]),
                    ("ssm_rda.weight", vec![e, self.n_head as u64]),
                ]);
                dims.extend(kda::tail_dims(self.n_embd, self.n_head, self.kda.head_dim));
                dims
            }
            Kind::Mla => self.mla.dims(e, head).to_vec(),
        };
        dims.push(("ffn_norm.weight", vec![e]));
        if self.dense(l) {
            let ff = self.dense_ff as u64;
            if self.dialect == Dialect::Fused {
                dims.extend([
                    ("ffn_up.weight", vec![e, 2 * ff]),
                    ("ffn_down.weight", vec![ff, e]),
                ]);
            } else {
                dims.extend([
                    ("ffn_gate.weight", vec![e, ff]),
                    ("ffn_up.weight", vec![e, ff]),
                    ("ffn_down.weight", vec![ff, e]),
                ]);
            }
        } else {
            let experts = self
                .experts
                .as_ref()
                .expect("a routed layer: the read refused a file whose router keys are absent");
            dims.extend(experts.dims(e));
        }
        let mix = self.hc.mix_dim() as u64;
        let streams_e = self.hc.streams as u64 * e;
        let chart = self.hc.chart_dim() as u64;
        match self.dialect {
            Dialect::Split => {
                dims.extend([
                    ("hc_attn_fn.weight", vec![streams_e, mix]),
                    ("hc_attn_base.weight", vec![mix]),
                    ("hc_attn_scale.weight", vec![HC_SCALES]),
                    ("hc_ffn_fn.weight", vec![streams_e, mix]),
                    ("hc_ffn_base.weight", vec![mix]),
                    ("hc_ffn_scale.weight", vec![HC_SCALES]),
                ]);
            }
            Dialect::Fused => {
                dims.extend([
                    ("hc_attn_fn.weight", vec![streams_e, mix]),
                    ("hc_attn_base.weight", vec![mix]),
                    ("hc_attn_chart.weight", vec![chart]),
                    ("hc_ffn_fn.weight", vec![streams_e, mix]),
                    ("hc_ffn_base.weight", vec![mix]),
                    ("hc_ffn_chart.weight", vec![chart]),
                ]);
            }
        }
        if let (true, Some(ple)) = (l < self.n_trunk, self.ple) {
            let p = ple as u64;
            dims.extend([
                (names::INP_GATE, vec![e, p]),
                (names::PROJ, vec![p, e]),
                (names::POST_NORM, vec![e]),
            ]);
        }
        if l >= self.n_trunk {
            dims.extend([
                ("nextn.eh_proj.weight", vec![2 * e, e]),
                ("nextn.enorm.weight", vec![e]),
                ("nextn.hnorm.weight", vec![e]),
                ("nextn.shared_head_norm.weight", vec![e]),
            ]);
        }
        dims
    }
}

/// A sublayer's mix scalars: `alpha_pre`, `alpha_post`, `alpha_z` and `eta`.
pub const HC_SCALES: u64 = 4;

impl Hc {
    fn read(split: &Split) -> Result<Hc, PlacementError> {
        let streams = positive(split, "hyper_connection.count")?;
        if streams < 2 {
            return Err(metadata(
                split,
                "hyper_connection.count",
                format!("is {streams}; the chart mix needs at least two streams"),
            ));
        }
        Ok(Hc {
            streams,
            eps: finite_positive(split, "hyper_connection.epsilon")?,
            magnitude: finite_positive(split, "hyper_connection.magnitude")?,
        })
    }
}

/// The cache holds the latent as the value: the generic `attention.value_length` is the latent's
/// width, and the body reads the rows by it.
fn value_length_agrees(split: &Split, mla: &Mla) -> Result<(), PlacementError> {
    let got = positive(split, "attention.value_length")?;
    if got != mla.kv_lora {
        return Err(metadata(
            split,
            "attention.value_length",
            format!(
                "is {got}, not kv_lora_rank ({}): the cached row is the latent and its unrotated \
                 values",
                mla.kv_lora
            ),
        ));
    }
    Ok(())
}

/// `attention.head_count`, one value a block: a scalar names every block's. A count of 0 is
/// refused by the key's name.
fn head_counts(split: &Split, n_layer: usize) -> Result<Vec<usize>, PlacementError> {
    let key = "attention.head_count";
    let heads = per_layer_count(split, key, n_layer)?;
    for (l, &head) in heads.iter().enumerate() {
        if head == 0 {
            return Err(metadata(split, key, format!("is 0 at layer {l}")));
        }
    }
    Ok(heads)
}

/// Every delta layer's head count is one, the first delta layer's: the recurrent state is sized
/// by it (raon1nano.cpp:38-39).
fn delta_head(split: &Split, heads: &[usize], kinds: &[Kind]) -> Result<usize, PlacementError> {
    let key = "attention.head_count";
    let first = kinds
        .iter()
        .position(|k| matches!(k, Kind::Rda))
        .expect("the kinds read holds both kinds");
    for (l, (&head, &kind)) in heads.iter().zip(kinds).enumerate() {
        if matches!(kind, Kind::Rda) && head != heads[first] {
            return Err(metadata(
                split,
                key,
                format!(
                    "is {head} at layer {l}, not {} at layer {first}: the delta layers share one \
                     head count, the recurrent state's size",
                    heads[first]
                ),
            ));
        }
    }
    Ok(heads[first])
}

/// The `situ` keys, both or neither: one without the other is refused by the absent key's name.
fn situ_of(split: &Split) -> Result<Option<Situ>, PlacementError> {
    let beta = optional_f32(split, "activation.situ_beta")?;
    let linear = optional_f32(split, "activation.situ_linear_beta")?;
    match (beta, linear) {
        (Some(_), Some(_)) => Ok(Some(Situ::read(split)?)),
        (None, None) => Ok(None),
        (Some(_), None) => Err(metadata(
            split,
            "activation.situ_linear_beta",
            "is absent, and activation.situ_beta is: the situ activation's keys come together",
        )),
        (None, Some(_)) => Err(metadata(
            split,
            "activation.situ_beta",
            "is absent, and activation.situ_linear_beta is: the situ activation's keys come together",
        )),
    }
}

/// `embedding_length_per_layer_input`: the per-layer input's width, whose layout the description
/// row states; a file that carries the key with no row for its architecture is refused naming
/// both.
fn per_layer_of(
    split: &Split,
    row: Option<&'static rules::ArchRules>,
    defaults: &mut Vec<String>,
) -> Result<Option<usize>, PlacementError> {
    let key = "embedding_length_per_layer_input";
    let Some(width) = optional_usize(split, key)? else {
        return Ok(None);
    };
    let Some(rule) = row.filter(|r| r.per_layer.is_some()) else {
        return Err(metadata(
            split,
            key,
            format!(
                "states a per-layer input, and no description row of {} states its layout",
                split.architecture().unwrap_or("<missing>")
            ),
        ));
    };
    if width == 0 {
        return Err(metadata(split, key, "is 0"));
    }
    defaults.push(rule.per_layer_line().expect("the row states its rule"));
    Ok(Some(width))
}

/// The router's keys and the dense lead: a file that carries `expert_count` routes, and its keys
/// and its `leading_dense_block_count` are required (below the trunk); one that carries none is
/// dense in every layer, the lead the trunk's own count (raon1nano.cpp:23-24). A router key
/// without the rest is refused by its name.
fn ffn_of(
    split: &Split,
    n_trunk: usize,
    defaults: &mut Vec<String>,
) -> Result<(Option<Experts>, usize), PlacementError> {
    let count = "expert_count";
    if split.value(&split.arch_key(count)).is_none() {
        let strays: Vec<&str> = noaux::KEYS
            .iter()
            .filter(|key| split.value(&split.arch_key(key)).is_some())
            .copied()
            .collect();
        if let Some(stray) = strays.first() {
            return Err(metadata(
                split,
                stray,
                format!("is in the file, and {count} is not: the router's keys come together"),
            ));
        }
        let key = "leading_dense_block_count";
        match optional_usize(split, key)? {
            Some(lead) if lead != n_trunk => {
                return Err(metadata(
                    split,
                    key,
                    format!(
                        "is {lead} for {n_trunk} layers; the file routes none, so every layer is dense"
                    ),
                ));
            }
            _ => {}
        }
        defaults.push(format!(
            "leading_dense_block_count = {n_trunk} (absent; a dense FFN in every layer, \
             src/models/raon1nano.cpp:23-24)"
        ));
        return Ok((None, n_trunk));
    }
    let experts = Experts::read(split, "the raon1 router", defaults)?;
    Ok((Some(experts), delta_mla::dense_lead_of(split, n_trunk)?))
}

/// The next-token layers the file carries: `nextn_predict_layers`, believed only while the first
/// next-token block carries `nextn.eh_proj` (llama.cpp:2302-2308). A block past the trunk that
/// carries no tensor at all is a stripped layer, refused by its key's name.
fn nextn_of(
    split: &Split,
    n_layer: usize,
    defaults: &mut Vec<String>,
) -> Result<usize, PlacementError> {
    let n = nextn_layers(split, n_layer, "", ":2302-2308", defaults)?;
    if n.past_trunk != n.carried {
        let first = n_layer - n.past_trunk;
        return Err(metadata(
            split,
            "nextn_predict_layers",
            format!(
                "is {}, and blk.{first} carries no tensor: a block past the trunk is a carried \
                 next-token layer here",
                n.past_trunk
            ),
        ));
    }
    Ok(n.carried)
}

#[cfg(test)]
pub(super) mod tests {
    use super::{Dialect, HC_SCALES, Hparams, Kind, unread_keys};
    use crate::arch::raon1::names;
    use crate::arch::synthetic::{V, header_shaped};
    use gguf::Split;
    use models::KdaGate;

    /// The widths of [`keys`]: 6 layers, delta layers at 0, 1, 3 and 4 and latent ones at 2 and 5,
    /// the first dense; 64 wide, 4 heads, delta heads of 16 with a decay of rank 8; latent
    /// ranks 24 and 32, a head of 16 + 8 query values and 16 value values; dense 48; 8 experts of
    /// 24, 2 used, 1 shared; 3 streams.
    pub(in crate::arch::raon1) const E: u64 = 64;
    pub(in crate::arch::raon1) const HEADS: u64 = 4;
    pub(in crate::arch::raon1) const D: u64 = 16;
    pub(in crate::arch::raon1) const CONV: u64 = 4;
    pub(in crate::arch::raon1) const RANK: u64 = 8;
    pub(in crate::arch::raon1) const Q_LORA: u64 = 24;
    pub(in crate::arch::raon1) const KV_LORA: u64 = 32;
    pub(in crate::arch::raon1) const NOPE: u64 = 16;
    pub(in crate::arch::raon1) const PE: u64 = 8;
    pub(in crate::arch::raon1) const HEAD_V: u64 = 16;
    pub(in crate::arch::raon1) const LAYERS: usize = 6;
    pub(in crate::arch::raon1) const MLA_LAYERS: [usize; 2] = [2, 5];
    pub(in crate::arch::raon1) const FF: u64 = 48;
    pub(in crate::arch::raon1) const N_EXP: u64 = 8;
    pub(in crate::arch::raon1) const EFF: u64 = 24;
    pub(in crate::arch::raon1) const STREAMS: u64 = 3;
    /// The per-layer input's width of the nano tables.
    pub(in crate::arch::raon1) const PLE: u64 = 16;
    /// The latent layers' head count of the nano tables: the delta layers' 4 apart.
    pub(in crate::arch::raon1) const MLA_HEADS: u64 = 8;
    /// The two-token vocabulary of the synthetic header.
    pub(in crate::arch::raon1) const VOCAB: u64 = 2;

    /// The keys of the released Mini file, at this table's widths.
    pub(in crate::arch::raon1) fn keys() -> Vec<(&'static str, V)> {
        let kv: Vec<i32> = (0..LAYERS)
            .map(|l| i32::from(MLA_LAYERS.contains(&l)))
            .collect();
        vec![
            ("block_count", V::U32(LAYERS as u32)),
            ("context_length", V::U32(4096)),
            ("embedding_length", V::U32(E as u32)),
            ("feed_forward_length", V::U32(FF as u32)),
            ("leading_dense_block_count", V::U32(1)),
            ("attention.head_count", V::U32(HEADS as u32)),
            ("attention.head_count_kv", V::I32s(kv)),
            ("attention.key_length", V::U32((KV_LORA + PE) as u32)),
            ("attention.value_length", V::U32(KV_LORA as u32)),
            ("attention.key_length_mla", V::U32((NOPE + PE) as u32)),
            ("attention.value_length_mla", V::U32(HEAD_V as u32)),
            ("attention.q_lora_rank", V::U32(Q_LORA as u32)),
            ("attention.kv_lora_rank", V::U32(KV_LORA as u32)),
            ("attention.layer_norm_rms_epsilon", V::F32(1e-5)),
            ("attention.latent_norm_rms_epsilon", V::F32(1e-6)),
            ("rope.dimension_count", V::U32(PE as u32)),
            ("ssm.conv_kernel", V::U32(CONV as u32)),
            ("ssm.time_step_rank", V::U32(RANK as u32)),
            ("kda.head_dim", V::U32(D as u32)),
            ("kda.gate_lower_bound", V::F32(-5.0)),
            ("expert_count", V::U32(N_EXP as u32)),
            ("expert_used_count", V::U32(2)),
            ("expert_feed_forward_length", V::U32(EFF as u32)),
            ("expert_shared_count", V::U32(1)),
            ("expert_gating_func", V::U32(2)),
            ("expert_weights_scale", V::F32(1.0)),
            ("expert_weights_norm", V::Bool(true)),
            ("hyper_connection.count", V::U32(STREAMS as u32)),
            ("hyper_connection.epsilon", V::F32(1e-5)),
            ("hyper_connection.magnitude", V::F32(2.0)),
            ("activation.situ_beta", V::F32(4.0)),
            ("activation.situ_linear_beta", V::F32(25.0)),
        ]
    }

    /// The released file's tensor table (731 tensors at 25 layers), at [`keys`]' widths: the three
    /// model-level tensors, then each layer's mixer, feed-forward block and mixes. Written from the
    /// table, apart from the reader.
    pub(in crate::arch::raon1) fn tensors() -> Vec<(String, Vec<u64>)> {
        let mut t: Vec<(String, Vec<u64>)> = vec![
            ("token_embd.weight".into(), vec![E, VOCAB]),
            ("output.weight".into(), vec![E, VOCAB]),
            ("output_norm.weight".into(), vec![E]),
        ];
        let inner = HEADS * D;
        let mix = 2 * STREAMS + (STREAMS - 1) * (STREAMS - 1);
        for l in 0..LAYERS {
            let mut layer: Vec<(&str, Vec<u64>)> = vec![("attn_norm.weight", vec![E])];
            if MLA_LAYERS.contains(&l) {
                layer.extend([
                    ("attn_q_a.weight", vec![E, Q_LORA]),
                    ("attn_q_a_norm.weight", vec![Q_LORA]),
                    ("attn_q_b.weight", vec![Q_LORA, HEADS * (NOPE + PE)]),
                    ("attn_kv_a_mqa.weight", vec![E, KV_LORA + PE]),
                    ("attn_kv_a_norm.weight", vec![KV_LORA]),
                    ("attn_k_b.weight", vec![NOPE, KV_LORA, HEADS]),
                    ("attn_v_b.weight", vec![KV_LORA, HEAD_V, HEADS]),
                    ("attn_gate.weight", vec![E, HEADS * HEAD_V]),
                    ("attn_output.weight", vec![HEADS * HEAD_V, E]),
                ]);
            } else {
                layer.extend([
                    ("attn_q.weight", vec![E, inner]),
                    ("attn_k.weight", vec![E, inner]),
                    ("attn_v.weight", vec![E, inner]),
                    ("ssm_conv1d_q.weight", vec![CONV, 1, inner]),
                    ("ssm_conv1d_k.weight", vec![CONV, 1, inner]),
                    ("ssm_conv1d_v.weight", vec![CONV, 1, inner]),
                    ("ssm_f_a.weight", vec![E, RANK]),
                    ("ssm_f_b.weight", vec![RANK, inner]),
                    ("ssm_beta.weight", vec![E, HEADS]),
                    ("ssm_rda.weight", vec![E, HEADS]),
                    ("ssm_a", vec![HEADS]),
                    ("ssm_dt.bias", vec![inner]),
                    ("ssm_g.weight", vec![E, inner]),
                    ("ssm_norm.weight", vec![D]),
                    ("attn_output.weight", vec![inner, E]),
                ]);
            }
            layer.push(("ffn_norm.weight", vec![E]));
            if l == 0 {
                layer.extend([
                    ("ffn_gate.weight", vec![E, FF]),
                    ("ffn_up.weight", vec![E, FF]),
                    ("ffn_down.weight", vec![FF, E]),
                ]);
            } else {
                layer.extend([
                    ("ffn_gate_inp.weight", vec![E, N_EXP]),
                    ("exp_probs_b.bias", vec![N_EXP]),
                    ("ffn_gate_exps.weight", vec![E, EFF, N_EXP]),
                    ("ffn_up_exps.weight", vec![E, EFF, N_EXP]),
                    ("ffn_down_exps.weight", vec![EFF, E, N_EXP]),
                    ("ffn_gate_shexp.weight", vec![E, EFF]),
                    ("ffn_up_shexp.weight", vec![E, EFF]),
                    ("ffn_down_shexp.weight", vec![EFF, E]),
                ]);
            }
            for sub in ["attn", "ffn"] {
                layer.push((leak(format!("hc_{sub}_fn.weight")), vec![STREAMS * E, mix]));
                layer.push((leak(format!("hc_{sub}_base.weight")), vec![mix]));
                layer.push((leak(format!("hc_{sub}_scale.weight")), vec![HC_SCALES]));
            }
            t.extend(layer.into_iter().map(|(s, d)| (format!("blk.{l}.{s}"), d)));
        }
        t
    }

    /// The keys of the released Nano file, at this table's widths: the head count an array, no
    /// `situ`, no router keys, no dense lead, a per-layer input.
    pub(in crate::arch::raon1) fn keys_nano() -> Vec<(&'static str, V)> {
        let heads: Vec<i32> = (0..LAYERS)
            .map(|l| {
                i32::try_from(if MLA_LAYERS.contains(&l) {
                    MLA_HEADS
                } else {
                    HEADS
                })
                .expect("small")
            })
            .collect();
        let kv: Vec<i32> = (0..LAYERS)
            .map(|l| i32::from(MLA_LAYERS.contains(&l)))
            .collect();
        vec![
            ("block_count", V::U32(LAYERS as u32)),
            ("context_length", V::U32(4096)),
            ("embedding_length", V::U32(E as u32)),
            ("feed_forward_length", V::U32(FF as u32)),
            ("embedding_length_per_layer_input", V::U32(PLE as u32)),
            ("attention.head_count", V::I32s(heads)),
            ("attention.head_count_kv", V::I32s(kv)),
            ("attention.key_length", V::U32((KV_LORA + PE) as u32)),
            ("attention.value_length", V::U32(KV_LORA as u32)),
            ("attention.key_length_mla", V::U32((NOPE + PE) as u32)),
            ("attention.value_length_mla", V::U32(HEAD_V as u32)),
            ("attention.q_lora_rank", V::U32(Q_LORA as u32)),
            ("attention.kv_lora_rank", V::U32(KV_LORA as u32)),
            ("attention.layer_norm_rms_epsilon", V::F32(1e-6)),
            ("attention.latent_norm_rms_epsilon", V::F32(1e-6)),
            ("rope.dimension_count", V::U32(PE as u32)),
            ("ssm.conv_kernel", V::U32(CONV as u32)),
            ("ssm.time_step_rank", V::U32(RANK as u32)),
            ("kda.head_dim", V::U32(D as u32)),
            ("kda.gate_lower_bound", V::F32(-5.0)),
            ("hyper_connection.count", V::U32(STREAMS as u32)),
            ("hyper_connection.epsilon", V::F32(1e-5)),
            ("hyper_connection.magnitude", V::F32(1.0)),
        ]
    }

    /// The Nano file's tensor table, at [`keys_nano`]' widths: no head of its own, the per-layer
    /// input's three names, and each layer's fused mixer, dense block, chart mix and per-layer
    /// input site.
    pub(in crate::arch::raon1) fn tensors_nano() -> Vec<(String, Vec<u64>)> {
        let mut t: Vec<(String, Vec<u64>)> = vec![
            ("token_embd.weight".into(), vec![E, VOCAB]),
            ("output_norm.weight".into(), vec![E]),
            (
                names::PER_LAYER_TOKEN_EMBD.into(),
                vec![PLE * LAYERS as u64, VOCAB],
            ),
            (
                names::PER_LAYER_MODEL_PROJ.into(),
                vec![E, PLE * LAYERS as u64],
            ),
            (names::PER_LAYER_PROJ_NORM.into(), vec![PLE]),
        ];
        let inner = HEADS * D;
        let mix = 2 * STREAMS + (STREAMS - 1) * (STREAMS - 1);
        let chart = mix * STREAMS * STREAMS + 2 * STREAMS * STREAMS;
        for l in 0..LAYERS {
            let mla = MLA_LAYERS.contains(&l);
            let head = if mla { MLA_HEADS } else { HEADS };
            let mut layer: Vec<(&str, Vec<u64>)> = vec![("attn_norm.weight", vec![E])];
            if mla {
                layer.extend([
                    ("attn_q_a.weight", vec![E, Q_LORA]),
                    ("attn_q_a_norm.weight", vec![Q_LORA]),
                    ("attn_q_b.weight", vec![Q_LORA, head * (NOPE + PE)]),
                    ("attn_kv_a_mqa.weight", vec![E, KV_LORA + PE]),
                    ("attn_kv_a_norm.weight", vec![KV_LORA]),
                    ("attn_k_b.weight", vec![NOPE, KV_LORA, head]),
                    ("attn_v_b.weight", vec![KV_LORA, HEAD_V, head]),
                    ("attn_gate.weight", vec![E, head * HEAD_V]),
                    ("attn_output.weight", vec![head * HEAD_V, E]),
                ]);
            } else {
                layer.extend([
                    (names::ATTN_QKV, vec![E, 4 * inner]),
                    (names::SSM_CONV1D, vec![CONV, 3 * inner]),
                    ("ssm_f_a.weight", vec![E, RANK + 2 * HEADS]),
                    ("ssm_f_b.weight", vec![RANK, inner]),
                    ("ssm_dt.bias", vec![inner]),
                    ("ssm_norm.weight", vec![D]),
                    ("attn_output.weight", vec![inner, E]),
                ]);
            }
            layer.extend([
                ("ffn_norm.weight", vec![E]),
                ("ffn_up.weight", vec![E, 2 * FF]),
                ("ffn_down.weight", vec![FF, E]),
            ]);
            for sub in ["attn", "ffn"] {
                layer.push((leak(format!("hc_{sub}_fn.weight")), vec![STREAMS * E, mix]));
                layer.push((leak(format!("hc_{sub}_base.weight")), vec![mix]));
                layer.push((leak(format!("hc_{sub}_chart.weight")), vec![chart]));
            }
            layer.extend([
                (names::INP_GATE, vec![E, PLE]),
                (names::PROJ, vec![PLE, E]),
                (names::POST_NORM, vec![E]),
            ]);
            t.extend(layer.into_iter().map(|(s, d)| (format!("blk.{l}.{s}"), d)));
        }
        t
    }

    /// One synthetic header: its keys and its tensor table.
    pub(in crate::arch::raon1) type Header = (Vec<(&'static str, V)>, Vec<(String, Vec<u64>)>);

    /// [`keys_nano`] and [`tensors_nano`] with a seventh block: a latent next-token layer with the
    /// trunk's mixer, dense block and mix, its own `nextn.*` stems and no per-layer input.
    pub(in crate::arch::raon1) fn mtp_nano() -> Header {
        let mut kv = keys_nano();
        let heads: Vec<i32> = (0..=LAYERS)
            .map(|l| {
                let h = if l >= LAYERS || MLA_LAYERS.contains(&l) {
                    MLA_HEADS
                } else {
                    HEADS
                };
                i32::try_from(h).expect("small")
            })
            .collect();
        let kvs: Vec<i32> = (0..=LAYERS)
            .map(|l| i32::from(l >= LAYERS || MLA_LAYERS.contains(&l)))
            .collect();
        kv.retain(|(k, _)| {
            !matches!(
                *k,
                "block_count" | "attention.head_count" | "attention.head_count_kv"
            )
        });
        kv.extend([
            ("block_count", V::U32(LAYERS as u32 + 1)),
            ("attention.head_count", V::I32s(heads)),
            ("attention.head_count_kv", V::I32s(kvs)),
            ("nextn_predict_layers", V::U32(1)),
        ]);
        let mut t = tensors_nano();
        let mix = 2 * STREAMS + (STREAMS - 1) * (STREAMS - 1);
        let chart = mix * STREAMS * STREAMS + 2 * STREAMS * STREAMS;
        let l = LAYERS;
        let mut block: Vec<(&str, Vec<u64>)> = vec![
            ("attn_norm.weight", vec![E]),
            ("attn_q_a.weight", vec![E, Q_LORA]),
            ("attn_q_a_norm.weight", vec![Q_LORA]),
            ("attn_q_b.weight", vec![Q_LORA, MLA_HEADS * (NOPE + PE)]),
            ("attn_kv_a_mqa.weight", vec![E, KV_LORA + PE]),
            ("attn_kv_a_norm.weight", vec![KV_LORA]),
            ("attn_k_b.weight", vec![NOPE, KV_LORA, MLA_HEADS]),
            ("attn_v_b.weight", vec![KV_LORA, HEAD_V, MLA_HEADS]),
            ("attn_gate.weight", vec![E, MLA_HEADS * HEAD_V]),
            ("attn_output.weight", vec![MLA_HEADS * HEAD_V, E]),
            ("ffn_norm.weight", vec![E]),
            ("ffn_up.weight", vec![E, 2 * FF]),
            ("ffn_down.weight", vec![FF, E]),
        ];
        for sub in ["attn", "ffn"] {
            block.push((leak(format!("hc_{sub}_fn.weight")), vec![STREAMS * E, mix]));
            block.push((leak(format!("hc_{sub}_base.weight")), vec![mix]));
            block.push((leak(format!("hc_{sub}_chart.weight")), vec![chart]));
        }
        block.extend([
            ("nextn.eh_proj.weight", vec![2 * E, E]),
            ("nextn.enorm.weight", vec![E]),
            ("nextn.hnorm.weight", vec![E]),
            ("nextn.shared_head_norm.weight", vec![E]),
        ]);
        t.extend(block.into_iter().map(|(s, d)| (format!("blk.{l}.{s}"), d)));
        (kv, t)
    }

    /// A name that lives as long as the process: a test table's computed stem.
    fn leak(s: String) -> &'static str {
        Box::leak(s.into_boxed_str())
    }

    /// [`keys`] with `key` replaced by `v`, or dropped when `v` is `None`.
    pub(in crate::arch::raon1) fn with(key: &'static str, v: Option<V>) -> Vec<(&'static str, V)> {
        swapped(keys(), key, v)
    }

    /// `kv` with `key` replaced by `v`, or dropped when `v` is `None`.
    pub(in crate::arch::raon1) fn swapped(
        mut kv: Vec<(&'static str, V)>,
        key: &'static str,
        v: Option<V>,
    ) -> Vec<(&'static str, V)> {
        kv.retain(|(k, _)| *k != key);
        if let Some(v) = v {
            kv.push((key, v));
        }
        kv
    }

    /// The reader's verdict on a header of `kv` and `tensors`, as the error text.
    fn verdict(
        tag: &str,
        arch: &str,
        kv: &[(&str, V)],
        tensors: &[(String, Vec<u64>)],
    ) -> Result<Hparams, String> {
        let path = header_shaped(&format!("raon1-hparams-{tag}"), arch, kv, &[], tensors);
        let split = Split::open(&path).map_err(|e| e.to_string());
        let got = split.and_then(|s| Hparams::read(&s).map_err(|e| e.to_string()));
        let _ = std::fs::remove_file(&path);
        got
    }

    /// The small Mini header's read: panics with the error text when it does not read.
    fn read(tag: &str, kv: &[(&'static str, V)], tensors: &[(String, Vec<u64>)]) -> Hparams {
        verdict(tag, "raon1", kv, tensors).unwrap_or_else(|e| panic!("{tag} does not read: {e}"))
    }

    /// The small Nano header's read and its tensor table: panics with the error text when it does
    /// not read.
    pub(in crate::arch::raon1) fn read_nano(tag: &str) -> (Hparams, Vec<(String, Vec<u64>)>) {
        let hp = verdict(tag, "raon1nano", &keys_nano(), &tensors_nano())
            .unwrap_or_else(|e| panic!("{tag} does not read: {e}"));
        (hp, tensors_nano())
    }

    /// The small Nano MTP header's read and its tensor table.
    pub(in crate::arch::raon1) fn read_nano_mtp(tag: &str) -> (Hparams, Vec<(String, Vec<u64>)>) {
        let (kv, t) = mtp_nano();
        let hp = verdict(tag, "raon1nano", &kv, &t)
            .unwrap_or_else(|e| panic!("{tag} does not read: {e}"));
        (hp, t)
    }

    /// One refusal case: a tag, the header's keys, the tensor shapes and the name the refusal must hold.
    type Case = (
        String,
        Vec<(&'static str, V)>,
        Vec<(String, Vec<u64>)>,
        &'static str,
    );

    /// What is wrong with the verdict on `tag`'s header: it read, or the error lacks `want`.
    fn wrong_verdict(
        tag: &str,
        arch: &str,
        kv: &[(&str, V)],
        tensors: &[(String, Vec<u64>)],
        want: &str,
    ) -> Option<String> {
        match verdict(tag, arch, kv, tensors) {
            Ok(_) => Some(format!("{tag}: read; expected {want:?}")),
            Err(e) if e.contains(want) => None,
            Err(e) => Some(format!("{tag}: want {want:?} in: {e}")),
        }
    }

    /// The tensors with `name` dropped.
    fn without(name: &str) -> Vec<(String, Vec<u64>)> {
        tensors().into_iter().filter(|(n, _)| n != name).collect()
    }

    /// The tensors with `name` given `dims`, or added when the table has none.
    fn shaped(name: &str, dims: Vec<u64>) -> Vec<(String, Vec<u64>)> {
        let mut t = without(name);
        t.push((name.to_string(), dims));
        t
    }

    /// The header the table gives reads, to the values its keys hold, with the head counts
    /// splitting the layers and no default taken.
    #[test]
    fn a_small_header_reads() {
        let hp = read("small", &keys(), &tensors());
        assert_eq!(
            (hp.arch, hp.n_layer, hp.n_trunk, hp.n_embd),
            (models::Arch::Raon1, LAYERS, LAYERS, 64)
        );
        assert_eq!(hp.dialect, Dialect::Split);
        assert_eq!(hp.n_head, 4);
        assert_eq!(hp.heads, vec![4; LAYERS]);
        assert_eq!((hp.kda.conv, hp.kda.head_dim, hp.decay_rank), (4, 16, 8));
        assert_eq!(hp.gate, KdaGate::Sigmoid { lower_bound: -5.0 });
        assert_eq!(
            (
                hp.mla.q_lora,
                hp.mla.kv_lora,
                hp.mla.head_k,
                hp.mla.head_v,
                hp.mla.pe
            ),
            (24, 32, 24, 16, 8)
        );
        assert_eq!((hp.mla.nope(), hp.mla.row()), (16, 40));
        assert_eq!((hp.rms_eps, hp.latent_eps), (1e-5, 1e-6));
        assert_eq!((hp.hc.streams, hp.hc.eps, hp.hc.magnitude), (3, 1e-5, 2.0));
        assert_eq!(hp.hc.mix_dim(), 10);
        assert_eq!(hp.hc.chart_dim(), 108);
        assert_eq!(
            (
                hp.situ.expect("the situ keys").gate_beta,
                hp.situ.expect("the situ keys").value_beta
            ),
            (4.0, 25.0)
        );
        assert_eq!(
            hp.act,
            models::Act::Situ {
                gate_beta: 4.0,
                value_beta: 25.0
            }
        );
        assert_eq!(hp.ple, None);
        assert!(!hp.head_tied);
        assert_eq!(
            (hp.n_vocab, hp.n_ctx_train, hp.dense_lead, hp.dense_ff),
            (2, 4096, 1, 48)
        );
        let experts = hp.experts.expect("the router keys");
        assert_eq!(
            (
                experts.n_expert,
                experts.n_used,
                experts.ff,
                experts.shared_ff()
            ),
            (8, 2, 24, 24)
        );
        assert_eq!((experts.weights_scale, experts.weights_norm), (1.0, true));
        assert_eq!(hp.inner(), 64);
        let kinds: Vec<Kind> = (0..LAYERS).map(|l| hp.kind(l)).collect();
        assert_eq!(
            kinds,
            [
                Kind::Rda,
                Kind::Rda,
                Kind::Mla,
                Kind::Rda,
                Kind::Rda,
                Kind::Mla
            ]
        );
        assert!(hp.dense(0) && !hp.dense(1));
        assert!(hp.defaults.is_empty(), "{:?}", hp.defaults);
    }

    /// The small Nano header reads: the fused dialect, the head count an array with the latent
    /// layers' own, every layer a dense SwiGLU block, the per-layer input, the tied head, and the
    /// row's and the loader's defaults recorded. Mutant: the unrouted lead read as 0, the
    /// per-layer width dropped, or a default line left out.
    #[test]
    fn a_small_nano_header_reads() {
        let (hp, _) = read_nano("nano");
        assert_eq!(hp.arch, models::Arch::Raon1Nano);
        assert_eq!((hp.n_layer, hp.n_trunk), (LAYERS, LAYERS));
        assert_eq!(hp.dialect, Dialect::Fused);
        assert_eq!(hp.heads, vec![4, 4, 8, 4, 4, 8]);
        assert_eq!((hp.head(0), hp.head(2), hp.n_head), (4, 8, 4));
        assert_eq!(hp.hc.chart_dim(), 108);
        assert_eq!(hp.situ, None);
        assert_eq!(hp.act, models::Act::SwiGlu { limit: None });
        assert_eq!(hp.ple, Some(PLE as usize));
        assert!(hp.head_tied);
        assert_eq!(hp.experts, None);
        assert_eq!(hp.dense_lead, LAYERS);
        assert!(hp.dense(0) && hp.dense(5));
        assert_eq!(hp.inner(), 64);
        let defaults = hp.defaults.join("\n");
        for needle in [
            "leading_dense_block_count = 6 (absent; a dense FFN in every layer",
            "ffn act = SwiGLU (",
            "per-layer input = token table, layout 2 (",
            "head = tied (",
        ] {
            assert!(defaults.contains(needle), "{needle:?} not in:\n{defaults}");
        }
    }

    /// The small Nano MTP header reads: the seventh block a latent next-token layer, the trunk's
    /// count the per-layer input's tables are sized by.
    #[test]
    fn a_small_nano_mtp_header_reads() {
        let (hp, t) = read_nano_mtp("nano-mtp");
        assert_eq!((hp.n_layer, hp.n_trunk), (LAYERS + 1, LAYERS));
        assert_eq!(hp.kind(LAYERS), Kind::Mla);
        assert_eq!(hp.head(LAYERS), MLA_HEADS as usize);
        assert!(
            hp.dense(LAYERS),
            "the next-token block takes the dense block"
        );
        assert!(t.iter().any(|(n, _)| n == "blk.6.nextn.eh_proj.weight"));
        assert_eq!(
            t.iter()
                .find(|(n, _)| n == names::PER_LAYER_TOKEN_EMBD)
                .map(|(_, d)| d.clone()),
            Some(vec![PLE * LAYERS as u64, VOCAB])
        );
    }

    /// A file whose `nextn_predict_layers` names a block the file carries no next-token tensors
    /// for disbelieves the key: the block reads as a trunk layer and the default records it
    /// (llama.cpp:2302-2308). A block past the trunk that carries no tensor at all is a stripped
    /// layer, refused by the key's name. Mutant: the key taken at its word.
    #[test]
    fn a_nextn_key_without_tensors_is_disbelieved() {
        let hp = read(
            "nextn-believed",
            &with("nextn_predict_layers", Some(V::U32(1))),
            &tensors(),
        );
        assert_eq!((hp.n_layer, hp.n_trunk), (LAYERS, LAYERS));
        assert!(
            hp.defaults.iter().any(|d| d.starts_with(
                "nextn_predict_layers = 0 (blk.5.nextn.eh_proj.weight is not in the file"
            )),
            "{:?}",
            hp.defaults
        );
    }

    /// An absent `kda.gate_lower_bound` is the softplus decay, recorded as a default; a bound
    /// that is not below 0 is refused by its key.
    #[test]
    fn the_decay_gate_follows_its_bound() {
        let hp = read("no-bound", &with("kda.gate_lower_bound", None), &tensors());
        assert_eq!(hp.gate, KdaGate::Softplus);
        assert!(
            hp.defaults
                .iter()
                .any(|d| d.contains("kda.gate_lower_bound")),
            "{:?}",
            hp.defaults
        );
        for (tag, bound) in [("zero", 0.0_f32), ("up", 5.0), ("nan", f32::NAN)] {
            let kv = with("kda.gate_lower_bound", Some(V::F32(bound)));
            assert_eq!(
                wrong_verdict(tag, "raon1", &kv, &tensors(), "kda.gate_lower_bound"),
                None
            );
        }
    }

    /// A Nano file that carries the gate key above 0 is refused by it, as a Mini file is.
    #[test]
    fn the_nano_decay_gate_follows_its_bound() {
        let kv = swapped(keys_nano(), "kda.gate_lower_bound", Some(V::F32(0.0)));
        assert_eq!(
            wrong_verdict(
                "nano-bound",
                "raon1nano",
                &kv,
                &tensors_nano(),
                "kda.gate_lower_bound"
            ),
            None
        );
    }

    /// Every refusal names its key or tensor, both dialects'. Mutants: a check dropped, or the
    /// wrong name given.
    #[test]
    fn the_reader_refuses_by_name() {
        let cases: Vec<Case> = vec![
            (
                "no-rank".into(),
                with("ssm.time_step_rank", None),
                tensors(),
                "ssm.time_step_rank",
            ),
            (
                "no-hc".into(),
                with("hyper_connection.count", None),
                tensors(),
                "hyper_connection.count",
            ),
            (
                "one-stream".into(),
                with("hyper_connection.count", Some(V::U32(1))),
                tensors(),
                "at least two streams",
            ),
            (
                "no-magnitude".into(),
                with("hyper_connection.magnitude", None),
                tensors(),
                "hyper_connection.magnitude",
            ),
            (
                "no-situ".into(),
                with("activation.situ_beta", None),
                tensors(),
                "activation.situ_beta",
            ),
            (
                "situ-gone-no-row".into(),
                swapped(
                    with("activation.situ_beta", None),
                    "activation.situ_linear_beta",
                    None,
                ),
                tensors(),
                "no description row of raon1",
            ),
            (
                "zero-situ".into(),
                with("activation.situ_linear_beta", Some(V::F32(0.0))),
                tensors(),
                "activation.situ_linear_beta",
            ),
            (
                "no-latent-eps".into(),
                with("attention.latent_norm_rms_epsilon", None),
                tensors(),
                "attention.latent_norm_rms_epsilon",
            ),
            (
                "no-pe".into(),
                with("rope.dimension_count", None),
                tensors(),
                "rope.dimension_count",
            ),
            (
                "pe-fills-head".into(),
                with("rope.dimension_count", Some(V::U32(24))),
                tensors(),
                "attention.key_length_mla",
            ),
            (
                "row-width".into(),
                with("attention.key_length", Some(V::U32(576))),
                tensors(),
                "attention.key_length",
            ),
            (
                "value-width".into(),
                with("attention.value_length", Some(V::U32(16))),
                tensors(),
                "attention.value_length",
            ),
            (
                "softmax".into(),
                with("expert_gating_func", Some(V::U32(1))),
                tensors(),
                "expert_gating_func",
            ),
            (
                "lead-all".into(),
                with("leading_dense_block_count", Some(V::U32(6))),
                tensors(),
                "leading_dense_block_count",
            ),
            (
                "no-lead".into(),
                with("leading_dense_block_count", None),
                tensors(),
                "leading_dense_block_count",
            ),
            (
                "no-head".into(),
                keys(),
                without("output.weight"),
                "output.weight",
            ),
            (
                "nextn-stripped".into(),
                with("nextn_predict_layers", Some(V::U32(1))),
                tensors()
                    .into_iter()
                    .filter(|(n, _)| !n.starts_with("blk.5."))
                    .collect(),
                "nextn_predict_layers",
            ),
            (
                "kv-two".into(),
                with(
                    "attention.head_count_kv",
                    Some(V::I32s(vec![0, 0, 2, 0, 0, 1])),
                ),
                tensors(),
                "2 at layer 2",
            ),
            (
                "all-delta".into(),
                with("attention.head_count_kv", Some(V::I32s(vec![0; 6]))),
                tensors(),
                "6 delta layers of 6",
            ),
            (
                "all-latent".into(),
                with("attention.head_count_kv", Some(V::I32s(vec![1; 6]))),
                tensors(),
                "0 delta layers of 6",
            ),
            (
                "kv-short".into(),
                with("attention.head_count_kv", Some(V::I32s(vec![0, 1]))),
                tensors(),
                "attention.head_count_kv",
            ),
            (
                "head-zero".into(),
                with("attention.head_count", Some(V::U32(0))),
                tensors(),
                "attention.head_count",
            ),
            (
                "delta-heads-apart".into(),
                with(
                    "attention.head_count",
                    Some(V::I32s(vec![4, 4, 8, 4, 8, 8])),
                ),
                tensors(),
                "attention.head_count",
            ),
            (
                "no-gate".into(),
                keys(),
                without("blk.2.attn_gate.weight"),
                "blk.2.attn_gate.weight",
            ),
            (
                "no-write-gate".into(),
                keys(),
                without("blk.1.ssm_rda.weight"),
                "blk.1.ssm_rda.weight",
            ),
            (
                "no-hc-fn".into(),
                keys(),
                without("blk.4.hc_ffn_fn.weight"),
                "blk.4.hc_ffn_fn.weight",
            ),
            (
                "no-embd".into(),
                keys(),
                without("token_embd.weight"),
                "token_embd.weight",
            ),
            (
                "latent-on-delta".into(),
                keys(),
                shaped("blk.1.attn_gate.weight", vec![E, HEADS * HEAD_V]),
                "blk.1.attn_gate.weight",
            ),
            (
                "delta-on-latent".into(),
                keys(),
                shaped("blk.2.ssm_rda.weight", vec![E, HEADS]),
                "blk.2.ssm_rda.weight",
            ),
            (
                "dense-with-router".into(),
                keys(),
                shaped("blk.0.ffn_gate_inp.weight", vec![E, N_EXP]),
                "blk.0.ffn_gate_inp.weight",
            ),
            (
                "routed-with-dense".into(),
                keys(),
                shaped("blk.3.ffn_gate.weight", vec![E, FF]),
                "blk.3.ffn_gate.weight",
            ),
            (
                "decay-rank".into(),
                keys(),
                shaped("blk.1.ssm_f_b.weight", vec![RANK + 1, HEADS * D]),
                "blk.1.ssm_f_b.weight",
            ),
            (
                "latent-row".into(),
                keys(),
                shaped("blk.5.attn_kv_a_mqa.weight", vec![E, KV_LORA]),
                "blk.5.attn_kv_a_mqa.weight",
            ),
            (
                "hc-mix".into(),
                keys(),
                shaped("blk.3.hc_attn_fn.weight", vec![STREAMS * E, 9]),
                "blk.3.hc_attn_fn.weight",
            ),
            (
                "hc-scale".into(),
                keys(),
                shaped("blk.3.hc_ffn_scale.weight", vec![3]),
                "blk.3.hc_ffn_scale.weight",
            ),
            (
                "k-b".into(),
                keys(),
                shaped("blk.5.attn_k_b.weight", vec![NOPE, KV_LORA, 1]),
                "blk.5.attn_k_b.weight",
            ),
            (
                "mixed-dialect".into(),
                keys(),
                shaped("blk.0.attn_qkv.weight", vec![E, 4 * HEADS * D]),
                "never both",
            ),
        ];
        let wrong: Vec<String> = cases
            .iter()
            .filter_map(|(tag, kv, t, want)| wrong_verdict(tag, "raon1", kv, t, want))
            .collect();
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    /// The Nano dialect's own refusals: a per-layer input without the key that states its width,
    /// a row-less file that carries the key, a stray router key, a dense lead the unrouted file
    /// does not carry, and a fused tensor of the wrong shape. Mutant: the per-layer stems left
    /// out of the misplaced walk's pool, or the stray-key check dropped.
    #[test]
    fn the_nano_reader_refuses_by_name() {
        let cases: Vec<Case> = vec![
            (
                "ple-no-key".into(),
                swapped(keys_nano(), "embedding_length_per_layer_input", None),
                tensors_nano(),
                "blk.0.inp_gate.weight",
            ),
            (
                "ple-zero".into(),
                swapped(
                    keys_nano(),
                    "embedding_length_per_layer_input",
                    Some(V::U32(0)),
                ),
                tensors_nano(),
                "embedding_length_per_layer_input",
            ),
            (
                "situ-one".into(),
                swapped(keys_nano(), "activation.situ_beta", Some(V::F32(4.0))),
                tensors_nano(),
                "activation.situ_linear_beta",
            ),
            (
                "expert-stray".into(),
                swapped(keys_nano(), "expert_used_count", Some(V::U32(2))),
                tensors_nano(),
                "expert_used_count",
            ),
            (
                "lead-stray".into(),
                swapped(keys_nano(), "leading_dense_block_count", Some(V::U32(4))),
                tensors_nano(),
                "leading_dense_block_count",
            ),
            (
                "chart-dims".into(),
                keys_nano(),
                shaped_nano("blk.3.hc_attn_chart.weight", vec![107]),
                "blk.3.hc_attn_chart.weight",
            ),
            (
                "fused-qkv-dims".into(),
                keys_nano(),
                shaped_nano("blk.0.attn_qkv.weight", vec![E, 3 * HEADS * D]),
                "blk.0.attn_qkv.weight",
            ),
            (
                "split-stem".into(),
                keys_nano(),
                shaped_nano("blk.3.attn_q.weight", vec![E, HEADS * D]),
                "blk.3.attn_q.weight",
            ),
        ];
        let wrong: Vec<String> = cases
            .iter()
            .filter_map(|(tag, kv, t, want)| wrong_verdict(tag, "raon1nano", kv, t, want))
            .collect();
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
        // A mini file that carries no situ keys and has no row takes none from the nano row, and
        // a mini file that carries the per-layer input's key states one no row of raon1 lays out.
        let kv = swapped(
            swapped(keys(), "activation.situ_beta", None),
            "activation.situ_linear_beta",
            None,
        );
        assert_eq!(
            wrong_verdict(
                "mini-situ-gone",
                "raon1",
                &kv,
                &tensors(),
                "no description row of raon1"
            ),
            None
        );
        let kv = with("embedding_length_per_layer_input", Some(V::U32(PLE as u32)));
        assert_eq!(
            wrong_verdict(
                "mini-ple-key",
                "raon1",
                &kv,
                &tensors(),
                "no description row of raon1"
            ),
            None
        );
    }

    /// A Nano MTP block that is a delta layer is refused: the next-token walk is written for a
    /// latent one (raon1nano.cpp:461).
    #[test]
    fn a_delta_next_token_block_is_refused() {
        let (mut kv, mut t) = mtp_nano();
        let heads: Vec<i32> = (0..=LAYERS)
            .map(|l| {
                let h = if l >= LAYERS || MLA_LAYERS.contains(&l) {
                    MLA_HEADS
                } else {
                    HEADS
                };
                i32::try_from(h).expect("small")
            })
            .collect();
        let kvs: Vec<i32> = vec![0, 0, 1, 0, 0, 1, 0];
        kv.retain(|(k, _)| !matches!(*k, "attention.head_count" | "attention.head_count_kv"));
        kv.extend([
            ("attention.head_count", V::I32s(heads)),
            ("attention.head_count_kv", V::I32s(kvs)),
        ]);
        // The seventh block as a delta layer of the fused dialect.
        t.retain(|(n, _)| !n.starts_with("blk.6."));
        let inner = HEADS * D;
        let mix = 2 * STREAMS + (STREAMS - 1) * (STREAMS - 1);
        let chart = mix * STREAMS * STREAMS + 2 * STREAMS * STREAMS;
        let mut block: Vec<(&str, Vec<u64>)> = vec![
            ("attn_norm.weight", vec![E]),
            (names::ATTN_QKV, vec![E, 4 * inner]),
            (names::SSM_CONV1D, vec![CONV, 3 * inner]),
            ("ssm_f_a.weight", vec![E, RANK + 2 * HEADS]),
            ("ssm_f_b.weight", vec![RANK, inner]),
            ("ssm_dt.bias", vec![inner]),
            ("ssm_norm.weight", vec![D]),
            ("attn_output.weight", vec![inner, E]),
            ("ffn_norm.weight", vec![E]),
            ("ffn_up.weight", vec![E, 2 * FF]),
            ("ffn_down.weight", vec![FF, E]),
        ];
        for sub in ["attn", "ffn"] {
            block.push((leak(format!("hc_{sub}_fn.weight")), vec![STREAMS * E, mix]));
            block.push((leak(format!("hc_{sub}_base.weight")), vec![mix]));
            block.push((leak(format!("hc_{sub}_chart.weight")), vec![chart]));
        }
        block.extend([
            ("nextn.eh_proj.weight", vec![2 * E, E]),
            ("nextn.enorm.weight", vec![E]),
            ("nextn.hnorm.weight", vec![E]),
            ("nextn.shared_head_norm.weight", vec![E]),
        ]);
        t.extend(
            block
                .into_iter()
                .map(|(s, d)| (format!("blk.{l}.{s}", l = LAYERS), d)),
        );
        assert_eq!(
            wrong_verdict("mtp-delta", "raon1nano", &kv, &t, "a latent layer"),
            None
        );
    }

    /// The reader lists, and does not refuse, a key it does not read.
    #[test]
    fn an_unread_key_is_listed() {
        let mut kv = keys();
        kv.push(("rope.freq_base", V::F32(10_000.0)));
        let path = header_shaped("raon1-hparams-unread", "raon1", &kv, &[], &tensors());
        let split = Split::open(&path).expect("opens");
        let unread = unread_keys(&split);
        let _ = std::fs::remove_file(&path);
        assert_eq!(unread, vec!["raon1.rope.freq_base".to_string()]);
    }

    /// The tensors of [`tensors_nano`] with `name` given `dims`.
    fn shaped_nano(name: &str, dims: Vec<u64>) -> Vec<(String, Vec<u64>)> {
        let mut t: Vec<(String, Vec<u64>)> = tensors_nano()
            .into_iter()
            .filter(|(n, _)| n != name)
            .collect();
        t.push((name.to_string(), dims));
        t
    }
}
