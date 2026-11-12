//! The description rows: the facts of an architecture no GGUF key states, one
//! [`ArchRules`] row per architecture string, each fact with the line that
//! states it. A reader that serves several architectures takes every such fact
//! from its file's row, never from another family's: a fact neither the file
//! nor the row states is refused naming both, and the row's values a read
//! consumes go into its `defaults`, each with its source, so a description
//! print carries its provenance.
//!
//! A row is data, not code: adding an architecture is a row here and a registry
//! row, and a fact a new file states as a key leaves the row.

use models::{Act, Arch, ReasoningFormat, RopeMode, RouterNorm, ToolFormat};

/// The order a SwiGLU clamp clamps in: which side of the activation a limit
/// reaches first. The two orders differ only where `g > L`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClampOrder {
    /// `silu(min(g, L))·clamp(u, ±L)`: the gate is clamped from above before the
    /// activation (ggml's `ggml_swiglu_clamp`, the arch list llama-graph.cpp runs
    /// it for).
    GateFirst,
    /// `min(silu(g), L)·clamp(u, ±L)`: the activation's result is clamped from
    /// above (llama-graph.cpp's generic clamped-SwiGLU branch).
    ActFirst,
    /// The architecture clamps nothing; a non-zero clamp limit is refused by name.
    None,
}

impl ClampOrder {
    /// The order as a `defaults` line names it.
    #[must_use]
    pub const fn what(self) -> &'static str {
        match self {
            ClampOrder::GateFirst => "gate-first",
            ClampOrder::ActFirst => "act-first",
            ClampOrder::None => "none",
        }
    }
}

/// What a layer's rope on/off is when the file carries no
/// `attention.rope_pattern`: the loader's own rule, not a key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeFallback {
    /// A layer with a sliding window is roped, a windowless one is not.
    WhereWindowed,
}

impl RopeFallback {
    /// The rule as a `defaults` line names it.
    #[must_use]
    pub const fn what(self) -> &'static str {
        match self {
            RopeFallback::WhereWindowed => "where windowed",
        }
    }
}

/// Where a file's chat parsers come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatParsers {
    /// The parsers every file of the architecture shares, whatever its template
    /// says: the family's own convention.
    Fixed {
        tools: Option<ToolFormat>,
        reasoning: Option<ReasoningFormat>,
    },
    /// The file's own template, through [`ToolFormat::of_template`] and
    /// [`ReasoningFormat::of_template`]: no parser where its markup selects none.
    FromTemplate,
}

/// What the head reads when the file carries no `output.weight`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadWeight {
    /// Its own matrix; a file without it is refused by the missing tensor's name.
    Own,
    /// The token embedding's matrix (tied).
    Tied,
}

/// How a per-layer input of `embedding_length_per_layer_input` values is made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PerLayerRule {
    /// Layout 2, the converter's: `p = (RMSNorm(W_mp·e) + √width·table[t]) / √2`, the table one
    /// row of `width` values a token a layer, gated into every stream after the layer's
    /// feed-forward mix.
    TokenTable,
}

/// The source line of each fact of a row, as its `defaults` line prints it.
#[derive(Clone, Copy, Debug)]
pub struct RuleSources {
    pub pairs: &'static str,
    pub clamp: &'static str,
    pub router_norm: &'static str,
    pub rope_fallback: &'static str,
    pub chat: &'static str,
    /// The facts only a row that states them reads: `-` where the row's fact is
    /// that the architecture has none.
    pub ffn_act: &'static str,
    pub head: &'static str,
    pub per_layer: &'static str,
}

/// One architecture's facts no GGUF key states, each with its source. The
/// values are what a reader serves the architecture by; [`row`] finds a row by
/// the file's own architecture string.
pub struct ArchRules {
    /// The architecture the row describes, named by its own string.
    pub arch: Arch,
    /// How the rope pairs a roped head's values.
    pub pairs: RopeMode,
    /// The clamp order of the SwiGLU limits the file carries.
    pub clamp: ClampOrder,
    /// The divisor policy the router's kept weights take when the file's
    /// `expert_weights_norm` is set.
    pub router_norm: RouterNorm,
    /// The rope on/off of a layer `attention.rope_pattern` does not name;
    /// `None`: the key is the only source, and its absence is refused.
    pub rope_fallback: Option<RopeFallback>,
    pub chat: ChatParsers,
    /// The activation the feed-forward blocks take when the file's keys name
    /// none; `None`: the file's keys state it.
    pub ffn_act: Option<Act>,
    /// What the head reads when the file carries no `output.weight`.
    pub head: HeadWeight,
    /// How the per-layer input is made when the file carries
    /// `embedding_length_per_layer_input`; `None`: no file of the architecture
    /// carries one.
    pub per_layer: Option<PerLayerRule>,
    pub sources: RuleSources,
}

impl ArchRules {
    /// The activation a SwiGLU limit takes under this row's clamp order: a
    /// limit of at most 1e-6 clamps nothing whatever the order
    /// (llama-graph.cpp:2064 checks `limit > eps`), a larger one the order's
    /// act. The reader refuses a non-zero limit under [`ClampOrder::None`]
    /// before this.
    #[must_use]
    pub const fn act(&self, limit: f32) -> models::Act {
        if limit <= EPS {
            models::Act::SwiGlu { limit: None }
        } else {
            match self.clamp {
                ClampOrder::GateFirst => models::Act::SwiGluGateFirst { limit },
                ClampOrder::ActFirst | ClampOrder::None => {
                    models::Act::SwiGlu { limit: Some(limit) }
                }
            }
        }
    }

    /// `rope pairs = <value> (<source>)`: the row's pair layout, for a read
    /// that roped a layer by it.
    #[must_use]
    pub fn rope_pairs_line(&self) -> String {
        format!(
            "rope pairs = {} ({})",
            pairs_name(self.pairs),
            self.sources.pairs
        )
    }

    /// `rope = <value> (<source>)`: the row's fallback, for a read that took
    /// its layers' rope on/off from it.
    #[must_use]
    pub fn rope_fallback_line(&self) -> String {
        format!(
            "rope = {} ({})",
            self.rope_fallback.map_or("none", RopeFallback::what),
            self.sources.rope_fallback
        )
    }

    /// `router norm = <value> (<source>)`: the row's divisor policy, for a read
    /// whose file set `expert_weights_norm`.
    #[must_use]
    pub fn router_norm_line(&self) -> String {
        format!(
            "router norm = {} ({})",
            norm_name(self.router_norm),
            self.sources.router_norm
        )
    }

    /// `swiglu clamp = <value> (<source>)`: the row's clamp order, for a read
    /// that turned a non-zero limit into an activation by it.
    #[must_use]
    pub fn clamp_line(&self) -> String {
        format!(
            "swiglu clamp = {} ({})",
            self.clamp.what(),
            self.sources.clamp
        )
    }

    /// `chat parsers = <value> (<source>)`: the row's fixed parsers, for a read
    /// that took them from it. A row that reads the file's template states
    /// nothing the file does not, and records no line.
    #[must_use]
    pub fn chat_line(&self) -> Option<String> {
        let ChatParsers::Fixed { tools, reasoning } = self.chat else {
            return None;
        };
        Some(format!(
            "chat parsers = {} tools, {} reasoning ({})",
            tools.map_or("no", tools_name),
            reasoning.map_or("no", reasoning_name),
            self.sources.chat
        ))
    }

    /// `ffn act = <what> (<source>)`: the row's activation, for a read whose
    /// file named none. `None`: the row states no activation.
    #[must_use]
    pub fn ffn_act_line(&self) -> Option<String> {
        self.ffn_act
            .map(|act| format!("ffn act = {} ({})", act.what(), self.sources.ffn_act))
    }

    /// `head = tied (<source>)`: the head reads the token embedding, for a read
    /// whose file carries no `output.weight`. `None`: the row states no tie.
    #[must_use]
    pub fn tied_head_line(&self) -> Option<String> {
        (self.head == HeadWeight::Tied).then(|| format!("head = tied ({})", self.sources.head))
    }

    /// `per-layer input = <what> (<source>)`: the row's per-layer-input rule,
    /// for a read whose file carries the key. `None`: the row states none.
    #[must_use]
    pub fn per_layer_line(&self) -> Option<String> {
        self.per_layer.map(|rule| {
            format!(
                "per-layer input = {} ({})",
                per_layer_name(rule),
                self.sources.per_layer
            )
        })
    }
}

/// The per-layer-input rule as a `defaults` line names it.
const fn per_layer_name(rule: PerLayerRule) -> &'static str {
    match rule {
        PerLayerRule::TokenTable => "token table, layout 2",
    }
}

/// A limit of at most this clamps nothing (llama-graph.cpp:2064's `eps`).
const EPS: f32 = 1e-6;

/// The rope pair layout as a `defaults` line names it.
const fn pairs_name(mode: RopeMode) -> &'static str {
    match mode {
        RopeMode::NormTail => "norm-tail",
        RopeMode::Neox => "neox",
        RopeMode::Imrope { .. } => "imrope",
        RopeMode::None => "none",
    }
}

/// The divisor policy as a `defaults` line names it.
const fn norm_name(norm: RouterNorm) -> &'static str {
    match norm {
        RouterNorm::None => "none",
        RouterNorm::SelectedSum => "selected-sum",
        RouterNorm::SelectedSumFloor => "selected-sum-floor",
    }
}

/// The tool-call syntax as a `defaults` line names it.
const fn tools_name(format: ToolFormat) -> &'static str {
    match format {
        ToolFormat::Dsml => "dsml",
        ToolFormat::GlmXml => "glm-xml",
        ToolFormat::QwenXml => "qwen-xml",
        ToolFormat::TagKv => "tag-kv",
        ToolFormat::MuseSpark => "muse-spark",
        ToolFormat::Mistral => "mistral",
    }
}

/// The reasoning span syntax as a `defaults` line names it.
const fn reasoning_name(format: ReasoningFormat) -> &'static str {
    match format {
        ReasoningFormat::ThinkSpan { .. } => "think-span",
        ReasoningFormat::ThinkThenResponse { .. } => "think-then-response",
    }
}

static ROWS: &[ArchRules] = &[
    ArchRules {
        arch: Arch::Flint,
        // Flint's rope type is NEOX (llama-model.cpp's table, the LLM_ARCH_FLINT case).
        pairs: RopeMode::Neox,
        // The flint loader reads no swiglu clamp key (flint.cpp:3-19), so a limit
        // the file does carry is not this architecture's rule.
        clamp: ClampOrder::None,
        // Flint is not in llama-graph.cpp's epsilon-added divisor list, so its kept
        // weights divide by the selected sum clamped at F16's least value.
        router_norm: RouterNorm::SelectedSumFloor,
        // The loader reads `attention.rope_pattern` as a required array (flint.cpp:16).
        rope_fallback: None,
        chat: ChatParsers::Fixed {
            tools: Some(ToolFormat::GlmXml),
            reasoning: Some(ReasoningFormat::THINK),
        },
        ffn_act: None,
        head: HeadWeight::Own,
        per_layer: None,
        sources: RuleSources {
            pairs: "llama.cpp src/llama-model.cpp:3769,3834: Flint's rope type is NEOX",
            clamp: "llama.cpp src/models/flint.cpp:3-19: the loader reads no clamp key",
            router_norm: "llama.cpp src/llama-graph.cpp:2401-2403: the selected sum clamped at 2^-14",
            rope_fallback: "llama.cpp src/models/flint.cpp:16: the pattern array is required",
            chat: "the flint family's own convention, which its reader carried before the row",
            ffn_act: "-",
            head: "-",
            per_layer: "-",
        },
    },
    ArchRules {
        arch: Arch::ExaoneMoe,
        // EXAONE's rope type is NEOX (llama-model.cpp's table, the LLM_ARCH_EXAONE_MOE case).
        pairs: RopeMode::Neox,
        // LG's modeling code clamps the gate before the activation, which is
        // ggml's `ggml_swiglu_clamp`, not the generic clamped branch.
        clamp: ClampOrder::GateFirst,
        // The kept weights divide by their sum plus 1e-20, the fork's own guard.
        router_norm: RouterNorm::SelectedSum,
        // The loader ropes a layer exactly where the sliding-window pattern puts it.
        rope_fallback: Some(RopeFallback::WhereWindowed),
        chat: ChatParsers::FromTemplate,
        ffn_act: None,
        head: HeadWeight::Own,
        per_layer: None,
        sources: RuleSources {
            pairs: "llama.cpp src/llama-model.cpp:3799,3834: EXAONE's rope type is NEOX",
            clamp: "LG modeling_exaone_moe.py:227-232: silu(min(g, L))·clamp(u, ±L)",
            router_norm: "LG modeling_exaone_moe.py:269-270: the kept weights over their sum + 1e-20",
            rope_fallback: "llama.cpp src/models/exaone-moe.cpp:7: rope_pattern is is_swa_impl",
            chat: "the file's own template, read by models' formats table",
            ffn_act: "-",
            head: "-",
            per_layer: "-",
        },
    },
    ArchRules {
        arch: Arch::Raon1Nano,
        // No layer rotates: the latent layers are NoPE and the delta layers carry no rope at all,
        // so no head's values are paired.
        pairs: RopeMode::None,
        // The loader reads no clamp key and no layer clamps.
        clamp: ClampOrder::None,
        // A dense feed-forward block in every layer (raon1nano.cpp:23-24): no router, no divisor.
        router_norm: RouterNorm::None,
        // No layer turns, so no layer's rope is named by a pattern the file might carry.
        rope_fallback: None,
        chat: ChatParsers::Fixed {
            tools: Some(ToolFormat::QwenXml),
            reasoning: Some(ReasoningFormat::THINK),
        },
        // The file's keys name no activation: every block takes one SwiGLU over the fused
        // [gate | up] rows of `ffn_up`.
        ffn_act: Some(Act::SwiGlu { limit: None }),
        head: HeadWeight::Tied,
        per_layer: Some(PerLayerRule::TokenTable),
        sources: RuleSources {
            pairs: "llama.cpp src/models/raon1nano.cpp:334: NoPE — no layer rotates",
            clamp: "llama.cpp src/models/raon1nano.cpp:7-63: the loader reads no clamp key",
            router_norm: "llama.cpp src/models/raon1nano.cpp:23-24: a dense FFN in every layer",
            rope_fallback: "-",
            chat: "the raon family's own convention, which its reader carried before the row",
            ffn_act: "llama.cpp src/models/raon1nano.cpp:152-154: one LLM_FFN_SWIGLU on ffn_up's [gate | up] rows",
            head: "the file carries no output tensor and LG's config ties the word embeddings (hub modeling_raon1_nano.py: tie_word_embeddings)",
            per_layer: "llama.cpp src/models/raon1nano.cpp:367-383: p = (RMSNorm(W_mp·e) + √n_ple·table[t])/√2; hub modeling_raon1_nano.py:495-502",
        },
    },
];

/// The description row of `arch`, the file's own architecture string; `None`
/// when no row states its facts.
#[must_use]
pub fn row(arch: &str) -> Option<&'static ArchRules> {
    ROWS.iter().find(|r| r.arch.name() == arch)
}

#[cfg(test)]
mod tests {
    use super::{ClampOrder, HeadWeight, PerLayerRule, ROWS, RopeFallback, row};
    use models::Act;

    /// A row is named by its architecture's own string, and the rows this
    /// table opens with state the facts their reader refuses to guess.
    #[test]
    fn a_row_is_named_by_its_architectures_string() {
        let flint = row("flint").expect("the flint row");
        let exaone = row("exaone-moe").expect("the exaone-moe row");
        let nano = row("raon1nano").expect("the raon1nano row");
        assert_eq!(flint.arch, models::Arch::Flint);
        assert_eq!(exaone.arch, models::Arch::ExaoneMoe);
        assert_eq!(nano.arch, models::Arch::Raon1Nano);
        assert_eq!(flint.clamp, ClampOrder::None);
        assert_eq!(exaone.clamp, ClampOrder::GateFirst);
        assert_eq!(flint.rope_fallback, None);
        assert_eq!(exaone.rope_fallback, Some(RopeFallback::WhereWindowed));
        assert_eq!(nano.ffn_act, Some(Act::SwiGlu { limit: None }));
        assert_eq!(nano.head, HeadWeight::Tied);
        assert_eq!(nano.per_layer, Some(PerLayerRule::TokenTable));
        assert!(row("llama").is_none(), "no row for another family");
        assert!(
            row("raon1").is_none(),
            "the mini dialect states its facts by its keys"
        );
    }

    /// Every fact a row states carries the line that states it, and the lines
    /// name their row's values: a row that states no fact of the reader's own
    /// reads none of them. Mutant: a source dropped for a stated fact.
    #[test]
    fn a_stated_fact_carries_its_source_line() {
        for r in ROWS {
            for (what, stated, source, line) in [
                (
                    "ffn act",
                    r.ffn_act.is_some(),
                    r.sources.ffn_act,
                    r.ffn_act_line(),
                ),
                (
                    "head",
                    r.head == HeadWeight::Tied,
                    r.sources.head,
                    r.tied_head_line(),
                ),
                (
                    "per-layer input",
                    r.per_layer.is_some(),
                    r.sources.per_layer,
                    r.per_layer_line(),
                ),
            ] {
                assert_ne!(
                    stated,
                    source == "-",
                    "{}: {what}'s source is {source:?} for a {} fact",
                    r.arch.name(),
                    if stated { "stated" } else { "denied" }
                );
                assert_eq!(
                    stated,
                    line.is_some(),
                    "{}: {what} states a fact with no line, or a line for none",
                    r.arch.name()
                );
            }
        }
        let nano = row("raon1nano").unwrap();
        assert_eq!(
            nano.ffn_act_line(),
            Some("ffn act = SwiGLU (llama.cpp src/models/raon1nano.cpp:152-154: one LLM_FFN_SWIGLU on ffn_up's [gate | up] rows)".to_string())
        );
        assert!(
            nano.tied_head_line()
                .is_some_and(|l| l.starts_with("head = tied ("))
        );
        assert!(
            nano.per_layer_line()
                .is_some_and(|l| l.starts_with("per-layer input = token table, layout 2 ("))
        );
    }

    /// A limit of at most 1e-6 clamps nothing under either order; a larger one
    /// takes the row's act. The none order never reaches `act` with a non-zero
    /// limit: the reader refuses the file first.
    #[test]
    fn a_limit_takes_the_rows_act() {
        let flint = row("flint").unwrap();
        let exaone = row("exaone-moe").unwrap();
        for rules in [flint, exaone] {
            assert_eq!(rules.act(0.0), Act::SwiGlu { limit: None });
            assert_eq!(rules.act(1e-6), Act::SwiGlu { limit: None });
        }
        assert_eq!(
            exaone.act(7.0),
            Act::SwiGluGateFirst { limit: 7.0 },
            "the exaone-moe row clamps the gate first"
        );
    }
}
