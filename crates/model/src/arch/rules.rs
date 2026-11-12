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

use models::{Arch, ReasoningFormat, RopeMode, RouterNorm, ToolFormat};

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

/// The source line of each fact of a row, as its `defaults` line prints it.
#[derive(Clone, Copy, Debug)]
pub struct RuleSources {
    pub pairs: &'static str,
    pub clamp: &'static str,
    pub router_norm: &'static str,
    pub rope_fallback: &'static str,
    pub chat: &'static str,
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
        sources: RuleSources {
            pairs: "llama.cpp src/llama-model.cpp:3769,3834: Flint's rope type is NEOX",
            clamp: "llama.cpp src/models/flint.cpp:3-19: the loader reads no clamp key",
            router_norm: "llama.cpp src/llama-graph.cpp:2401-2403: the selected sum clamped at 2^-14",
            rope_fallback: "llama.cpp src/models/flint.cpp:16: the pattern array is required",
            chat: "the flint family's own convention, which its reader carried before the row",
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
        sources: RuleSources {
            pairs: "llama.cpp src/llama-model.cpp:3799,3834: EXAONE's rope type is NEOX",
            clamp: "LG modeling_exaone_moe.py:227-232: silu(min(g, L))·clamp(u, ±L)",
            router_norm: "LG modeling_exaone_moe.py:269-270: the kept weights over their sum + 1e-20",
            rope_fallback: "llama.cpp src/models/exaone-moe.cpp:7: rope_pattern is is_swa_impl",
            chat: "the file's own template, read by models' formats table",
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
    use super::{ClampOrder, RopeFallback, row};
    use models::Act;

    /// A row is named by its architecture's own string, and the two rows this
    /// table opens with state the facts their reader refuses to guess.
    #[test]
    fn a_row_is_named_by_its_architectures_string() {
        let flint = row("flint").expect("the flint row");
        let exaone = row("exaone-moe").expect("the exaone-moe row");
        assert_eq!(flint.arch, models::Arch::Flint);
        assert_eq!(exaone.arch, models::Arch::ExaoneMoe);
        assert_eq!(flint.clamp, ClampOrder::None);
        assert_eq!(exaone.clamp, ClampOrder::GateFirst);
        assert_eq!(flint.rope_fallback, None);
        assert_eq!(exaone.rope_fallback, Some(RopeFallback::WhereWindowed));
        assert!(row("llama").is_none(), "no row for another family");
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
