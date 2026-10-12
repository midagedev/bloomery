//! [`SectionOrder`]: which position slot feeds each rotation section.

use super::MropeError;

/// The slots of a position, in the order the batch holds them: the time `t`, the line `h` (ggml's
/// `y`), the column `w` (ggml's `x`) and the extra `z`.
pub const SLOTS: usize = 4;

/// How the text file's `rope.section_order` maps the four position slots `[t, h, w, z]` onto the
/// four rotation sections (the file's `rope.dimension_sections`, in that order).
///
/// llama.cpp reads the key at `src/llama-model.cpp:1473` (default `tyxz`) and applies it in
/// `llm_graph_input_pos::set_input` (`src/llama-graph.cpp:176-190`) to the positions of embedding
/// rows only; a text row is `[p, p, p, 0]` under either order, so only an image row moves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SectionOrder {
    /// Section `i` reads slot `i`: the order of Qwen3.6 and Qwen3-VL.
    #[default]
    Tyxz,
    /// Section 0 reads `z`, sections 1 and 2 the line and the column, section 3 reads `t`:
    /// MiniCPM-V 4.7's, whose canvas keeps the origin in `z`.
    Zyxt,
}

impl SectionOrder {
    /// The value the key holds for this order.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            SectionOrder::Tyxz => "tyxz",
            SectionOrder::Zyxt => "zyxt",
        }
    }

    /// The order a `rope.section_order` value names; any other string is refused by name.
    pub fn parse(value: &str) -> Result<SectionOrder, MropeError> {
        match value {
            "tyxz" => Ok(SectionOrder::Tyxz),
            "zyxt" => Ok(SectionOrder::Zyxt),
            other => Err(MropeError::SectionOrder {
                value: other.to_string(),
            }),
        }
    }

    /// The slot each section reads, `[t, h, w, z]` being slots 0 to 3
    /// (`slot_of_section` of `src/llama-graph.cpp:176-190`).
    #[must_use]
    pub const fn slot_of_section(self) -> [usize; SLOTS] {
        match self {
            SectionOrder::Tyxz => [0, 1, 2, 3],
            SectionOrder::Zyxt => [3, 1, 2, 0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SectionOrder;
    use crate::mrope::MropeError;

    /// The two names the loader takes, and the slot maps `src/llama-graph.cpp:176-190` gives them;
    /// any other string is refused by name, an absent key being the caller's `Tyxz`.
    #[test]
    fn the_two_orders_and_their_slot_maps() {
        assert_eq!(SectionOrder::parse("tyxz"), Ok(SectionOrder::Tyxz));
        assert_eq!(SectionOrder::parse("zyxt"), Ok(SectionOrder::Zyxt));
        assert_eq!(SectionOrder::Tyxz.slot_of_section(), [0, 1, 2, 3]);
        assert_eq!(SectionOrder::Zyxt.slot_of_section(), [3, 1, 2, 0]);
        assert_eq!(SectionOrder::default(), SectionOrder::Tyxz);
        for name in ["", "ZYXT", "txyz", "zyxt "] {
            assert_eq!(
                SectionOrder::parse(name),
                Err(MropeError::SectionOrder {
                    value: name.to_string()
                }),
                "{name:?}"
            );
        }
        assert_eq!(
            SectionOrder::parse("xyzt").unwrap_err().to_string(),
            "rope.section_order is \"xyzt\"; this crate runs \"tyxz\" and \"zyxt\""
        );
    }
}
