//! The chat-template spellings of the tool-call and reasoning formats: one table of each, keyed by
//! the format. A reader of an architecture string several model families share asks it which
//! format a file's own template teaches ([`ToolFormat::of_template`],
//! [`ReasoningFormat::of_template`]); a format with no row here is named by its family's reader.

use crate::function_xml::{CALL_OPEN, FUNCTION_OPEN, PARAM_OPEN};
use crate::{ReasoningFormat, ToolFormat};

/// The tool formats a template alone can name, each with the markup every call of it spells: the
/// first row whose markers all appear in the template is the template's.
const TOOLS: &[(ToolFormat, &[&str])] = &[
    (ToolFormat::QwenXml, &[CALL_OPEN, FUNCTION_OPEN, PARAM_OPEN]),
    (ToolFormat::Mistral, &["[TOOL_CALLS]", "[ARGS]"]),
];

/// The reasoning formats a template alone can name: the one whose opening tag the template writes.
const SPANS: &[ReasoningFormat] = &[ReasoningFormat::THINK];

/// The tool format `template` teaches; `None`: no markup of a row.
pub(crate) fn tool_of(template: &str) -> Option<ToolFormat> {
    TOOLS
        .iter()
        .find(|(_, markers)| markers.iter().all(|marker| template.contains(marker)))
        .map(|&(format, _)| format)
}

/// The reasoning format `template` opens a span of; `None`: no span of a row.
pub(crate) fn span_of(template: &str) -> Option<ReasoningFormat> {
    SPANS
        .iter()
        .copied()
        .find(|&ReasoningFormat::ThinkSpan { open, .. }| template.contains(open))
}
