//! The prompt the stage-1 oracle set was dumped for: the manifest's `# tokens` line.

/// The prompt's ids. A manifest without the line, or with an id that is not a
/// token id, is a named panic.
pub fn tokens() -> Vec<u32> {
    let ids = super::manifest::read()
        .header
        .tokens
        .unwrap_or_else(|| panic!("the oracle manifest has no `# tokens` line"));
    assert!(
        !ids.is_empty(),
        "the oracle manifest's `# tokens` line is empty"
    );
    ids
}
