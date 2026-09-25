//! The prompt the stage-1 oracle set was dumped for: the manifest's `# tokens` line.

/// The prompt's ids. A manifest without the line, or with an id that is not a
/// token id, is a named panic.
pub fn tokens() -> Vec<u32> {
    let (_, text) = super::manifest::read();
    let line = text
        .lines()
        .find_map(|l| l.strip_prefix("# tokens\t"))
        .unwrap_or_else(|| panic!("the oracle manifest has no `# tokens` line"));
    let ids: Vec<u32> = line
        .split(',')
        .map(|s| {
            s.trim()
                .parse()
                .unwrap_or_else(|e| panic!("the oracle's token id {s:?} is not a token id ({e})"))
        })
        .collect();
    assert!(
        !ids.is_empty(),
        "the oracle manifest's `# tokens` line is empty"
    );
    ids
}
