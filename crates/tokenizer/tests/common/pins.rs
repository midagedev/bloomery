//! The reference's pinned ids a vocabulary gate compares with: rows of
//! `<text bytes in hex>\t<ids, special tokens parsed>\t<ids, --no-parse-special>`, `#` comment lines,
//! and the `# complete\t<rows>` trailer the oracle script writes last.

use tokenizer::Tokenizer;

/// The bytes a row's hex column names.
fn unhex(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("a hex byte"))
        .collect()
}

/// A row's id column.
fn ids(col: &str) -> Vec<u32> {
    col.split_whitespace()
        .map(|id| id.parse().expect("an id"))
        .collect()
}

/// Encode every row of `pins` with `tok`, special tokens parsed and without, and return the row
/// count and one line per (row, mode) whose ids are not the reference's. The trailer must name every
/// row, so a pins file cut short is a failure, not a smaller gate.
pub fn compare(tok: &Tokenizer, pins: &str) -> (usize, Vec<String>) {
    let mut differ = Vec::new();
    let mut rows = 0;
    let mut complete = None;
    for line in pins.lines() {
        if let Some(n) = line.strip_prefix("# complete\t") {
            complete = Some(n.parse::<usize>().expect("a count"));
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 3, "{line}");
        let text = unhex(cols[0]);
        rows += 1;
        for (parse, want) in [(true, ids(cols[1])), (false, ids(cols[2]))] {
            let got = tok.encode(&text, false, parse);
            if got != want {
                differ.push(format!(
                    "{:?} parse_special {parse}: ours {got:?}, reference {want:?}",
                    String::from_utf8_lossy(&text)
                ));
            }
        }
    }
    assert_eq!(complete, Some(rows), "the pins' trailer names every row");
    (rows, differ)
}
