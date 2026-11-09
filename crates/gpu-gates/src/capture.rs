//! The `capture` records an open prints when the decode step and the verify
//! pass are captured. Every `OpenLog`/`RowsLog` impl that prints them writes
//! these two; one that only keeps the node count stores it itself.

use crate::record::{self, Record};

/// Where a capture line goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Out {
    Stdout,
    Stderr,
}

impl Out {
    fn err(self) -> bool {
        self == Out::Stderr
    }
}

/// The one-token step's `capture` record: its graph's `nodes`.
pub fn step(nodes: usize, out: Out) {
    Record::new(&record::CAPTURE)
        .u("graph_nodes", nodes)
        .emit(out.err());
}

/// The refusal of a verify capture whose row count is not the pair pass's
/// `expect`.
pub fn check_pair(rows: usize, expect: usize) -> Result<(), String> {
    if rows == expect {
        Ok(())
    } else {
        Err(format!(
            "a verify capture of {rows} rows; the pair pass runs {expect}"
        ))
    }
}

/// The pair pass's `capture_pair` record of its graph's `nodes`, once the
/// capture is of `expect` rows ([`check_pair`]).
pub fn pair(rows: usize, expect: usize, nodes: usize, out: Out) -> Result<(), String> {
    check_pair(rows, expect)?;
    Record::new(&record::CAPTURE_PAIR)
        .u("pair_graph_nodes", nodes)
        .emit(out.err());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_capture_of_another_row_count_is_refused_by_name() {
        assert_eq!(check_pair(2, 2), Ok(()));
        for rows in [1, 3, 8] {
            let why = check_pair(rows, 2).unwrap_err();
            assert_eq!(
                why,
                format!("a verify capture of {rows} rows; the pair pass runs 2")
            );
        }
    }
}
