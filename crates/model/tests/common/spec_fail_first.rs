//! The coverage check's FAIL-first specimen, for the header tests of the
//! files a program runs (`hw_*_spec`).

use std::fmt::Write as _;

use model::arch::coverage::{self, AVAILABLE, Available};
use model::arch::models::ModelSpec;
use model::placement::ModelTensors;

use crate::spec_view::items;

/// The check's FAIL-first specimen: with the row of [`AVAILABLE`] at `at`
/// removed, the check lists `want` (the need that row covered, on its layers)
/// beside the items the whole table leaves; restored, it lists only those.
pub fn fail_first(
    out: &mut String,
    bad: &mut Vec<String>,
    spec: &ModelSpec,
    tensors: &ModelTensors,
    at: &str,
    want: &str,
) {
    let whole = items(&coverage::check(spec, tensors));
    let table: Vec<Available> = AVAILABLE.iter().copied().filter(|a| a.at != at).collect();
    assert_eq!(
        table.len() + 1,
        AVAILABLE.len(),
        "one row of AVAILABLE is at {at}"
    );
    let less = items(&coverage::check_with(spec, tensors, &table));
    let added: Vec<&String> = less.iter().filter(|l| !whole.contains(l)).collect();
    let _ = writeln!(out, "coverage without `{at}`: added {added:?}");
    if added != [want] {
        bad.push(format!(
            "without `{at}` the check added {added:?}, want [{want:?}]"
        ));
    }
    let _ = writeln!(out, "coverage with the whole table: {whole:?}");
}
