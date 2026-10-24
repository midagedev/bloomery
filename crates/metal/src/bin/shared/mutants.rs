// The mutant support both Metal gates share: included into each gate's
// `mutants.rs`, whose own tables `MUTANTS` and `TENSOR_MUTANTS` it walks.

use bloomery_metal::library::{SOURCE, TENSOR_SOURCE};

/// The program a mutant's edits go into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Program {
    /// The kernels' library ([`SOURCE`]).
    Main,
    /// The tensor-op GEMMs' library ([`TENSOR_SOURCE`]), which the tensor
    /// clauses force.
    Tensor,
}

/// A defect and the clause that must be red on it.
pub struct Mutant {
    pub name: &'static str,
    pub clause: &'static str,
    pub why: &'static str,
    /// `(find, replace, occurrences of find)`, applied in order.
    pub edits: &'static [(&'static str, &'static str, usize)],
}

/// Every mutant of both tables, with the program it edits.
pub fn all() -> impl Iterator<Item = (&'static Mutant, Program)> {
    MUTANTS
        .iter()
        .map(|m| (m, Program::Main))
        .chain(TENSOR_MUTANTS.iter().map(|m| (m, Program::Tensor)))
}

/// The mutant of this name, and the program it edits.
pub fn find(name: &str) -> Option<(&'static Mutant, Program)> {
    all().find(|(m, _)| m.name == name)
}

/// The source of `program` with the mutant's edits applied; an edit whose
/// find text occurs another number of times than it names is refused.
pub fn source_of(mutant: &Mutant, program: Program) -> Result<String, String> {
    let mut text = match program {
        Program::Main => SOURCE,
        Program::Tensor => TENSOR_SOURCE,
    }
    .to_string();
    for (find, replace, count) in mutant.edits {
        let found = text.matches(find).count();
        if found != *count {
            return Err(format!(
                "mutant {}: {find:?} occurs {found} times, not {count}",
                mutant.name
            ));
        }
        text = text.replace(find, replace);
    }
    Ok(text)
}
