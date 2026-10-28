// The mutant support both Metal gates share: a module of each gate's
// `mutants.rs`, whose own tables `MUTANTS` and `TENSOR_MUTANTS` it walks.

use bloomery_metal::instance::Instance;
use bloomery_metal::library::sources_for;

/// The program a mutant's edits go into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Program {
    /// The kernels' library.
    Main,
    /// The tensor-op GEMMs' library, which the tensor clauses force.
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

/// Every mutant of a gate's two tables, with the program it edits: `main`
/// edits the kernels' library, `tensor` the tensor-op GEMMs'.
pub fn all(
    main: &'static [Mutant],
    tensor: &'static [Mutant],
) -> impl Iterator<Item = (&'static Mutant, Program)> {
    main.iter()
        .map(|m| (m, Program::Main))
        .chain(tensor.iter().map(|m| (m, Program::Tensor)))
}

/// The source of `program` of `inst`'s kernels with the mutant's edits
/// applied; an edit whose find text occurs another number of times than it
/// names is refused.
pub fn source_at(inst: &Instance, mutant: &Mutant, program: Program) -> Result<String, String> {
    let (main, tensor) = sources_for(inst).map_err(|e| e.to_string())?;
    let base = match program {
        Program::Main => main,
        Program::Tensor => tensor,
    };
    edited(&base, mutant)
}

/// `base` with the mutant's edits applied, in order.
fn edited(base: &str, mutant: &Mutant) -> Result<String, String> {
    let mut text = base.to_string();
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
