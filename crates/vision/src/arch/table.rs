//! The tensor table of an encoder file and the check of a file against it, written once.
//!
//! A projector module builds its table as data ([`Expected`] rows: name, dims, type, and whatever
//! the module records about where the tensor lives, `H`) from its hyperparameters, and calls
//! [`check`]: every tensor of the file is known, none appears twice, each has its shape and type,
//! and none of the table is missing. A tensor under a name the table does not know is refused by
//! that name before anything else is looked at, so a file of another layout fails on its first
//! unfamiliar tensor, not on a shape.

use std::collections::{HashMap, HashSet};

use gguf::GgmlType;

use crate::VisionError;

/// One tensor the file must hold, dims in ggml `ne[]` order (the contiguous axis first). `H` is
/// the module's own note on the row, `()` for a module that has none (for example where the
/// tensor lives once loaded).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expected<H> {
    pub name: String,
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    pub home: H,
}

impl<H> Expected<H> {
    /// A row: `name` at `dims` in type `ty`, noted `home`.
    #[must_use]
    pub fn row(name: String, dims: &[u64], ty: GgmlType, home: H) -> Expected<H> {
        Expected {
            name,
            dims: dims.to_vec(),
            ty,
            home,
        }
    }

    /// The tensor's bytes, in the file and wherever it is loaded: ggml's `type_size` for each
    /// `blck_size` values. A type ggml has no size for, or a value count that is not whole blocks,
    /// is a table that was built wrong and panics by the row's name.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        let values: u64 = self.dims.iter().product();
        let (Some(block), Some(size)) = (self.ty.blck_size(), self.ty.type_size()) else {
            panic!(
                "tensor {}: {} has no block or type size in ggml's table",
                self.name, self.ty
            );
        };
        assert!(
            values.is_multiple_of(block),
            "tensor {}: {values} values are not whole {} blocks of {block}",
            self.name,
            self.ty
        );
        values / block * size
    }
}

/// Check a file's tensors (name, ggml dims, type) against `table`: every one known, none twice,
/// each at its shape and type, and none of the table missing. `projector` names the table in the
/// refusal of an unknown name. Returns how many were named.
pub fn check<'a, H>(
    projector: &str,
    table: &[Expected<H>],
    tensors: impl IntoIterator<Item = (&'a str, &'a [u64], GgmlType)>,
) -> Result<usize, VisionError> {
    let by_name: HashMap<&str, &Expected<H>> = table.iter().map(|e| (e.name.as_str(), e)).collect();
    let mut seen: HashSet<&str> = HashSet::with_capacity(table.len());
    let err = |name: &str, detail: String| VisionError::Tensor {
        name: name.to_string(),
        detail,
    };
    for (name, dims, ty) in tensors {
        let want = by_name
            .get(name)
            .ok_or_else(|| err(name, format!("is not a {projector} tensor name")))?;
        if !seen.insert(want.name.as_str()) {
            return Err(err(name, "appears twice".into()));
        }
        if dims != want.dims.as_slice() {
            return Err(err(
                name,
                format!("has dims {dims:?}; the table gives {:?}", want.dims),
            ));
        }
        if ty != want.ty {
            return Err(err(
                name,
                format!("is {ty}; this crate reads it as {}", want.ty),
            ));
        }
    }
    if let Some(missing) = table.iter().find(|e| !seen.contains(e.name.as_str())) {
        return Err(err(&missing.name, "is missing from the file".into()));
    }
    Ok(seen.len())
}

/// A table as a file's tensor list, and the checks of the list's refusals the projector modules
/// share.
#[cfg(test)]
pub(crate) mod testing {
    use gguf::GgmlType;

    use super::Expected;
    use crate::VisionError;
    use crate::arch::naming::{attn_out_weight, patch_embd_weight};

    /// A file's tensors as owned `(name, dims, type)` triples.
    pub(crate) type List = Vec<(String, Vec<u64>, GgmlType)>;

    /// The table's rows as a file would list them.
    pub(crate) fn list<H>(table: &[Expected<H>]) -> List {
        table
            .iter()
            .map(|e| (e.name.clone(), e.dims.clone(), e.ty))
            .collect()
    }

    /// A list as the borrowed triples the checks take.
    pub(crate) fn triples(l: &List) -> impl Iterator<Item = (&str, &[u64], GgmlType)> {
        l.iter().map(|(n, d, ty)| (n.as_str(), d.as_slice(), *ty))
    }

    /// The module's own check of a list, the error as its text.
    pub(crate) fn text<T>(r: Result<T, VisionError>) -> Result<T, String> {
        r.map_err(|e| e.to_string())
    }

    /// The position of the row named `name`.
    pub(crate) fn row_of(l: &List, name: &str) -> usize {
        l.iter()
            .position(|r| r.0 == name)
            .unwrap_or_else(|| panic!("the list has no row {name}"))
    }

    /// The table itself passes with `n` rows named; one renamed tensor is refused by its new
    /// name, a dropped one is named as missing, a doubled one as twice, a reshaped one by its
    /// dims, a retyped one by its type. Each change picks its row by name from the names every
    /// projector type holds (the patch embedding, block 0's attention output); `last` is the
    /// name of the row to drop. `run` is the module's own check of a list.
    pub(crate) fn changes_are_refused_by_name<H>(
        projector: &str,
        table: &[Expected<H>],
        n: usize,
        last: &str,
        run: impl Fn(&List) -> Result<usize, String>,
    ) {
        let (first, probe) = (patch_embd_weight(), attn_out_weight(0));
        assert_eq!(run(&list(table)), Ok(n));

        let mut renamed = list(table);
        let i = row_of(&renamed, &probe);
        renamed[i].0 = "v.blk.0.attn_o.weight".into();
        assert_eq!(
            run(&renamed).unwrap_err(),
            format!("tensor v.blk.0.attn_o.weight: is not a {projector} tensor name")
        );
        let mut dropped = list(table);
        let i = row_of(&dropped, last);
        dropped.remove(i);
        assert_eq!(
            run(&dropped).unwrap_err(),
            format!("tensor {last}: is missing from the file")
        );
        let mut doubled = list(table);
        let row = doubled[row_of(&doubled, &first)].clone();
        doubled.push(row);
        assert_eq!(
            run(&doubled).unwrap_err(),
            format!("tensor {first}: appears twice")
        );
        let mut reshaped = list(table);
        let i = row_of(&reshaped, &probe);
        reshaped[i].1.push(1);
        let want = format!("tensor {probe}: has dims {:?}", reshaped[i].1);
        let got = run(&reshaped).unwrap_err();
        assert!(got.starts_with(&want), "{got}");
        let mut retyped = list(table);
        let i = row_of(&retyped, &probe);
        assert_ne!(
            retyped[i].2,
            GgmlType::Q4_K,
            "the probe row is not q4_K already"
        );
        retyped[i].2 = GgmlType::Q4_K;
        let want = format!("tensor {probe}: is q4_K");
        let got = run(&retyped).unwrap_err();
        assert!(got.starts_with(&want), "{got}");
    }
}
