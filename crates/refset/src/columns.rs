//! Header-named columns: a manifest's column line (`# kind\tname\t…`,
//! `# image columns\tname sha256 …`) names the fields of one kind of row, and
//! a row is read by those names. A field the line does not name, or a row
//! wider or narrower than its line, is `Malformed` — never a default.

use crate::RefError;
use std::fmt::Display;
use std::str::FromStr;

/// The names of one row kind's fields, from its column line: field 0 is the
/// kind, field `i` is `names[i]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Columns {
    names: Vec<String>,
}

impl Columns {
    /// `first` is the name the line gives field 0 (`kind` in a dump's tensor
    /// line, else the row kind), `rest` the names of the fields after it.
    pub(crate) fn new<'a>(first: &str, rest: impl IntoIterator<Item = &'a str>) -> Columns {
        Columns {
            names: std::iter::once(first.to_string())
                .chain(rest.into_iter().map(str::to_string))
                .collect(),
        }
    }

    /// How many fields a row of this kind has.
    pub(crate) fn len(&self) -> usize {
        self.names.len()
    }

    fn find(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    /// `line` split on tabs as a row of this kind; a row whose width is not
    /// the line's is `Malformed` at `at`.
    pub(crate) fn row<'a>(
        &'a self,
        line: &'a str,
        at: &'a dyn Display,
    ) -> Result<Row<'a>, RefError> {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != self.len() {
            return Err(RefError::malformed(
                at,
                format!(
                    "a {} row of {} fields, its column line names {}",
                    f.first().copied().unwrap_or(""),
                    f.len(),
                    self.len()
                ),
            ));
        }
        Ok(Row { cols: self, f, at })
    }
}

/// One data row read by its kind's column names.
pub(crate) struct Row<'a> {
    cols: &'a Columns,
    f: Vec<&'a str>,
    at: &'a dyn Display,
}

impl<'a> Row<'a> {
    /// Field `name`; a column line that does not name it is `Malformed`.
    pub(crate) fn text(&self, name: &str) -> Result<&'a str, RefError> {
        self.opt(name).ok_or_else(|| {
            RefError::malformed(self.at, format!("the column line names no {name} field"))
        })
    }

    /// Field `name`, `None` when the column line does not name it (a column
    /// a later writer added).
    pub(crate) fn opt(&self, name: &str) -> Option<&'a str> {
        self.cols.find(name).and_then(|i| self.f.get(i).copied())
    }

    /// Field `name` parsed; a value that does not parse is `Malformed`,
    /// naming the field and the value.
    pub(crate) fn parse<T>(&self, name: &str) -> Result<T, RefError>
    where
        T: FromStr,
        T::Err: Display,
    {
        parse_field(self.text(name)?, name, self.at)
    }

    /// [`parse`](Self::parse) of a field the column line may not name.
    pub(crate) fn parse_opt<T>(&self, name: &str) -> Result<Option<T>, RefError>
    where
        T: FromStr,
        T::Err: Display,
    {
        self.opt(name)
            .map(|v| parse_field(v, name, self.at))
            .transpose()
    }
}

/// `v`, the value of field `name`, parsed; `Malformed` at `at` otherwise.
pub(crate) fn parse_field<T>(v: &str, name: &str, at: &dyn Display) -> Result<T, RefError>
where
    T: FromStr,
    T::Err: Display,
{
    v.parse()
        .map_err(|e| RefError::malformed(at, format!("{name} {v:?}: {e}")))
}
