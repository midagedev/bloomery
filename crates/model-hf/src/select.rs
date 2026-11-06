//! The tensors a split holds when `prepare` is asked for some of a folder's: `--drop` leaves the
//! tensors a pattern names out, `--only` keeps those and no others. A pattern is a glob on the
//! GGUF tensor names the split is written under (`*` any run of characters, `?` one), and one
//! that names no tensor is refused, so a misspelt pattern is never a silent no-op. A tensor's
//! `.scale` goes with it.

use std::collections::HashSet;

use crate::folder::{Error, Make, Planned};

/// The tensors of a split kept: all of them, all but those a pattern names, or only those.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Select {
    #[default]
    All,
    Drop(Vec<String>),
    Only(Vec<String>),
}

impl Select {
    /// The selection `--drop` and `--only` patterns ask for. Both at once, and an empty pattern,
    /// are refused: what they would keep is ambiguous.
    pub fn of(drop: Vec<String>, only: Vec<String>) -> Result<Select, Error> {
        if !drop.is_empty() && !only.is_empty() {
            return Err(Error::Refused(
                "--drop and --only both given: a split holds all but the dropped tensors or only \
                 the chosen ones, not both"
                    .into(),
            ));
        }
        if let Some(i) = drop.iter().chain(&only).position(String::is_empty) {
            let flag = if i < drop.len() { "--drop" } else { "--only" };
            return Err(Error::Refused(format!(
                "{flag} '': an empty pattern names no tensor"
            )));
        }
        Ok(match (drop.is_empty(), only.is_empty()) {
            (true, true) => Select::All,
            (false, _) => Select::Drop(drop),
            (_, false) => Select::Only(only),
        })
    }

    /// `planned` with the tensors this selection leaves out removed, in their order. A pattern
    /// that matches no tensor name, and a selection that leaves no tensor, are refused by name.
    pub fn apply<S>(&self, planned: Vec<Planned<S>>) -> Result<Vec<Planned<S>>, Error> {
        let (flag, patterns, keep_matches) = match self {
            Select::All => return Ok(planned),
            Select::Drop(p) => ("--drop", p, false),
            Select::Only(p) => ("--only", p, true),
        };
        for pattern in patterns {
            let hit = planned
                .iter()
                .any(|p| !matches!(p.make, Make::Scale { .. }) && glob(pattern, &p.decl.name));
            if !hit {
                return Err(Error::Refused(format!(
                    "{flag} {pattern:?}: no tensor of the split has a name it matches"
                )));
            }
        }
        let mut left_out: HashSet<String> = HashSet::new();
        let mut kept = Vec::with_capacity(planned.len());
        for p in planned {
            let out = match &p.make {
                Make::Scale { of } => left_out.contains(of),
                _ => patterns.iter().any(|pat| glob(pat, &p.decl.name)) != keep_matches,
            };
            if out {
                left_out.insert(p.decl.name.clone());
            } else {
                kept.push(p);
            }
        }
        if kept.is_empty() {
            return Err(Error::Refused(format!(
                "{flag} {patterns:?}: no tensor is left in the split"
            )));
        }
        Ok(kept)
    }
}

/// Whether the glob `pattern` matches the whole of `name`: `*` is any run of characters, none
/// included, and `?` one character.
pub fn glob(pattern: &str, name: &str) -> bool {
    let (p, n): (Vec<char>, Vec<char>) = (pattern.chars().collect(), name.chars().collect());
    let (mut pi, mut ni) = (0, 0);
    let mut star: Option<(usize, usize)> = None;
    while ni < n.len() {
        match p.get(pi) {
            Some('*') => {
                star = Some((pi, ni));
                pi += 1;
            }
            Some(&c) if c == '?' || c == n[ni] => {
                pi += 1;
                ni += 1;
            }
            _ => match star {
                Some((sp, sn)) => {
                    pi = sp + 1;
                    ni = sn + 1;
                    star = Some((sp, sn + 1));
                }
                None => return false,
            },
        }
    }
    p[pi..].iter().all(|&c| c == '*')
}

#[cfg(test)]
mod tests;
