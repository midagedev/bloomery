//! Where a server's model file comes from: a path (`-m PATH`, `--model-file
//! PATH`, and whatever other spellings a binary gives the flag), a repo
//! (`--hf <repo>[:<quant>]`), the gates' environment variable
//! `BLOOMERY_REF_MODEL`, or a catalog model by name ([`CatalogRef`], which the
//! pick turns into a repo). A model is named once: a flag given twice, a path
//! beside `--hf`, and a flag beside the variable when the two name different
//! files are each refused by name with both values. A path flag equal to the
//! variable is the same file named twice and stands. A catalog name beside
//! one of those only chooses the seat, as a seat word does; a catalog variant
//! beside one is the model named twice.

use crate::catalog::CatalogRef;
use crate::{HfError, RepoRef};

/// The model flags taken out of a binary's arguments.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Flags {
    /// The path flag's spelling and value.
    pub path: Option<(String, String)>,
    /// `--hf`'s value.
    pub hf: Option<String>,
    /// The catalog model a binary's name flag gave, parsed by the caller; [`take`] never sets it.
    pub name: Option<CatalogRef>,
}

/// `args` with every model flag and its value taken out: `path_flags`
/// spell the path (`-m`, `--model-file`, …), `--hf` the repo. A flag with
/// no value, and a model flag given twice by any of its spellings, is
/// refused by name with both values.
pub fn take(args: &[String], path_flags: &[&str]) -> Result<(Flags, Vec<String>), HfError> {
    let mut f = Flags::default();
    let mut rest = Vec::with_capacity(args.len());
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let is_path = path_flags.contains(&a.as_str());
        if !is_path && a != "--hf" {
            rest.push(a.clone());
            continue;
        }
        let v = it
            .next()
            .ok_or_else(|| HfError::Source(format!("{a} needs a value")))?
            .clone();
        if let Some((flag, old)) = &f.path {
            return Err(HfError::Source(format!(
                "the model is named twice: {flag} {old} and {a} {v}"
            )));
        }
        if let Some(old) = &f.hf {
            return Err(HfError::Source(format!(
                "the model is named twice: --hf {old} and {a} {v}"
            )));
        }
        if is_path {
            f.path = Some((a.clone(), v));
        } else {
            f.hf = Some(v);
        }
    }
    Ok((f, rest))
}

/// Where the model comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// A path, from a flag or from the variable.
    File(String),
    /// A repo to fetch.
    Hf(RepoRef),
}

/// What the flags and the variable name: a source, or a catalog model the pick turns into one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Selected {
    /// A path or a repo.
    Source(Source),
    /// A catalog model by name, no model flag beside it.
    Catalog(CatalogRef),
}

/// What `flags` and the variable's value `env` name; `None` when nothing names a model. A
/// catalog name beside a path, a repo or the variable only chooses the seat and the other
/// source stands; a catalog variant beside one of them is the model named twice.
pub fn select(flags: &Flags, env: Option<&str>) -> Result<Option<Selected>, HfError> {
    let source = file_or_repo(flags, env)?;
    match (flags.name, source) {
        (None, source) => Ok(source.map(Selected::Source)),
        (Some(name), None) => Ok(Some(Selected::Catalog(name))),
        (Some(name), Some(source)) => match name.variant {
            None => Ok(Some(Selected::Source(source))),
            Some(_) => Err(HfError::Source(format!(
                "the model is named twice: --model {name} and {}",
                other_source(flags, env)
            ))),
        },
    }
}

/// The source `flags` and the variable's value `env` name; `None` when
/// neither names one. An empty `env` counts as unset. A catalog name alone is
/// not a source: [`select`] returns it for the pick, and this refuses it.
pub fn resolve(flags: &Flags, env: Option<&str>) -> Result<Option<Source>, HfError> {
    match select(flags, env)? {
        None => Ok(None),
        Some(Selected::Source(source)) => Ok(Some(source)),
        Some(Selected::Catalog(name)) => Err(HfError::Source(format!(
            "{name} names a catalog model, which the pick turns into a source, not `resolve`"
        ))),
    }
}

/// How the path flag, `--hf` or the variable that `file_or_repo` took is spelled.
fn other_source(flags: &Flags, env: Option<&str>) -> String {
    match (&flags.path, &flags.hf, env.filter(|e| !e.is_empty())) {
        (Some((flag, p)), _, _) => format!("{flag} {p}"),
        (None, Some(h), _) => format!("--hf {h}"),
        (None, None, Some(e)) => format!("BLOOMERY_REF_MODEL={e}"),
        (None, None, None) => String::new(),
    }
}

/// The path, repo or variable `flags` and `env` name, the catalog name aside.
fn file_or_repo(flags: &Flags, env: Option<&str>) -> Result<Option<Source>, HfError> {
    let env = env.filter(|e| !e.is_empty());
    match (&flags.path, &flags.hf, env) {
        (Some((flag, p)), _, Some(e)) if p != e => Err(HfError::Source(format!(
            "the model is named twice: {flag} {p} and BLOOMERY_REF_MODEL={e}"
        ))),
        (_, Some(h), Some(e)) => Err(HfError::Source(format!(
            "the model is named twice: --hf {h} and BLOOMERY_REF_MODEL={e}"
        ))),
        (Some((_, p)), _, _) => Ok(Some(Source::File(p.clone()))),
        (None, Some(h), None) => Ok(Some(Source::Hf(RepoRef::parse(h)?))),
        (None, None, Some(e)) => Ok(Some(Source::File(e.to_owned()))),
        (None, None, None) => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PATHS: &[&str] = &["-m", "--model-file"];

    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn the_model_flags_come_out_and_the_rest_stays() {
        let (f, rest) = take(
            &args(&["--model", "qwen3", "-m", "/x.gguf", "--port", "0"]),
            PATHS,
        )
        .expect("take");
        assert_eq!(f.path, Some(("-m".to_owned(), "/x.gguf".to_owned())));
        assert_eq!(rest, args(&["--model", "qwen3", "--port", "0"]));
        let (f, _) = take(&args(&["--hf", "a/b:Q4_K_M"]), PATHS).expect("take");
        assert_eq!(f.hf.as_deref(), Some("a/b:Q4_K_M"));
    }

    #[test]
    fn a_model_named_twice_by_flags_is_refused_with_both_values() {
        for a in [
            &["-m", "/a", "--model-file", "/b"][..],
            &["-m", "/a", "-m", "/a"],
            &["-m", "/a", "--hf", "x/y"],
            &["--hf", "x/y", "--hf", "x/z"],
        ] {
            let e = take(&args(a), PATHS).expect_err("twice").to_string();
            assert!(e.contains(a[1]) && e.contains(a[3]), "{e}");
        }
        assert!(take(&args(&["-m"]), PATHS).is_err());
    }

    #[test]
    fn a_flag_and_the_variable_are_one_model_or_refused() {
        let m = |p: &str| Flags {
            path: Some(("-m".to_owned(), p.to_owned())),
            ..Flags::default()
        };
        assert_eq!(
            resolve(&m("/a"), None).expect("flag"),
            Some(Source::File("/a".into()))
        );
        assert_eq!(
            resolve(&m("/a"), Some("/a")).expect("same file"),
            Some(Source::File("/a".into()))
        );
        let e = resolve(&m("/a"), Some("/b")).expect_err("two files");
        assert!(
            e.to_string().contains("-m /a") && e.to_string().contains("BLOOMERY_REF_MODEL=/b"),
            "{e}"
        );
        assert_eq!(
            resolve(&m("/a"), Some("")).expect("empty env"),
            Some(Source::File("/a".into()))
        );
        let hf = Flags {
            hf: Some("a/b:Q8_0".to_owned()),
            ..Flags::default()
        };
        assert!(resolve(&hf, Some("/b")).is_err());
        assert_eq!(
            resolve(&hf, None).expect("hf"),
            Some(Source::Hf(RepoRef {
                repo: "a/b".into(),
                quant: Some("Q8_0".into())
            }))
        );
        assert!(matches!(
            resolve(
                &Flags {
                    hf: Some("bad".into()),
                    ..Flags::default()
                },
                None
            ),
            Err(HfError::Repo(_))
        ));
        assert_eq!(
            resolve(&Flags::default(), Some("/e")).expect("env"),
            Some(Source::File("/e".into()))
        );
        assert_eq!(resolve(&Flags::default(), None).expect("none"), None);
    }

    #[test]
    fn a_catalog_name_beside_a_model_source_only_chooses_the_seat() {
        let named = |target: &str| {
            Some(CatalogRef::parse(target).unwrap_or_else(|e| panic!("{target}: {e}")))
        };
        let file = |p: &str| Some(("-m".to_owned(), p.to_owned()));

        // Alone, a name is the catalog's: `select` hands it to the pick, `resolve` refuses it.
        let alone = Flags {
            name: named("qwen3.8-flash-next:q8_0"),
            ..Flags::default()
        };
        assert_eq!(
            select(&alone, None).expect("name"),
            alone.name.map(Selected::Catalog)
        );
        let e = resolve(&alone, None).expect_err("not a source").to_string();
        assert!(e.contains("qwen3.8-flash-next:Q8_0"), "{e}");

        // A bare name beside a path, a repo or the variable chooses the seat; the source stands.
        for (path, hf, env, source) in [
            (file("/a"), None, None, Source::File("/a".into())),
            (
                None,
                Some("x/y:Q4_K_M"),
                None,
                Source::Hf(RepoRef::parse("x/y:Q4_K_M").expect("repo")),
            ),
            (None, None, Some("/e"), Source::File("/e".into())),
        ] {
            let flags = Flags {
                path,
                hf: hf.map(str::to_owned),
                name: named("glm"),
            };
            assert_eq!(
                select(&flags, env).expect("seat only"),
                Some(Selected::Source(source))
            );
        }

        // A variant beside one of them names the model twice, with both spellings.
        for (path, hf, env, other) in [
            (file("/a"), None, None, "-m /a"),
            (None, Some("x/y"), None, "--hf x/y"),
            (None, None, Some("/e"), "BLOOMERY_REF_MODEL=/e"),
        ] {
            let flags = Flags {
                path,
                hf: hf.map(str::to_owned),
                name: named("glm:UD-Q4_K_XL"),
            };
            let e = select(&flags, env).expect_err("twice").to_string();
            assert!(
                e.contains("glm-5.3-flash:UD-Q4_K_XL") && e.contains(other),
                "{e}"
            );
        }
    }
}
