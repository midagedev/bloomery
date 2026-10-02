//! Where a server's model file comes from: a path (`-m PATH`, `--model-file
//! PATH`, and whatever other spellings a binary gives the flag), a repo
//! (`--hf <repo>[:<quant>]`), or the gates' environment variable
//! `BLOOMERY_REF_MODEL`. A model is named once: a flag given twice, a path
//! beside `--hf`, and a flag beside the variable when the two name different
//! files are each refused by name with both values. A path flag equal to the
//! variable is the same file named twice and stands.

use crate::{HfError, RepoRef};

/// The model flags taken out of a binary's arguments.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Flags {
    /// The path flag's spelling and value.
    pub path: Option<(String, String)>,
    /// `--hf`'s value.
    pub hf: Option<String>,
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

/// The source `flags` and the variable's value `env` name; `None` when
/// neither names one. An empty `env` counts as unset.
pub fn resolve(flags: &Flags, env: Option<&str>) -> Result<Option<Source>, HfError> {
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

/// The refusal when nothing names the model.
pub const NONE: &str = "no model file: pass -m PATH (the first shard) or --hf <repo>[:<quant>], \
                        or set BLOOMERY_REF_MODEL";

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
            hf: None,
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
            path: None,
            hf: Some("a/b:Q8_0".to_owned()),
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
                    path: None,
                    hf: Some("bad".into())
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
}
