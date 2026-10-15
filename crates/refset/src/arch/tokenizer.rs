//! The tokenizer families: one per vocabulary the engine runs, each set the
//! ids `llama-tokenize` gave the corpus texts and the hand-picked cases
//! (`crates/tokenizer/tools/oracle.sh`, `just dump-ref-tokenizer`). A
//! vocabulary is not an architecture; the table keys these families by
//! [`ARCH`], and each one's `runs` is the vocabulary file the tokenizer
//! gate loads, so a set and the tokenizer under test come from one file. The
//! sets' directories are `ref-tokenizer-*`: the `tokenizer*` directories are
//! the format before the trailer, which a checkout that still runs that gate
//! rewrites.
//!
//! V4.1's set is read by the tree patched so `~` is a symbol
//! (`/home/user/ik-tilde`); Qwen3-MoE, GLM-5.3-Flash and Qwen3.8 by the tree
//! with the tolower fix (`/home/user/ik-tokref`). Nemotron's file is
//! `nemotron_h`, which that tree refuses, so its set is read by
//! `/home/user/llama.cpp-mainline`, whose `pixtral` pre is the LLAMA3 regex
//! with `ignore_merges`. A family pins the executable and the library it
//! loads by md5 ([`crate::tokenizer::build_id`]): the tokenizer is the
//! library's code, the executable only calls it.

use crate::RefError;
use crate::family::{Build, Family, Identity};

/// The key of these families in the table of all of them.
pub const ARCH: &str = "tokenizer";

/// The vocabulary of the Qwen3-MoE file the engine runs, `$BLOOMERY_QWEN3MOE_VOCAB`
/// when set.
pub const QWEN3MOE_MODEL: &str = "/models/Qwen3-30B-A3B/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf";

/// The executable and library of `/home/user/ik-tilde`: ik with `~` in the S
/// class, whose `llama-tokenize` reads V4.1's vocabulary (`oracle.sh` refuses
/// a tree whose `~/` is not one id).
// PIN(2026-10-08): /home/user/ik-tilde's llama-tokenize and libllama.so.
pub const TILDE_BUILD: &str = "llama-tokenize d03083b01a24f6a9e5c2190abf7d38c6 \
                               libllama.so f855ad0b949f0b7d871973be1fcc419f";

/// The executable and library of `/home/user/ik-tokref`: the ik-idxkey commit
/// plus the `unicode_tolower` fix the `(?i:'re)` contractions of the qwen2 and
/// llama3 regexes need, whose `llama-tokenize` reads Qwen3-MoE, GLM-5.3-Flash
/// and Qwen3.8.
// PIN(2026-10-08): /home/user/ik-tokref's llama-tokenize and libllama.so.
pub const TOKREF_BUILD: &str = "llama-tokenize 0128e83c5e71e5f2b72f3d5d2c2de6b2 \
                                libllama.so 1674b0181160aba8599f5750cba1b30d";

/// The executable and library of `/home/user/llama.cpp-mainline`, whose
/// `llama-tokenize` loads the Nemotron file. ik-tokref does not: it has no
/// `nemotron_h` architecture. The soname is `libllama.so.0`; the pin's word
/// is the one [`crate::tokenizer::build_id`] spells.
// PIN(2026-10-14): /home/user/llama.cpp-mainline's llama-tokenize and libllama.so.0.
pub(crate) const NEMOTRON_BUILD: &str = "llama-tokenize 5df95d3a942effc971ff170432d840fd \
                                  libllama.so 3575b1f39e7773379fec011d39f32a77";

/// V4.1's set (`deepseek-v3` pre-tokenizer), from the file the tree runs.
pub static V41: Family = Family {
    name: "tokenizer-v41",
    sets: &["ref-tokenizer-v41"],
    resolve: None,
    recipe: "just dump-ref-tokenizer v41",
    identity: Identity::Vocabulary,
    arch: None,
    build: Some(Build::Is(TILDE_BUILD)),
    runs: Some(super::deepseek41::model),
    draft_runs: None,
    consumers: &["gate-tokenizer"],
};

/// The Qwen3-MoE file's set (`qwen2` pre-tokenizer).
pub static QWEN3MOE: Family = Family {
    name: "tokenizer-qwen3moe",
    sets: &["ref-tokenizer-qwen3moe"],
    resolve: None,
    recipe: "just dump-ref-tokenizer qwen3moe",
    identity: Identity::Vocabulary,
    arch: None,
    build: Some(Build::Is(TOKREF_BUILD)),
    runs: Some(qwen3moe_model),
    draft_runs: None,
    consumers: &["gate-tokenizer"],
};

/// GLM-5.3-Flash's set (`glm4` pre-tokenizer).
pub static GLM5NEXT: Family = Family {
    name: "tokenizer-glm5next",
    sets: &["ref-tokenizer-glm5next"],
    resolve: None,
    recipe: "just dump-ref-tokenizer glm5next",
    identity: Identity::Vocabulary,
    arch: None,
    build: Some(Build::Is(TOKREF_BUILD)),
    runs: Some(glm5next_model),
    draft_runs: None,
    consumers: &["gate-tokenizer"],
};

/// Qwen3.8-Flash-Next's set (`qwen35` pre-tokenizer).
pub static QWEN38: Family = Family {
    name: "tokenizer-qwen38",
    sets: &["ref-tokenizer-qwen38"],
    resolve: None,
    recipe: "just dump-ref-tokenizer qwen38",
    identity: Identity::Vocabulary,
    arch: None,
    build: Some(Build::Is(TOKREF_BUILD)),
    runs: Some(qwen38_model),
    draft_runs: None,
    consumers: &["gate-tokenizer"],
};

/// The Nemotron 3 Nano 4B file the tokenizer gate loads (`pixtral`).
const NEMOTRON_MODEL: &str =
    "/models/NVIDIA-Nemotron-3-Nano-4B-GGUF/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf";

/// Nemotron 3 Nano 4B's set (`pixtral` pre-tokenizer).
pub static NEMOTRON: Family = Family {
    name: "tokenizer-nemotron",
    sets: &["ref-tokenizer-nemotron"],
    resolve: None,
    recipe: "just dump-ref-tokenizer nemotron",
    identity: Identity::Vocabulary,
    arch: None,
    build: Some(Build::Is(NEMOTRON_BUILD)),
    runs: Some(nemotron_model),
    draft_runs: None,
    consumers: &["gate-tokenizer"],
};

/// The Qwen3-MoE vocabulary file: `$BLOOMERY_QWEN3MOE_VOCAB`, else
/// [`QWEN3MOE_MODEL`]. An empty value counts as unset.
fn qwen3moe_model() -> Result<String, RefError> {
    Ok(match std::env::var("BLOOMERY_QWEN3MOE_VOCAB") {
        Ok(p) if !p.is_empty() => p,
        _ => QWEN3MOE_MODEL.to_string(),
    })
}

fn glm5next_model() -> Result<String, RefError> {
    Ok(super::glm5next::MODEL.to_string())
}

fn qwen38_model() -> Result<String, RefError> {
    Ok(super::qwen4exp::MODEL.to_string())
}

pub(crate) fn nemotron_model() -> Result<String, RefError> {
    Ok(NEMOTRON_MODEL.to_string())
}

/// The families, in the order `refset-check` lists them.
pub static FAMILIES: &[&Family] = &[&V41, &QWEN3MOE, &GLM5NEXT, &QWEN38, &NEMOTRON];

#[cfg(test)]
mod tests {
    use super::{
        ARCH, FAMILIES, GLM5NEXT, NEMOTRON, NEMOTRON_BUILD, QWEN3MOE, QWEN38, TILDE_BUILD,
        TOKREF_BUILD, V41,
    };
    use crate::RefError;
    use crate::family::Family;
    use crate::md5::check_hex;
    use crate::tokenizer::tests::{CASES_MD5, RIGHT, Stated, set_dir, write_set};
    use std::path::Path;

    /// The executable's and the library's md5 a pin names.
    fn md5s(pin: &str) -> (&str, &str) {
        let mut words = pin.split_whitespace();
        (words.nth(1).unwrap_or(""), words.nth(1).unwrap_or(""))
    }

    /// Write a set into `dir` as `f`'s dumper would for the vocabulary `file`,
    /// stating the pin `build`.
    fn write(dir: &Path, file: &str, build: &str) {
        let (binary, library) = md5s(build);
        write_set(
            dir,
            &Stated {
                binary,
                library,
                vocabulary: file,
                ..RIGHT
            },
            CASES_MD5,
        );
    }

    fn check(f: &Family, what: &str, file: &str, build: &str) -> Result<(), RefError> {
        let dir = set_dir(what);
        write(&dir, file, build);
        let r = f.check_set(&dir).map(|_| ());
        std::fs::remove_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        r
    }

    /// The table holds the vocabularies under one key, each its own row with
    /// its own set directory and the recipe that writes it; a pin is two md5s
    /// the reader's `build_id` spells.
    #[test]
    fn the_vocabularies_are_rows_of_the_table() {
        assert_eq!(crate::arch::families(ARCH).len(), 5);
        for f in FAMILIES {
            assert!(std::ptr::eq(crate::arch::named(f.name).unwrap_or(&V41), *f));
            assert!(
                f.recipe.starts_with("just dump-ref-tokenizer "),
                "{}",
                f.recipe
            );
            assert_eq!(f.sets.len(), 1);
            assert_eq!(f.path(f.sets[0]), crate::data_dir().join(f.sets[0]));
            let Some(crate::family::Build::Is(pin)) = f.build else {
                panic!("{} pins no build", f.name);
            };
            let (binary, library) = md5s(pin);
            assert_eq!(pin, crate::tokenizer::build_id(binary, library));
            for md5 in [binary, library] {
                check_hex(f.name, md5, &f.name).unwrap_or_else(|e| panic!("{e}"));
            }
        }
        for (f, pin) in [
            (&V41, TILDE_BUILD),
            (&QWEN3MOE, TOKREF_BUILD),
            (&GLM5NEXT, TOKREF_BUILD),
            (&QWEN38, TOKREF_BUILD),
            (&NEMOTRON, NEMOTRON_BUILD),
        ] {
            assert!(
                matches!(f.build, Some(crate::family::Build::Is(p)) if p == pin),
                "{} pins the wrong tree",
                f.name
            );
        }
        let sets: Vec<_> = FAMILIES.iter().map(|f| f.sets[0]).collect();
        assert_eq!(
            sets,
            [
                "ref-tokenizer-v41",
                "ref-tokenizer-qwen3moe",
                "ref-tokenizer-glm5next",
                "ref-tokenizer-qwen38",
                "ref-tokenizer-nemotron",
            ]
        );
    }

    /// Each family takes a set dumped from its own vocabulary file with the
    /// executable and library it pins, and no other family's pin: V4.1's set
    /// read by the tolower-fixed tree is as foreign as a Qwen set read by the
    /// tilde tree.
    #[test]
    fn each_family_takes_its_own_build_and_vocabulary_only() -> Result<(), RefError> {
        for f in FAMILIES {
            let runs = f.runs()?;
            let Some(crate::family::Build::Is(own)) = f.build else {
                panic!("{} pins no build", f.name);
            };
            let other = if own == TILDE_BUILD {
                TOKREF_BUILD
            } else {
                TILDE_BUILD
            };
            check(f, &format!("{}-own", f.name), &runs, own)?;
            match check(f, &format!("{}-other", f.name), &runs, other) {
                Err(RefError::Foreign { field: "build", .. }) => {}
                r => panic!("{}: a set of the other tree: {r:?}", f.name),
            }
        }
        Ok(())
    }

    /// A set of another vocabulary's file is stale under every family, though its
    /// executable and library are the family's: the files differ.
    #[test]
    fn a_set_of_another_vocabulary_file_is_stale_under_each_family() -> Result<(), RefError> {
        let files: Vec<String> = FAMILIES
            .iter()
            .map(|f| f.runs())
            .collect::<Result<_, _>>()?;
        for (f, own) in FAMILIES.iter().zip(&files) {
            let Some(crate::family::Build::Is(pin)) = f.build else {
                panic!("{} pins no build", f.name);
            };
            for file in files.iter().filter(|x| *x != own) {
                match check(f, &format!("{}-vocab", f.name), file, pin) {
                    Err(RefError::Stale {
                        dumped_from, runs, ..
                    }) => {
                        assert_eq!((&dumped_from, &runs), (file, own));
                    }
                    r => panic!("{} given {file}: {r:?}", f.name),
                }
            }
        }
        Ok(())
    }
}
