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
//! `/home/user/llama.cpp-19cce691` (mainline at `19cce6919`, no patch;
//! [`NEMOTRON_BUILD`]), whose `pixtral` pre is the TEKKEN regex with
//! `ignore_merges`; Nanbeige4.3-33B's file is SentencePiece (`llama`) and is
//! read by `/home/user/llama.cpp-mainline` ([`MAINLINE_BUILD`]), whose
//! `pixtral` is still LLAMA3. MiniMax-M3's file is read by `/home/user/llama.cpp-solar2`
//! (the Solar-Open2 patch on mainline `6ea215d17`; [`MINIMAXM3_BUILD`]).
//! Muse Spark Open's `gpt-4o` pre is this tree's `minimax-m2` regex under its
//! own name, and its set is read by `/home/user/llama.cpp-muse`
//! (mainline at PR 30616's merge, no patch; [`MUSESPARK_BUILD`], the same tree
//! the musespark node dumps are read by). Mistral Large 4's `tekken-num3` pre is the Tekken regex with
//! digit runs grouped by three from the right, and its set is read by `/home/user/llama.cpp-ml4`
//! (mainline at PR 31376's merge, no patch; [`MISTRALLARGE4_BUILD`], the same tree the
//! mistrallarge4 node dumps are read by). A family pins the tree its oracle is built from, the
//! `# build` line its set states after `crates/tokenizer/tools/oracle.sh` has
//! checked the tree (`tools/ref/emb2/oracle-tree.sh`): the upstream commit's
//! first 9 digits, and for a tree with a local patch `+` and the 12 digits
//! that name the patch (the tree's tracked files against that commit). The
//! executable's and the library's md5s change with every rebuild of one tree
//! and name no other machine's build, so no family pins them.

use crate::RefError;
use crate::family::{Build, Family, Identity};

/// The key of these families in the table of all of them.
pub const ARCH: &str = "tokenizer";

/// The vocabulary of the Qwen3-MoE file the engine runs, `$BLOOMERY_QWEN3MOE_VOCAB`
/// when set.
pub const QWEN3MOE_MODEL: &str = "/models/Qwen3-30B-A3B/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf";

/// The tree `/home/user/ik-tilde`: ik main at `dad2cb3e5` plus the one line
/// that puts `~` in the S class, whose `llama-tokenize` reads V4.1's
/// vocabulary (`oracle.sh` refuses a tree whose `~/` is not one id).
// PIN(2026-10-18): the commit and the patch, not one machine's llama-tokenize and libllama.so md5s, which no rebuild matched.
pub const TILDE_BUILD: &str = "dad2cb3e5+0e498d9dab2a";

/// The tree `/home/user/ik-tokref`: the ik-idxkey commit `db517b690` plus the
/// `unicode_tolower` fix the `(?i:'re)` contractions of the qwen2 and llama3
/// regexes need, whose `llama-tokenize` reads Qwen3-MoE, GLM-5.3-Flash and
/// Qwen3.8.
// PIN(2026-10-18): the commit and the patch, not one machine's llama-tokenize and libllama.so md5s, which no rebuild matched.
pub const TOKREF_BUILD: &str = "db517b690+29cabc43ed4d";

/// The tree `/home/user/llama.cpp-mainline`, whose `llama-tokenize` loads the
/// Nanbeige4.3 file: the mainline commit the Qwen3.5 families name for the same
/// tree, with no patch. It predates the mainline change that maps the GGUF pre
/// `pixtral` to TEKKEN, so it is no reference for [`NEMOTRON_BUILD`]'s file.
// PIN(2026-10-18): the commit, not one machine's llama-tokenize and libllama.so.0 md5s, which no rebuild matched.
pub(crate) const MAINLINE_BUILD: &str = super::qwen35::LCPP_BUILD;

/// The tree `/home/user/llama.cpp-19cce691` (the lfm3 logits oracle's tree, no
/// patch), whose `llama-tokenize` loads the Nemotron file (ik-tokref has no
/// `nemotron_h` architecture). Its `pixtral` pre is the TEKKEN type, which
/// `Pre::Tekken` follows; a set dumped by a tree that reads the name as LLAMA3
/// ([`MAINLINE_BUILD`]) is refused by its `# build` line.
// PIN(2026-10-29): the commit that holds the pixtral-to-TEKKEN mapping, not one machine's llama-tokenize and libllama.so.0 md5s.
pub(crate) const NEMOTRON_BUILD: &str = "19cce6919";

/// The tree `/home/user/llama.cpp-solar2`, whose `llama-tokenize` loads the MiniMax-M3 file: mainline `6ea215d17` plus the
/// Solar-Open2 patch ([`super::solar2::PAI_BUILD`] names the patch by its file's hash, this pin by the tree's tracked files).
/// Its UTF-8 decoder is the lenient one `tokenizer::unicode::decode_lenient` follows; the mainline tree the MiniMax-M3 node
/// dumps are read by (`cd0a5b6a1`) rejects an overlong form byte by byte, and its ids equal this tree's on every other text.
// PIN(2026-10-21): the commit and the patch, not one machine's llama-tokenize and libllama.so.0 md5s, which no rebuild matched.
pub(crate) const MINIMAXM3_BUILD: &str = "6ea215d17+1616f7f3d842";

/// The tree `/home/user/llama.cpp-muse`, whose `llama-tokenize` loads Muse Spark Open's file
/// (the merge of PR 30616, no patch): the commit the musespark node-dump family reads the same
/// tree at ([`super::musespark::COMMIT`]).
// PIN(2026-10-23): the commit, not one machine's llama-tokenize and libllama.so.0 md5s, which no rebuild matched.
pub const MUSESPARK_BUILD: &str = super::musespark::COMMIT;

/// The tree `/home/user/llama.cpp-ml4`, whose `llama-tokenize` loads Mistral Large 4's file
/// (the merge of PR 31376, no patch): the commit the mistrallarge4 node-dump family reads the
/// same tree at ([`super::mistrallarge4::COMMIT`]).
// PIN(2026-10-31): the commit, not one machine's llama-tokenize and libllama.so.0 md5s, which no rebuild matched.
pub const MISTRALLARGE4_BUILD: &str = super::mistrallarge4::COMMIT;

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

/// MiniMax-M3-UD-IQ4_XS's set (`minimax-m2` pre-tokenizer).
pub static MINIMAXM3: Family = Family {
    name: "tokenizer-minimaxm3",
    sets: &["ref-tokenizer-minimaxm3"],
    resolve: None,
    recipe: "just dump-ref-tokenizer minimaxm3",
    identity: Identity::Vocabulary,
    arch: None,
    build: Some(Build::Is(MINIMAXM3_BUILD)),
    runs: Some(minimaxm3_model),
    draft_runs: None,
    consumers: &["gate-tokenizer"],
};

/// Muse Spark Open's set (`gpt-4o` pre-tokenizer).
pub static MUSESPARK: Family = Family {
    name: "tokenizer-musespark",
    sets: &["ref-tokenizer-musespark"],
    resolve: None,
    recipe: "just dump-ref-tokenizer musespark",
    identity: Identity::Vocabulary,
    arch: None,
    build: Some(Build::Is(MUSESPARK_BUILD)),
    runs: Some(musespark_model),
    draft_runs: None,
    consumers: &["gate-tokenizer"],
};

/// Mistral Large 4's set (`tekken-num3` pre-tokenizer, BOS `<s>` added by the file, so the set is dumped without it).
pub static MISTRALLARGE4: Family = Family {
    name: "tokenizer-mistrallarge4",
    sets: &["ref-tokenizer-mistrallarge4"],
    resolve: None,
    recipe: "just dump-ref-tokenizer mistrallarge4",
    identity: Identity::Vocabulary,
    arch: None,
    build: Some(Build::Is(MISTRALLARGE4_BUILD)),
    runs: Some(mistrallarge4_model),
    draft_runs: None,
    consumers: &["gate-tokenizer"],
};

/// Nanbeige4.3-33B's set (`llama` SentencePiece, byte fallback, BOS `<|im_start|>`): the ids keep the BOS that
/// `llama-tokenize` adds.
pub static NANBEIGE43: Family = Family {
    name: "tokenizer-nanbeige43",
    sets: &["ref-tokenizer-nanbeige43"],
    resolve: None,
    recipe: "just dump-ref-tokenizer nanbeige43",
    identity: Identity::Vocabulary,
    arch: None,
    build: Some(Build::Is(MAINLINE_BUILD)),
    runs: Some(nanbeige43_model),
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

fn minimaxm3_model() -> Result<String, RefError> {
    Ok(super::minimaxm3::MODEL.to_string())
}

fn musespark_model() -> Result<String, RefError> {
    Ok(super::musespark::MODEL.to_string())
}

fn mistrallarge4_model() -> Result<String, RefError> {
    Ok(super::mistrallarge4::MODEL.to_string())
}

fn nanbeige43_model() -> Result<String, RefError> {
    Ok(super::llama::dense::MODEL.to_string())
}

/// The families, in the order `refset-check` lists them.
pub static FAMILIES: &[&Family] = &[
    &V41,
    &QWEN3MOE,
    &GLM5NEXT,
    &QWEN38,
    &NEMOTRON,
    &MINIMAXM3,
    &MUSESPARK,
    &MISTRALLARGE4,
    &NANBEIGE43,
    &super::gemma_embedding::TOKENIZER,
];

#[cfg(test)]
mod tests {
    use super::{
        ARCH, FAMILIES, GLM5NEXT, MAINLINE_BUILD, MINIMAXM3, MINIMAXM3_BUILD, MISTRALLARGE4,
        MISTRALLARGE4_BUILD, MUSESPARK, MUSESPARK_BUILD, NANBEIGE43, NEMOTRON, NEMOTRON_BUILD,
        QWEN3MOE, QWEN38, TILDE_BUILD, TOKREF_BUILD, V41,
    };
    use crate::RefError;
    use crate::emb2::LCPP_BUILD;
    use crate::family::{Build, Family};
    use crate::tokenizer::tests::{CASES_MD5, RIGHT, Stated, set_dir, write_set};
    use std::path::Path;

    /// A tree pin's two parts: the commit's 9 digits and, for a tree with a local patch, `+` and the patch's 12.
    fn parts(pin: &str) -> (&str, Option<&str>) {
        match pin.split_once('+') {
            Some((commit, patch)) => (commit, Some(patch)),
            None => (pin, None),
        }
    }

    fn is_hex(s: &str, digits: usize) -> bool {
        s.len() == digits && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    }

    /// `s` with its last digit changed: the same pin of another commit or patch.
    fn flip(s: &str) -> String {
        let (head, last) = s.split_at(s.len() - 1);
        format!("{head}{}", if last == "0" { "1" } else { "0" })
    }

    /// The pin a family names.
    fn pin(f: &Family) -> &'static str {
        let Some(Build::Is(pin)) = f.build else {
            panic!("{} pins no build", f.name);
        };
        pin
    }

    /// Write a set into `dir` as `f`'s dumper would for the vocabulary `file`,
    /// stating the tree `build` its `# build` line names, or, with `None`, the
    /// md5s of an executable and a library only: a set dumped before the trees were pinned.
    fn write(dir: &Path, file: &str, build: Option<&str>) {
        let stated = Stated {
            commit: build,
            vocabulary: file,
            ..RIGHT
        };
        write_set(dir, &stated, CASES_MD5);
    }

    fn check(f: &Family, what: &str, file: &str, build: Option<&str>) -> Result<(), RefError> {
        let dir = set_dir(what);
        write(&dir, file, build);
        let r = f.check_set(&dir).map(|_| ());
        std::fs::remove_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        r
    }

    /// The table holds the vocabularies under one key, each its own row with
    /// its own set directory and the recipe that writes it, and each pins the
    /// tree its oracle is built from: the commit's 9 digits, and for the two
    /// ik side trees that carry local commits, the patch's 12. The trees
    /// without a patch are mainline's, whose commit the pin states as it is.
    #[test]
    fn the_vocabularies_are_rows_of_the_table() {
        assert_eq!(crate::arch::families(ARCH).len(), 10);
        for f in FAMILIES {
            assert!(std::ptr::eq(crate::arch::named(f.name).unwrap_or(&V41), *f));
            assert!(
                f.recipe.starts_with("just dump-ref-tokenizer "),
                "{}",
                f.recipe
            );
            assert_eq!(f.sets.len(), 1);
            assert_eq!(f.path(f.sets[0]), crate::data_dir().join(f.sets[0]));
            let (commit, patch) = parts(pin(f));
            assert!(is_hex(commit, 9), "{}: commit {commit}", f.name);
            assert!(patch.is_none_or(|p| is_hex(p, 12)), "{}: {patch:?}", f.name);
        }
        for (f, want, patched) in [
            (&V41, TILDE_BUILD, true),
            (&QWEN3MOE, TOKREF_BUILD, true),
            (&GLM5NEXT, TOKREF_BUILD, true),
            (&QWEN38, TOKREF_BUILD, true),
            (&NEMOTRON, NEMOTRON_BUILD, false),
            (&MINIMAXM3, MINIMAXM3_BUILD, true),
            (&MUSESPARK, MUSESPARK_BUILD, false),
            (&MISTRALLARGE4, MISTRALLARGE4_BUILD, false),
            (&NANBEIGE43, MAINLINE_BUILD, false),
            (&super::super::gemma_embedding::TOKENIZER, LCPP_BUILD, false),
        ] {
            assert_eq!(pin(f), want, "{} pins the wrong tree", f.name);
            assert_eq!(parts(want).1.is_some(), patched, "{}", f.name);
        }
        // Nanbeige4.3's tree is the one the Qwen3.5 families name, and no other family's. Nemotron's is a later
        // one: the commit that gives `pixtral` the TEKKEN type the engine follows.
        assert_eq!(MAINLINE_BUILD, crate::arch::qwen35::LCPP_BUILD);
        assert_ne!(parts(NEMOTRON_BUILD).0, parts(MAINLINE_BUILD).0);
        assert_ne!(parts(TILDE_BUILD).0, parts(TOKREF_BUILD).0);
        let sets: Vec<_> = FAMILIES.iter().map(|f| f.sets[0]).collect();
        assert_eq!(
            sets,
            [
                "ref-tokenizer-v41",
                "ref-tokenizer-qwen3moe",
                "ref-tokenizer-glm5next",
                "ref-tokenizer-qwen38",
                "ref-tokenizer-nemotron",
                "ref-tokenizer-minimaxm3",
                "ref-tokenizer-musespark",
                "ref-tokenizer-mistrallarge4",
                "ref-tokenizer-nanbeige43",
                "ref-tokenizer-llama",
            ]
        );
    }

    /// Each family takes a set dumped from its own vocabulary file by the tree
    /// it pins, and no other: another commit, the same commit with another
    /// patch, a patched tree's commit without its patch, an unpatched tree's
    /// commit with one, another family's tree, and the md5s of a binary that
    /// the set states without a `# build` line (the sets dumped before the
    /// trees were pinned) are each refused by name, with what the set stated
    /// and what the family wants.
    #[test]
    fn each_family_takes_its_own_tree_and_vocabulary_only() -> Result<(), RefError> {
        for f in FAMILIES {
            let runs = f.runs()?;
            let own = pin(f);
            let (commit, patch) = parts(own);
            check(f, &format!("{}-own", f.name), &runs, Some(own))?;
            let mut foreign = vec![flip(commit), format!("{commit}x")];
            match patch {
                Some(patch) => {
                    foreign.push(format!("{}+{patch}", flip(commit)));
                    foreign.push(format!("{commit}+{}", flip(patch)));
                    foreign.push(commit.to_string());
                }
                None => foreign.push(format!("{commit}+000000000000")),
            }
            foreign.extend(
                FAMILIES
                    .iter()
                    .map(|g| pin(g).to_string())
                    .filter(|p| p != own),
            );
            for other in foreign {
                match check(f, &format!("{}-other", f.name), &runs, Some(&other)) {
                    Err(RefError::Foreign {
                        field: "build",
                        got,
                        want,
                        ..
                    }) => assert_eq!((got.as_str(), want.as_str()), (other.as_str(), own)),
                    r => panic!("{}: a set of the tree {other}: {r:?}", f.name),
                }
            }
            let md5s = crate::tokenizer::build_id(RIGHT.binary, RIGHT.library);
            match check(f, &format!("{}-md5", f.name), &runs, None) {
                Err(RefError::Foreign {
                    field: "build",
                    got,
                    ..
                }) => assert_eq!(got, md5s),
                r => panic!("{}: a set stating md5s only: {r:?}", f.name),
            }
        }
        Ok(())
    }

    /// A set of another vocabulary's file is stale under every family, though its
    /// tree is the family's: the files differ.
    #[test]
    fn a_set_of_another_vocabulary_file_is_stale_under_each_family() -> Result<(), RefError> {
        let files: Vec<String> = FAMILIES
            .iter()
            .map(|f| f.runs())
            .collect::<Result<_, _>>()?;
        for (f, own) in FAMILIES.iter().zip(&files) {
            for file in files.iter().filter(|x| *x != own) {
                match check(f, &format!("{}-vocab", f.name), file, Some(pin(f))) {
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
