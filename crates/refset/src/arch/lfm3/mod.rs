//! The LFM3-1.7B families (`lfm3`). A logits set is dumped from one GGUF file by mainline llama.cpp at `19cce6919`
//! ([`BUILD`]), the commit that added the architecture, on the CPU with CUDA hidden (`just dump-ref-lfm3`,
//! `tools/ref/dump-lfm3.sh`): one family a file, so a set of one file is refused by name where another's is read.
//! The dump script, the harness (`tools/ref/nemotron_ref.cpp`) and the reader ([`crate::nemotron::LogitsSet`]) are the
//! Nemotron sets' own: the manifest is the same, and a last-position logits row is the same fact whatever the architecture.
//! ik has no `lfm3`. The files and their sets are the rows of `LOGITS_SETS` in `tools/ref/models/lfm3.sh`.

use crate::RefError;
use crate::family::{Build, Family, Identity};

/// The architecture every lfm3 manifest names in its `# arch` line.
pub const ARCH: &str = "lfm3";

/// The Q4_K_M file, the headline one.
pub const MODEL: &str = "/models/LFM3-1.7B-GGUF/LFM3-1.7B-Q4_K_M.gguf";

/// The Q8_0 file: the Q8_0 sites' reference.
pub const MODEL_Q8: &str = "/models/LFM3-1.7B-GGUF/LFM3-1.7B-Q8_0.gguf";

/// The BF16 file: every linear layer unquantized.
pub const MODEL_BF16: &str = "/models/LFM3-1.7B-GGUF/LFM3-1.7B-BF16.gguf";

/// Last-position logits of each prompt from [`MODEL`].
pub const LOGITS_SET: &str = "ref-lfm3-logits";

/// Last-position logits of each prompt from [`MODEL_Q8`].
pub const LOGITS_Q8_SET: &str = "ref-lfm3-q8-logits";

/// Last-position logits of each prompt from [`MODEL_BF16`].
pub const LOGITS_BF16_SET: &str = "ref-lfm3-bf16-logits";

/// The executable and library of `/home/user/llama.cpp-19cce691` that `nemotron_ref_lfm3` is linked against, spelled as
/// [`crate::tokenizer::build_id`]. The two words after `llama-tokenize` and `libllama.so` are the md5s that
/// `just build-ref-lfm3` prints on its `build` line; a set dumped by another build is refused by name.
const BUILD: &str =
    "llama-tokenize 3a2f117f9db7b3195ccbb64ada4c4f24 libllama.so 2a6f0050bbd0280f604182bbc881da6a";

/// [`MODEL`], as a family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// [`MODEL_Q8`], as a family's `runs`.
fn model_q8() -> Result<String, RefError> {
    Ok(MODEL_Q8.to_string())
}

/// [`MODEL_BF16`], as a family's `runs`.
fn model_bf16() -> Result<String, RefError> {
    Ok(MODEL_BF16.to_string())
}

/// Mainline's last-position logits from the Q4_K_M file (`just dump-ref-lfm3`).
pub static LOGITS: Family = Family {
    name: "ref-lfm3-logits",
    sets: &[LOGITS_SET],
    resolve: None,
    recipe: "just dump-ref-lfm3",
    identity: Identity::NemotronLogits,
    arch: Some(ARCH),
    build: Some(Build::Is(BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &[],
};

/// Mainline's last-position logits from the Q8_0 file (`just dump-ref-lfm3`).
pub static LOGITS_Q8: Family = Family {
    name: "ref-lfm3-q8-logits",
    sets: &[LOGITS_Q8_SET],
    resolve: None,
    recipe: "just dump-ref-lfm3",
    identity: Identity::NemotronLogits,
    arch: Some(ARCH),
    build: Some(Build::Is(BUILD)),
    runs: Some(model_q8),
    draft_runs: None,
    consumers: &[],
};

/// Mainline's last-position logits from the BF16 file (`just dump-ref-lfm3`).
pub static LOGITS_BF16: Family = Family {
    name: "ref-lfm3-bf16-logits",
    sets: &[LOGITS_BF16_SET],
    resolve: None,
    recipe: "just dump-ref-lfm3",
    identity: Identity::NemotronLogits,
    arch: Some(ARCH),
    build: Some(Build::Is(BUILD)),
    runs: Some(model_bf16),
    draft_runs: None,
    consumers: &[],
};

/// The architecture's families, in the order `refset-check` lists them.
pub static FAMILIES: &[&Family] = &[&LOGITS, &LOGITS_Q8, &LOGITS_BF16];

#[cfg(test)]
mod tests {
    use super::{
        ARCH, BUILD, FAMILIES, LOGITS, LOGITS_BF16, LOGITS_BF16_SET, LOGITS_Q8, LOGITS_Q8_SET,
        LOGITS_SET, MODEL, MODEL_BF16, MODEL_Q8,
    };
    use crate::family::{Build, Identity};

    /// The tools profile the sets and the files come from.
    const PROFILE: &str = include_str!("../../../../../tools/ref/models/lfm3.sh");

    /// The build pin is spelled as `build_id` spells the dumper's two md5s, and the families share it.
    #[test]
    fn the_build_pin_is_spelled_as_the_dumper_records_it() {
        let mut words = BUILD.split_whitespace();
        let (binary, library) = (words.nth(1).unwrap_or(""), words.nth(1).unwrap_or(""));
        assert_eq!(BUILD, crate::tokenizer::build_id(binary, library));
        for fam in FAMILIES {
            assert_eq!(fam.build, Some(Build::Is(BUILD)), "{}", fam.name);
        }
    }

    /// One family a file, each with a set of its own, the same architecture and manifest identity as the
    /// Nemotron sets (one reader), and the file and set the profile's table names together.
    #[test]
    fn each_family_is_one_file_and_one_set_of_the_profile() {
        let rows = [
            (&LOGITS, LOGITS_SET, MODEL),
            (&LOGITS_Q8, LOGITS_Q8_SET, MODEL_Q8),
            (&LOGITS_BF16, LOGITS_BF16_SET, MODEL_BF16),
        ];
        assert_eq!(FAMILIES.len(), rows.len());
        for (i, (fam, set, file)) in rows.into_iter().enumerate() {
            assert!(std::ptr::eq(FAMILIES[i], fam), "{set}: order");
            assert_eq!(fam.name, set);
            assert_eq!(fam.sets, &[set]);
            assert_eq!(fam.runs().unwrap_or_else(|e| panic!("{e}")), file);
            assert_eq!(fam.identity, Identity::NemotronLogits);
            assert_eq!(fam.arch, Some(ARCH));
            assert!(fam.resolve.is_none(), "{set}: one file, no per-file suffix");
            assert!(
                PROFILE.contains(&format!("\"{set}:{file}:")),
                "{set}: the profile's LOGITS_SETS has no {file} for it"
            );
        }
        assert!(
            PROFILE.contains(&format!("MODEL=${{BLOOMERY_REF_MODEL:-{MODEL}}}")),
            "the profile's MODEL is not the headline file"
        );
    }

    /// The recipe that builds the oracle tree builds the commit and the tree the profile names: the pin has one owner,
    /// the profile, and the recipe's environment is its copy.
    #[test]
    fn the_tree_recipe_builds_the_profiles_commit_in_the_profiles_tree() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../justfile");
        let justfile =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let commit = PROFILE
            .lines()
            .find_map(|l| l.strip_prefix("LCPP_COMMIT="))
            .unwrap_or_else(|| panic!("the profile pins no LCPP_COMMIT"));
        let tree = PROFILE
            .lines()
            .find_map(|l| l.strip_prefix(": \"${LCPP:="))
            .and_then(|l| l.strip_suffix("}\""))
            .unwrap_or_else(|| panic!("the profile names no LCPP"));
        assert!(
            justfile.contains(&format!("LCPP_BUILD_PIN={commit} ")),
            "build-lfm3-lcpp does not build {commit}"
        );
        assert!(
            justfile.contains(&format!("LCPP_BUILD_TREE={tree} ")),
            "build-lfm3-lcpp does not build in {tree}"
        );
    }
}
