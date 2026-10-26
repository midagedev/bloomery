//! The dense Nanbeige4.3-33B family (`logits-nanbeige43`). The file is `llama` in the GGUF, so the family is a row of
//! that architecture's table with a model file of its own. A set is llama.cpp mainline's logits after a prompt and after
//! each id forced behind it, written from [`MODEL`] by the mainline tree [`super::LCPP_BUILD`] names:
//! `tools/ref/dense_logits_ref.cpp` through `tools/ref/dense-logits.py` (`just dump-logits-nanbeige43`,
//! `tools/ref/models/nanbeige43.sh`). The prompt is the first ids of the prose corpus under this vocabulary
//! (`$BLOOMERY_DATA/nanbeige43/corpus-prose.ids`) and the forced ids the ones after them. The set's format and its
//! reader are [`crate::arch::forced_logits`]'s; this module is the family's row and the reader aimed at it.

use super::{ARCH, LCPP_BUILD};
use crate::RefError;
use crate::arch::forced_logits::{self, Kind};
use crate::family::{Build, Family, Identity};

/// The file every `logits-nanbeige43` set is dumped from: bartowski's Nanbeige4.3-33B Q4_K_M
/// (`tools/ref/models/nanbeige43.sh`).
pub const MODEL: &str = "/models/Nanbeige4.3-33B/Nanbeige_Nanbeige4.3-33B-Q4_K_M.gguf";

/// The first 64 and 4,096 ids of the prose corpus as the prompt, each with the 32 ids after them forced.
pub const P64: &str = "ref_nanbeige43_logits_p64";
pub const P4096: &str = "ref_nanbeige43_logits_p4096";

/// The sets, shortest first.
pub const SETS: &[&str] = &[P64, P4096];

/// [`MODEL`], as a family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// llama.cpp mainline's logits rows of the 33B Q4_K_M file.
pub static LOGITS: Family = Family {
    name: "logits-nanbeige43",
    sets: SETS,
    resolve: None,
    recipe: "just dump-logits-nanbeige43",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &[],
};

/// [`LOGITS`], as the family [`ForcedLogits`] reads.
#[derive(Debug)]
pub enum Nanbeige43 {}

impl Kind for Nanbeige43 {
    fn family() -> &'static Family {
        &LOGITS
    }
}

/// A set of [`LOGITS`], opened and checked: the ids it was fed and its rows.
pub type ForcedLogits = forced_logits::ForcedLogits<Nanbeige43>;

#[cfg(test)]
mod tests {
    use super::{ForcedLogits, LOGITS, MODEL};
    use crate::RefError;
    use crate::arch::forced_logits::tests::write_set;
    use crate::arch::llama::{ARCH, LCPP_BUILD};

    /// The family takes a complete set of the 33B file, the pinned mainline commit and this architecture, and refuses
    /// by name another model file (one the same build runs), an ik build, another architecture,
    /// and a set without its trailer; the reader aimed at it opens that set and refuses the others by the same names.
    #[test]
    fn the_family_refuses_a_set_of_another_file_build_or_arch() -> Result<(), RefError> {
        let dir = write_set("nanbeige43-family", MODEL, LCPP_BUILD, ARCH, true);
        assert_eq!(LOGITS.check_set(&dir)?.dumped_from, MODEL);
        ForcedLogits::open(&dir)?;
        let cases = [
            (
                "/models/another/another-Q4_K_M.gguf",
                LCPP_BUILD,
                ARCH,
                true,
                "dumped from",
            ),
            (MODEL, "db517b69", ARCH, true, "build"),
            (MODEL, LCPP_BUILD, "qwen35", true, "arch"),
            (MODEL, LCPP_BUILD, ARCH, false, "no `# complete` trailer"),
        ];
        for (model, build, arch, complete, want) in cases {
            let dir = write_set("nanbeige43-family", model, build, arch, complete);
            let e = LOGITS.check_set(&dir).expect_err("refused").to_string();
            assert!(e.contains(want), "{model} {build} {arch} {complete}: {e}");
            let e = ForcedLogits::open(&dir).expect_err("refused").to_string();
            assert!(e.contains(want), "{model} {build} {arch} {complete}: {e}");
        }
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }
}
