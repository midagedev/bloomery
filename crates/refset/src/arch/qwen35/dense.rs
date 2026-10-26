//! The dense Qwen3.8-27B family (`logits-qwen38d`). The file is `qwen35` in the GGUF, the dense architecture Clef's
//! backbone shares, so the family is one more row of that architecture's table with a model file of its own. A set is
//! llama.cpp mainline's logits after a prompt and after each id forced behind it, written from [`MODEL`] by the mainline
//! tree [`super::LCPP_BUILD`] names: `tools/ref/dense_logits_ref.cpp` through `tools/ref/dense-logits.py`
//! (`just dump-logits-qwen38d`, `tools/ref/models/qwen38d.sh`). The prompt is the first ids of the qwen35 prose corpus
//! and the forced ids the ones after them. The set's format and its reader are
//! [`crate::arch::forced_logits`]'s; this module is the family's row and the reader aimed at it.

use super::{ARCH, LCPP_BUILD};
use crate::RefError;
use crate::arch::forced_logits::{self, Kind};
use crate::family::{Build, Family, Identity};

/// The file every `logits-qwen38d` set is dumped from: lmstudio-community's Qwen3.8-27B Q4_K_M
/// (`tools/ref/models/qwen38d.sh`).
pub const MODEL: &str = "/models/Qwen3.8-27B/Qwen3.8-27B-Q4_K_M.gguf";

/// The first 64 and 4,096 ids of the prose corpus as the prompt, each with the 32 ids after them forced.
pub const P64: &str = "ref_qwen38d_logits_p64";
pub const P4096: &str = "ref_qwen38d_logits_p4096";

/// The sets, shortest first.
pub const SETS: &[&str] = &[P64, P4096];

/// [`MODEL`], as a family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// llama.cpp mainline's logits rows of the 27B Q4_K_M file.
pub static LOGITS: Family = Family {
    name: "logits-qwen38d",
    sets: SETS,
    resolve: None,
    recipe: "just dump-logits-qwen38d",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &[],
};

/// [`LOGITS`], as the family [`ForcedLogits`] reads.
#[derive(Debug)]
pub enum Qwen38d {}

impl Kind for Qwen38d {
    fn family() -> &'static Family {
        &LOGITS
    }
}

/// A set of [`LOGITS`], opened and checked: the ids it was fed and its rows.
pub type ForcedLogits = forced_logits::ForcedLogits<Qwen38d>;

#[cfg(test)]
mod tests {
    use super::{ForcedLogits, LOGITS, MODEL};
    use crate::RefError;
    use crate::arch::forced_logits::tests::write_set;
    use crate::arch::qwen35::{ARCH, HIDDEN, LCPP_BUILD};

    /// The family takes a complete set of the 27B file, the pinned mainline commit and this architecture, and refuses
    /// by name another model file (Clef's, which shares the architecture and the build), an ik build, the MoE
    /// architecture, and a set without its trailer; the reader aimed at it opens that set and refuses Clef's.
    #[test]
    fn the_family_refuses_a_set_of_another_file_build_or_arch() -> Result<(), RefError> {
        let dir = write_set("qwen38d-family", MODEL, LCPP_BUILD, ARCH, true);
        assert_eq!(LOGITS.check_set(&dir)?.dumped_from, MODEL);
        ForcedLogits::open(&dir)?;
        let clef = crate::arch::qwen35::MODEL;
        let cases = [
            (clef, LCPP_BUILD, ARCH, true, "dumped from"),
            (MODEL, "db517b69", ARCH, true, "build"),
            (MODEL, LCPP_BUILD, "qwen35moe", true, "arch"),
            (MODEL, LCPP_BUILD, ARCH, false, "no `# complete` trailer"),
        ];
        for (model, build, arch, complete, want) in cases {
            let dir = write_set("qwen38d-family", model, build, arch, complete);
            let e = LOGITS.check_set(&dir).expect_err("refused").to_string();
            assert!(e.contains(want), "{model} {build} {arch} {complete}: {e}");
            let e = ForcedLogits::open(&dir).expect_err("refused").to_string();
            assert!(e.contains(want), "{model} {build} {arch} {complete}: {e}");
        }
        // One family a file: the 27B's hidden-state family refuses this set, and this family Clef's.
        let dir = write_set("qwen38d-family", MODEL, LCPP_BUILD, ARCH, true);
        let e = HIDDEN.check_set(&dir).expect_err("refused").to_string();
        assert!(e.contains("dumped from"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }
}
