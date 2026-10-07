//! The Qwen3.5 dense (Clef backbone) families. The hidden-state sets are
//! llama.cpp mainline's `result_norm` of every position, written by
//! `tools/ref/hidden_ref.cpp` in the node dumps' set format by the mainline
//! tree [`LCPP_BUILD`] (`tools/ref/models/qwen35.sh`): [`HIDDEN`] from the
//! 27B Q4_K_M file [`MODEL`], [`HIDDEN_FLASH_Q8`] from Clef-Flash's published
//! Q8_0 file [`MODEL_FLASH_Q8`], [`HIDDEN_LEV`] from lev's Q4_K_M file
//! [`MODEL_LEV`] (Qwen3.5-4B, its head tied to the token embedding) — one
//! family a file, so a set of one is refused by name where another is read.

use crate::family::{Build, Family, Identity};

/// The file every qwen35 set is dumped from and the tree runs: Clef's text
/// backbone through mainline's converter and `llama-quantize` Q4_K_M.
pub const MODEL: &str = "/models/clef-27b/clef-27b-Q4_K_M.gguf";

/// The mainline commit every qwen35 set names in its `# build` line.
// PIN(2026-10-02): /home/user/llama.cpp-mainline at its HEAD, the tree that
// converted and quantized MODEL and that hidden_ref links against.
pub const LCPP_BUILD: &str = "53ed051ce";

/// The architecture every qwen35 manifest names in its `# arch` line.
pub const ARCH: &str = "qwen35";

/// The first 64, 600 and 4,096 ids of the prose corpus under this
/// vocabulary (`$BLOOMERY_DATA/qwen35/corpus-prose.ids`).
pub const P64: &str = "ref_qwen35_hidden_p64";
pub const P600: &str = "ref_qwen35_hidden_p600";
pub const P4096: &str = "ref_qwen35_hidden_p4096";

/// The sets, shortest first.
pub const HIDDEN_SETS: &[&str] = &[P64, P600, P4096];

/// Clef-Flash (32 layers, width 4,096) as bartowski publishes it at Q8_0:
/// every projection Q8_0, β and α F32.
pub const MODEL_FLASH_Q8: &str = "/models/clef-flash/Cloudflare_clef-flash-Q8_0.gguf";

/// The same 64, 600 and 4,096 prose ids (Clef-Flash's tokenizer is the
/// 27B's, byte for byte) through [`MODEL_FLASH_Q8`].
pub const FLASH_Q8_P64: &str = "ref_qwen35_flashq8_hidden_p64";
pub const FLASH_Q8_P600: &str = "ref_qwen35_flashq8_hidden_p600";
pub const FLASH_Q8_P4096: &str = "ref_qwen35_flashq8_hidden_p4096";

/// The Flash Q8_0 sets, shortest first.
pub const FLASH_Q8_SETS: &[&str] = &[FLASH_Q8_P64, FLASH_Q8_P600, FLASH_Q8_P4096];

/// lev (`ggml-org/lev-GGUF`'s Q4_K_M: 32 layers of width 2,560, a tied head),
/// under the box's `/root/models`.
pub const MODEL_LEV: &str = "/root/models/lev-4b/lev-Q4_K_M.gguf";

/// The same 64, 600 and 4,096 prose ids (lev's vocabulary is the 27B's)
/// through [`MODEL_LEV`].
pub const LEV_P64: &str = "ref_qwen35_lev_hidden_p64";
pub const LEV_P600: &str = "ref_qwen35_lev_hidden_p600";
pub const LEV_P4096: &str = "ref_qwen35_lev_hidden_p4096";

/// The lev sets, shortest first.
pub const LEV_SETS: &[&str] = &[LEV_P64, LEV_P600, LEV_P4096];

/// [`MODEL`], as a family's `runs`.
fn model() -> String {
    MODEL.to_string()
}

/// [`MODEL_FLASH_Q8`], as a family's `runs`.
fn model_flash_q8() -> String {
    MODEL_FLASH_Q8.to_string()
}

/// [`MODEL_LEV`], as a family's `runs`.
fn model_lev() -> String {
    MODEL_LEV.to_string()
}

/// llama.cpp mainline's final-norm hidden states.
pub static HIDDEN: Family = Family {
    name: "hidden-qwen35",
    sets: HIDDEN_SETS,
    resolve: None,
    recipe: "just dump-hidden-qwen35",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &["gate-gpu-clef-hidden"],
};

/// llama.cpp mainline's final-norm hidden states of Clef-Flash at Q8_0.
pub static HIDDEN_FLASH_Q8: Family = Family {
    name: "hidden-qwen35-flash-q8",
    sets: FLASH_Q8_SETS,
    resolve: None,
    recipe: "just dump-hidden-qwen35",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(model_flash_q8),
    draft_runs: None,
    consumers: &["gate-gpu-clef-hidden"],
};

/// llama.cpp mainline's final-norm hidden states of lev at Q4_K_M.
pub static HIDDEN_LEV: Family = Family {
    name: "hidden-qwen35-lev",
    sets: LEV_SETS,
    resolve: None,
    recipe: "just dump-hidden-qwen35",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(model_lev),
    draft_runs: None,
    consumers: &["gate-gpu-clef-hidden"],
};

/// The architecture's families, in the order `refset-check` lists them.
pub static FAMILIES: &[&Family] = &[&HIDDEN, &HIDDEN_FLASH_Q8, &HIDDEN_LEV];

#[cfg(test)]
mod tests {
    use super::{
        ARCH, HIDDEN, HIDDEN_FLASH_Q8, HIDDEN_LEV, LCPP_BUILD, MODEL, MODEL_FLASH_Q8, MODEL_LEV,
    };
    use crate::RefError;
    use std::path::Path;

    /// A set of hidden_ref's shape at `dir` with these header lines.
    fn write_set(dir: &Path, model: &str, build: &str, arch: &str, complete: bool) {
        let mut lines = vec![
            "# hidden_ref — llama.cpp mainline result_norm of every position".to_string(),
            format!("# model\t{model}"),
            format!("# build\t{build}"),
            format!("# arch\t{arch}"),
            "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1"
                .to_string(),
            "tensor\tresult_norm\t0\tf32\t1\t1\t1\t1\t4\t0\tRMS_NORM\t1\t0\t-\t-".to_string(),
        ];
        if complete {
            lines.push("# complete\t1\t0".to_string());
        }
        let path = dir.join("MANIFEST.tsv");
        std::fs::write(&path, lines.join("\n") + "\n")
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    }

    /// The family takes a complete set of the Q4_K_M file, the pinned
    /// mainline commit and this architecture, and refuses by name the BF16
    /// file it was quantized from, an ik build, the MoE architecture, and a
    /// set without its trailer.
    #[test]
    fn the_family_refuses_a_set_of_another_file_build_or_arch() -> Result<(), RefError> {
        let dir =
            std::env::temp_dir().join(format!("bloomery-refset-qwen35-{}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
        write_set(&dir, MODEL, LCPP_BUILD, ARCH, true);
        let p = HIDDEN.check_set(&dir)?;
        assert_eq!(p.dumped_from, MODEL);
        let bf16 = "/models/clef-27b/clef-27b-bf16.gguf";
        let cases = [
            (bf16, LCPP_BUILD, ARCH, true, "dumped from"),
            (MODEL, "db517b69", ARCH, true, "build"),
            (MODEL, LCPP_BUILD, "qwen35moe", true, "arch"),
            (MODEL, LCPP_BUILD, ARCH, false, "no `# complete` trailer"),
        ];
        for (model, build, arch, complete, want) in cases {
            write_set(&dir, model, build, arch, complete);
            let e = HIDDEN.check_set(&dir).expect_err("refused").to_string();
            assert!(e.contains(want), "{model} {build} {arch} {complete}: {e}");
        }
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    /// Each file's sets open through its own family only: a Flash Q8_0 set
    /// is read by its family and refused by the 27B's and by lev's, and the
    /// other way round.
    #[test]
    fn a_set_opens_through_its_files_family_only() -> Result<(), RefError> {
        let dir = std::env::temp_dir().join(format!(
            "bloomery-refset-qwen35-flash-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
        write_set(&dir, MODEL_FLASH_Q8, LCPP_BUILD, ARCH, true);
        assert_eq!(HIDDEN_FLASH_Q8.check_set(&dir)?.dumped_from, MODEL_FLASH_Q8);
        let e = HIDDEN.check_set(&dir).expect_err("refused").to_string();
        assert!(e.contains("dumped from"), "{e}");
        write_set(&dir, MODEL, LCPP_BUILD, ARCH, true);
        let e = HIDDEN_FLASH_Q8
            .check_set(&dir)
            .expect_err("refused")
            .to_string();
        assert!(e.contains("dumped from"), "{e}");
        write_set(&dir, MODEL_LEV, LCPP_BUILD, ARCH, true);
        assert_eq!(HIDDEN_LEV.check_set(&dir)?.dumped_from, MODEL_LEV);
        for family in [&HIDDEN, &HIDDEN_FLASH_Q8] {
            let e = family.check_set(&dir).expect_err("refused").to_string();
            assert!(e.contains("dumped from"), "{}: {e}", family.name);
        }
        write_set(&dir, MODEL_FLASH_Q8, LCPP_BUILD, ARCH, true);
        let e = HIDDEN_LEV.check_set(&dir).expect_err("refused").to_string();
        assert!(e.contains("dumped from"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }
}
