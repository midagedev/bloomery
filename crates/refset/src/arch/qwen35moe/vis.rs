//! The Qwen3.6-35B-A3B image-input families: llama.cpp mainline's mtmd `qwen3vl_merger` tower and its qwen35moe text
//! model on [`MODEL`], dumped by `tools/ref/clefvis/dump_mtmd.cpp` (`just dump-ref-qvis qwen35moe`) in the node dumps'
//! set format, at the mainline tree [`QVIS_LCPP_BUILD`], with the bf16 projector [`MMPROJ_BF16`]. One family a kind of
//! set, as the Clef-Flash families are ([`crate::arch::qwen35::clefvis`]); the kinds and what they hold are
//! [`crate::clefvis::Kind`]'s. A set of the Clef-Flash seat, or of Qwen3.8's, fails these families' check by its model
//! file or its projector.

use super::{ARCH, MODEL};
use crate::RefError;
use crate::clefvis::{Profile, ProjectorKind, QVIS_LCPP_BUILD, vis_families};

/// The projector every Qwen3.6 image-input set is dumped with: lmstudio-community's bf16 file, the repository
/// `docs/models.md` names for this model, under the box's `/models/mmproj`.
pub const MMPROJ_BF16: &str =
    "/models/mmproj/lmstudio-community--Qwen3.6-35B-A3B-GGUF/mmproj-Qwen3.6-35B-A3B-BF16.gguf";

/// The sha256 of [`MMPROJ_BF16`], the LFS oid of that file (902,822,016 bytes).
// PIN(2026-10-10): lmstudio-community/Qwen3.6-35B-A3B-GGUF at rev 68a34855558a, read from its tree listing
// (huggingface.co/api/models/<repo>/tree/main) and equal to `sha256sum` of the downloaded file.
pub const MMPROJ_BF16_SHA256: &str =
    "e5c205cec2fd28f66c3895e4040021ab994b860323c3db8531640305ff49b322";

/// The seat: the projector, the text width 2,048 and the image-pad id of the shared Qwen vocabulary.
pub static PROFILE: Profile = Profile {
    mmproj: MMPROJ_BF16,
    mmproj_sha256: MMPROJ_BF16_SHA256,
    n_embd: 2048,
    image_pad_id: 248_056,
    projector: ProjectorKind::Qwen3vlMerger,
    respell: None,
};

/// [`MMPROJ_BF16`], as a tower family's `runs`.
fn mmproj() -> Result<String, RefError> {
    Ok(MMPROJ_BF16.to_string())
}

/// [`MODEL`], as a prompt family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

vis_families! {
    recipe: "just dump-ref-qvis qwen35moe",
    prefix: "ref_qwen35moe_vis",
    name: "qwen35moe-vis",
    profile: PROFILE,
    arch: ARCH,
    build: QVIS_LCPP_BUILD,
    mmproj: mmproj,
    text: model,
}

/// The Qwen3.6-35B-A3B image-input families of the fixture tier: the same seat's dump (mainline's
/// mtmd `qwen3vl_merger` tower and its qwen35moe text model, `tools/ref/clefvis/dump_mtmd.cpp`),
/// with the text model the fixture file itself ([`crate::fixture::first_shard`], handed to the
/// dump as `-m` as it is: the fixture says `qwen35moe`, mainline's own spelling, so no respelled
/// copy) and the projector the synthetic encoder file (`vision::arch::qwen3vl::fixture`, written by
/// `just fixture-mmproj`). One family a kind of set, as the real seat's are, minus the prose
/// control (no clause reads one on this seat). What states the fixture generation is the seat's
/// own pin: the mmproj's sha256 (the file is a function of its seed, so it never moves without a
/// fill change), while the text model's `# model` line names the fixture file by path — a Qwen
/// seat's sets carry no `# fixture` line (a `qwen3vl_merger` dump writes none), so a regenerated
/// text fixture is caught by the gates' bands, not at open.
pub mod fx {
    use super::ARCH;
    use crate::RefError;
    use crate::clefvis::{Profile, ProjectorKind, QVIS_LCPP_BUILD, vis_families};

    /// The projector every fixture-tier set of this seat is dumped with: the synthetic encoder
    /// file (`vision::arch::qwen3vl::fixture`'s table at the qwen35moe fixture's width), written by
    /// `just fixture-mmproj` under the box's `/models/fixtures-mainline`.
    pub const MMPROJ_BF16: &str = "/models/fixtures-mainline/qwen35moe/mmproj-fixture-bf16.gguf";

    /// The sha256 of [`MMPROJ_BF16`] (902,821,632 bytes).
    // PIN(2026-11-21): `vision::arch::qwen3vl::fixture`'s digest test pins the same bytes from inside the writer;
    // this is the sha256sum of the file the recipe wrote, equal to the test's pinned file sha256.
    pub const MMPROJ_BF16_SHA256: &str =
        "8fb7ca9c074a62131187e6641a753b3debcda108397128a73fff0fe962d08ac7";

    /// The seat: the synthetic projector, the text width 2,048 and the image-pad id of the shared
    /// Qwen vocabulary, on the real file's own fixture.
    pub static PROFILE: Profile = Profile {
        mmproj: MMPROJ_BF16,
        mmproj_sha256: MMPROJ_BF16_SHA256,
        n_embd: 2048,
        image_pad_id: 248_056,
        projector: ProjectorKind::Qwen3vlMerger,
        respell: None,
    };

    /// [`MMPROJ_BF16`], as a tower family's `runs`.
    fn mmproj() -> Result<String, RefError> {
        Ok(MMPROJ_BF16.to_string())
    }

    /// The fixture file the tree runs ([`crate::fixture::first_shard`]), as a prompt family's
    /// `runs`.
    fn fixture() -> Result<String, RefError> {
        crate::fixture::first_shard(ARCH)
    }

    vis_families! {
        recipe: "just dump-ref-qvis qwen35moe",
        prefix: "fx_ref_qwen35moe_vis",
        name: "qwen35moe-vis-fx",
        profile: PROFILE,
        arch: ARCH,
        build: QVIS_LCPP_BUILD,
        mmproj: mmproj,
        text: fixture,
        kinds: [taps, hidden, bf16rows, chatids, decode],
    }
}
