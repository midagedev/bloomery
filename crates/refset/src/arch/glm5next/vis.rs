//! The GLM-5.3-Flash image-input families of the fixture tier: llama.cpp mainline's mtmd `glm5v` tower and its
//! `glm5-next` text model on [`TEXT_MODEL`], the respelled copy of the fixture file ([`crate::fixture::first_shard`]),
//! dumped by `tools/ref/clefvis/dump_mtmd.cpp` (`just dump-ref-glmv`) in the node dumps'
//! set format, at the mainline tree [`QVIS_LCPP_BUILD`], with the bf16 projector [`MMPROJ_BF16`]. One family a kind of
//! set, as the Qwen seats' are ([`crate::arch::qwen35moe::vis`]; the kinds and what they hold are
//! [`crate::clefvis::Kind`]'s), without the prose and bf16-rows controls, which are Qwen seats'. What this seat's sets
//! state that a Qwen seat's do not is read from their own lines ([`crate::clefvis`]): the `# projector glm5v` layout,
//! `# positions normal`, the `<|image|>` token; and its identity is the fixture's — the `# fixture` line of the original
//! file, the copy as `# text_model` and the copy's `# respell` record ([`PROFILE`]'s [`Respell`]).

use super::ARCH;
use crate::RefError;
use crate::clefvis::{Profile, ProjectorKind, QVIS_LCPP_BUILD, Respell, vis_families};

/// The projector every GLM-5.3-Flash image-input set is dumped with: unsloth's bf16 file from the repository the
/// text family's model comes from, under the box's `/models/mmproj`.
pub const MMPROJ_BF16: &str = "/models/mmproj/unsloth--GLM-5.3-Flash-GGUF/mmproj-BF16.gguf";

/// The sha256 of [`MMPROJ_BF16`] (1,164,010,176 bytes).
// PIN(2026-10-11): unsloth/GLM-5.3-Flash-GGUF's mmproj-BF16.gguf, equal to `sha256sum` of the file on the box and to
// the LFS oid `tools/ref/qvis/mmproj-dl.sh` checks it against.
pub const MMPROJ_BF16_SHA256: &str =
    "82b7adb44411dbb298b1c689567804e5ed0790d80f308ed8b5b6b8ac0527d2ac";

/// The text model of every set: the fixture's copy whose architecture takes mainline's spelling
/// (`tools/ref/gguf-respell.py`; mainline reads `glm5-next`, the fixture says `glm5next`), its first shard. The later
/// shards are links to the fixture's.
pub const TEXT_MODEL: &str = "/models/fixtures-mainline/glm5next/glm53-fixture-00001-of-00003.gguf";

/// The architecture the copy and so the prompt sets name in their `# arch` line.
pub const MAINLINE_ARCH: &str = "glm5-next";

/// The one value the copy changes besides names (`bloomery.respell.set`): the shared-expert count, 1 in the fixture and
/// 4 in the copy.
pub const RESPELL_SET: &str = "glm5-next.expert_shared_count=1->4";

/// The sha256 of the fixture's first shard, the file the copy was cut from (`bloomery.respell.source_sha256`).
// PIN(2026-10-11): `sha256sum` of /models/fixtures/glm5next/glm53-fixture-00001-of-00003.gguf (4,149,473,280 bytes), equal
// to the copy's own key. Too large to hash at a set's check: a regenerated fixture moves it with the `# fixture` line,
// and the copy and its sets are re-made with it.
pub const FIXTURE_SHA256: &str = "03963a052b1be333bf693bae8653a104d4c33cf534b90f82ff95ecc85b1c5f62";

/// The seat: the projector, the text width 4,096, the id of `<|image|>` and the respelled fixture copy.
pub static PROFILE: Profile = Profile {
    mmproj: MMPROJ_BF16,
    mmproj_sha256: MMPROJ_BF16_SHA256,
    n_embd: 4096,
    image_pad_id: 154_854,
    projector: ProjectorKind::Glm5v,
    respell: Some(Respell {
        from: ARCH,
        copy: TEXT_MODEL,
        set: RESPELL_SET,
        source_sha256: FIXTURE_SHA256,
    }),
};

/// [`MMPROJ_BF16`], as a tower family's `runs`.
fn mmproj() -> Result<String, RefError> {
    Ok(MMPROJ_BF16.to_string())
}

/// [`TEXT_MODEL`], as a prompt family's `runs`.
fn text_model() -> Result<String, RefError> {
    Ok(TEXT_MODEL.to_string())
}

vis_families! {
    recipe: "just dump-ref-glmv",
    prefix: "fx_ref_glm5next_vis",
    name: "glm5next-vis",
    profile: PROFILE,
    arch: MAINLINE_ARCH,
    build: QVIS_LCPP_BUILD,
    mmproj: mmproj,
    text: text_model,
    kinds: [preproc, taps, hidden, chatids, decode],
}
