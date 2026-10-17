//! EmbeddingGemma 2 (`gemma-embedding2`, its image tower `gemma4v`). Every set
//! is llama.cpp mainline's, dumped by the tree [`crate::emb2::LCPP_BUILD`]
//! (`tools/ref/emb2/emb2.sh`, `just dump-ref-emb2`) from the Q8_0 file
//! [`MODEL`] and the bf16 projector [`MMPROJ_BF16`]:
//!
//! - [`TEXT`]: `dump_emb2 text` over the text cases (the tokenizer oracle's
//!   gemma4 set, `ref-tokenizer-gemma4`):
//!   per case the token rows (`embd`, the output projection of every
//!   position), the mean-pooled vector before normalization (`pooled`), and
//!   the taps `inp_scaled`, `inp_per_layer`, `l_out-<il>` of every layer,
//!   `result_norm` and `result_embd`. [`TEXT_CPU`] is its CPU twin.
//! - [`TOWER`]: `dump_emb2 tower` over the image cases: per image mtmd's rows
//!   (`embd`) and the tower's taps. As the other image-input seats' tower sets,
//!   its `# model` is the projector file and its `# arch` the projector's
//!   (`clip`). [`TOWER_CPU`] is its CPU twin.
//! - [`SERVER`]: llama-server's `/v1/embeddings` vectors of every text case and
//!   image, normalized and raw (`tools/ref/emb2/server_vectors.py`). [`SERVER_CPU`] is its CPU twin.
//! - [`AUDIO_MEL`]: the pinned host preprocessor's mel from synthetic decoded
//!   mono PCM (`tools/ref/emb2/mel.sh`), with no model or device opened.
//!
//! The text, tower and server families open through [`crate::emb2::Emb2Set`]
//! with [`PROFILE`]: the node dumps' identity, then the kind, device and file
//! digests. [`AUDIO_MEL`] uses [`crate::ik::RefManifest`] with its synthetic
//! PCM identity, architecture, mainline build and completion trailer.

use crate::RefError;
use crate::emb2::{Kind, Profile, family};
use crate::family::{Build, Family, Identity};

/// The text file every set is dumped from: `ggml-org/embeddinggemma-2-GGUF`'s
/// Q8_0 at revision bfcd2987 (309,855,456 bytes).
pub const MODEL: &str = "/models/embeddinggemma-2/embeddinggemma-2-Q8_0.gguf";

/// The sha256 of [`MODEL`].
// PIN(2026-10-14): `sha256sum` of the file, equal on the DGX Spark and on the
// copy it was taken from.
pub const MODEL_SHA256: &str = "2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135";

/// The projector of the tower and server sets: the same repository's bf16
/// `mmproj-embeddinggemma-2-BF16.gguf` (982,074,784 bytes, projector
/// `gemma4v`).
pub const MMPROJ_BF16: &str = "/models/embeddinggemma-2/mmproj-embeddinggemma-2-BF16.gguf";

/// The sha256 of [`MMPROJ_BF16`].
// PIN(2026-10-14): `sha256sum` of the file, equal on the DGX Spark and on the
// copy it was taken from.
pub const MMPROJ_BF16_SHA256: &str =
    "d2033b3cd0223cfb2e7a2b50dc80a664c78c816d375f5b017ec3ef80ee0bc766";

/// The architecture the text and server sets name in their `# arch` line.
pub const ARCH: &str = "gemma-embedding2";

/// The tokenizer oracle set that supplies the text cases.
pub const IDS_SET: &str = "ref-tokenizer-gemma4";

/// The text sets.
pub const TEXT_SET: &str = "ref_gemma_embedding2_text";
pub const TEXT_CPU_SET: &str = "ref_gemma_embedding2_text.cpu";

/// The image tower sets.
pub const TOWER_SET: &str = "ref_gemma_embedding2_tower";
pub const TOWER_CPU_SET: &str = "ref_gemma_embedding2_tower.cpu";

/// The image inputs the mainline tower dump supplies, in the profile's order.
pub const TOWER_IMAGES: &[&str] = &[
    "small-200x200",
    "min-96x64",
    "free-448x448",
    "free-640x480",
    "pad-777x513",
    "strip-1000x1",
    "free-1024x768",
];

/// The gate that reads the tower sets.
pub const TOWER_GATE: &str = "gate-gpu-gemma4v-tower";

/// llama-server's vectors.
pub const SERVER_SET: &str = "ref_gemma_embedding2_server";
pub const SERVER_CPU_SET: &str = "ref_gemma_embedding2_server.cpu";

/// Frame identities in the decoded-video reference input, in sequence order.
pub const VIDEO_IMAGES: &str = include_str!("video-frames.txt");
/// The pinned mainline video helper's text before the first frame.
pub const VIDEO_PREFIX: &str = "Video:";
/// Ordered decoded-frame vectors at the checkpoint's video token budget.
pub const VIDEO_SET: &str = "ref_gemma_embedding2_video";
pub const VIDEO_CPU_SET: &str = "ref_gemma_embedding2_video.cpu";

/// A row's band in the encoder and serving gates: this many times the CPU
/// twin's distance from the card set (`shared/glm5next_media.rs`'s `TWIN`,
/// the Qwen seats' rule).
pub const TWIN: f64 = 2.5;

/// The least relative band in the encoder and serving gates.
// PIN(2026-10-15): 1e-2 over the distances whose twin band is smaller,
// measured on a DGX Spark: `inp_per_layer` 1.344e-3 to 1.594e-3 against
// twins of 1.066e-7 to 2.410e-7 in every case, and the shortest cases'
// `l_out` up to 9.046e-3 at layer 0.
pub const REL_FLOOR: f64 = 1.0e-2;

const RECIPE: &str = "just dump-ref-emb2";

/// The files every set is dumped from.
pub static PROFILE: Profile = Profile {
    model: MODEL,
    model_sha256: MODEL_SHA256,
    mmproj: Some((MMPROJ_BF16, MMPROJ_BF16_SHA256)),
};

/// [`MODEL`], as a text or server family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// [`MMPROJ_BF16`], as a tower family's `runs`.
fn mmproj() -> Result<String, RefError> {
    Ok(MMPROJ_BF16.to_string())
}

/// The gate that reads the text sets.
const E2E: &[&str] = &["gate-gpu-gemma-embedding2-e2e"];
const IMAGE: &[&str] = &[TOWER_GATE, "gate-gpu-gemma-embedding2-e2e"];

/// The text cases' token rows, pooled vectors and layer taps on the card.
pub static TEXT: Family = Family {
    consumers: E2E,
    ..family(
        "text-gemma-embedding2",
        &[TEXT_SET],
        Kind::Text,
        &PROFILE,
        ARCH,
        model,
        RECIPE,
    )
};
/// The same, with no card in view.
pub static TEXT_CPU: Family = Family {
    consumers: &[
        "gate-gpu-gemma-embedding2-e2e",
        "gate-gpu-gemma-embedding2-serve",
    ],
    ..family(
        "text-gemma-embedding2-cpu",
        &[TEXT_CPU_SET],
        Kind::TextCpu,
        &PROFILE,
        ARCH,
        model,
        RECIPE,
    )
};
/// The image cases' tower taps and rows on the card.
pub static TOWER: Family = Family {
    consumers: IMAGE,
    ..family(
        "tower-gemma4v",
        &[TOWER_SET],
        Kind::Tower,
        &PROFILE,
        crate::clefvis::TOWER_ARCH,
        mmproj,
        RECIPE,
    )
};
/// The same, with no card in view.
pub static TOWER_CPU: Family = Family {
    consumers: IMAGE,
    ..family(
        "tower-gemma4v-cpu",
        &[TOWER_CPU_SET],
        Kind::TowerCpu,
        &PROFILE,
        crate::clefvis::TOWER_ARCH,
        mmproj,
        RECIPE,
    )
};
/// llama-server's `/v1/embeddings` vectors.
pub static SERVER: Family = Family {
    consumers: &[
        "gate-gpu-gemma-embedding2-e2e",
        "gate-gpu-gemma-embedding2-serve",
    ],
    ..family(
        "server-gemma-embedding2",
        &[SERVER_SET],
        Kind::Server,
        &PROFILE,
        ARCH,
        model,
        RECIPE,
    )
};

/// The host audio preprocessing set, from synthetic decoded mono PCM.
pub const AUDIO_MEL_SET: &str = "ref_gemma_embedding2_mel";
pub const AUDIO_PCM: &str = "synthetic-gemma4a-pcm";

fn audio_pcm() -> Result<String, RefError> {
    Ok(AUDIO_PCM.to_string())
}

/// The pinned mainline host preprocessor, without a model or a device.
pub static AUDIO_MEL: Family = Family {
    name: "mel-gemma4a",
    sets: &[AUDIO_MEL_SET],
    resolve: None,
    recipe: "just dump-ref-emb2-mel",
    identity: Identity::Manifest,
    arch: Some("gemma4a"),
    build: Some(Build::Is(crate::emb2::LCPP_BUILD)),
    runs: Some(audio_pcm),
    draft_runs: None,
    consumers: &["gate-emb2-mel"],
};

/// The same requests with the text model and projector on the CPU.
pub static SERVER_CPU: Family = Family {
    consumers: E2E,
    ..family(
        "server-gemma-embedding2-cpu",
        &[SERVER_CPU_SET],
        Kind::ServerCpu,
        &PROFILE,
        ARCH,
        model,
        RECIPE,
    )
};

/// llama-server's timestamp-free decoded-frame sequence.
pub static VIDEO: Family = Family {
    consumers: E2E,
    ..family(
        "video-gemma-embedding2",
        &[VIDEO_SET],
        Kind::Server,
        &PROFILE,
        ARCH,
        model,
        RECIPE,
    )
};
/// The same decoded frames with both models on the CPU.
pub static VIDEO_CPU: Family = Family {
    consumers: E2E,
    ..family(
        "video-gemma-embedding2-cpu",
        &[VIDEO_CPU_SET],
        Kind::ServerCpu,
        &PROFILE,
        ARCH,
        model,
        RECIPE,
    )
};

/// Mainline's audio tower output from the BF16 projector and decoded mel chunks.
pub static AUDIO_TOWER: Family = Family {
    name: "tower-gemma4a",
    sets: &["ref_gemma_embedding2_audio"],
    resolve: None,
    recipe: "just dump-ref-emb2-audio",
    identity: Identity::Manifest,
    arch: Some("gemma4a"),
    build: Some(Build::Is(crate::emb2::LCPP_BUILD)),
    runs: Some(mmproj),
    draft_runs: None,
    consumers: &["gate-gpu-gemma4a"],
};

/// The audio tower's CPU twin, used to derive the card's rounding band.
pub static AUDIO_TOWER_CPU: Family = Family {
    name: "tower-gemma4a-cpu",
    sets: &["ref_gemma_embedding2_audio.cpu"],
    ..AUDIO_TOWER
};

/// The architecture's families, in the order `refset-check` lists them.
pub static FAMILIES: &[&Family] = &[
    &TEXT,
    &TEXT_CPU,
    &TOWER,
    &TOWER_CPU,
    &SERVER,
    &SERVER_CPU,
    &AUDIO_MEL,
    &VIDEO,
    &VIDEO_CPU,
    &AUDIO_TOWER,
    &AUDIO_TOWER_CPU,
];

#[cfg(test)]
mod tests;
