//! The DeepSeek-V4.1-Flash vision families: the official checkpoint's
//! encoder taps, which the vision gates compare against, and the fork
//! comparison, which feeds the encoder's rows to the text model in
//! smalinin's llama.cpp fork and in the engine.

use crate::family::{Build, Family, Identity};

/// The vision tower's architecture: its mmproj file's `clip.projector_type`.
pub const ARCH: &str = "deepseek41v";

/// The vision oracle's set.
pub const VISION_SET: &str = "ref-vision/deepseek41v";

/// The official checkpoint's revision every vision set must name.
pub const VISION_REVISION: &str = "dba1be0a40aa45a94ad051997016db3960a90277";

/// The official vision encoder's taps, from the fp8 checkpoint.
pub static VISION: Family = Family {
    name: "vision-deepseek41v",
    sets: &[VISION_SET],
    resolve: None,
    recipe: "just dump-ref-vision",
    identity: Identity::Checkpoint {
        revision: VISION_REVISION,
    },
    arch: None,
    build: None,
    runs: None,
    draft_runs: None,
    consumers: &["gate-vision", "gate-gpu-vision"],
};

/// The fork comparison's set: the scenes (`tools/ref/vision/scenes/`) and
/// their controls.
pub const VISREF_SET: &str = "ref-visref/deepseek41-scenes";

/// The fork every comparison set must name: smalinin's llama.cpp, branch
/// `my_build_deepseek41` (`tools/ref/vision/build-fork.sh`).
pub const FORK_BUILD: &str = "cfd8adcf6fb86ca523ad72fe21e8d2ac3860cd8e";

/// The fork's answers and logits on the V4.1 file the tree runs, fed the
/// vision oracle's rows.
pub static VISREF: Family = Family {
    name: "visref-deepseek41",
    sets: &[VISREF_SET],
    resolve: None,
    recipe: "just dump-ref-visref",
    identity: Identity::ForkManifest {
        rows_revision: VISION_REVISION,
    },
    arch: Some(crate::arch::deepseek41::ARCH),
    build: Some(Build::Is(FORK_BUILD)),
    runs: Some(gguf::v41::model),
    draft_runs: None,
    // No gate reads it: its reader is the driver tools/ref/vision/visref_ds41.rs,
    // which needs the engine's media prompt call and has no recipe in this tree.
    consumers: &[],
};

/// The architecture's families.
pub static FAMILIES: &[&Family] = &[&VISION, &VISREF];
