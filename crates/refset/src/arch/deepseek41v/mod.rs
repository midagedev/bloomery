//! The DeepSeek-V4.1-Flash vision tower's family: the official checkpoint's
//! encoder taps, which the vision gates compare against.

use crate::family::{Family, Identity};

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

/// The architecture's families.
pub static FAMILIES: &[&Family] = &[&VISION];
