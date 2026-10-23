//! The fixture family's reference sets, opened through the family's check
//! (`refset::arch::qwen35moe::fixture::IK`): a set dumped from another file,
//! another generation of the fixture or another ik build is refused by name.

use refset::arch::qwen35moe::fixture as fx;
use refset::ik::{RefManifest, ref_tensor_logical_in, topk_ids_logical_within};

/// The batch set (every token from position 0), the step set (the prompt,
/// then one decode step, every node dumped) or the d1k set (the same layout
/// at the depth of its own prompt).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Set {
    Batch,
    Step,
    D1k,
}

/// Set `set`, checked against its family.
pub fn open(set: Set) -> Result<RefManifest, String> {
    let name = match set {
        Set::Batch => fx::BATCH,
        Set::Step => fx::STEP4,
        Set::D1k => fx::D1K,
    };
    RefManifest::open(&fx::IK.path(name), &fx::IK).map_err(|e| format!("{name}: {e}"))
}

/// The logical values of node `name` (its first occurrence).
pub fn tap(man: &RefManifest, name: &str) -> Result<Vec<f32>, String> {
    let row = man.tensor(name, 0).map_err(|e| format!("{name}: {e}"))?;
    ref_tensor_logical_in(&man.dir, row).map_err(|e| format!("{name}: {e}"))
}

/// ik's picks at layer `l`, every kept row's `n_used` ids.
pub fn topk(man: &RefManifest, l: usize, experts: u32) -> Result<Vec<i32>, String> {
    let name = format!("ffn_moe_topk-{l}");
    let row = man.tensor(&name, 0).map_err(|e| format!("{name}: {e}"))?;
    topk_ids_logical_within(man, row, experts).map_err(|e| format!("{name}: {e}"))
}
