//! The pipelines of Inkling 2's kernels (`kernels/{res_mix,conv_res,qk_clip,
//! rel_flash,loop_control,out_gate,nvfp4,router_situ,situ,head_f32}.metal`),
//! one row each. Their launches are the files of the same names beside this
//! one; every shape they check is `inkling2_geom`'s.

use bloomery_device::DeviceError;

use crate::library::{Library, Pipeline};

/// The compute pipelines of the family's own kernels, built at open with
/// every other family: a missing entry is a named error there, and the first
/// launch pays no compile inside its own wall.
pub(crate) struct Pipelines {
    pub(super) res_mix: Pipeline,
    pub(super) conv_res: Pipeline,
    pub(super) qk_clip: Pipeline,
    pub(super) loop_control: Pipeline,
    pub(super) out_gate: Pipeline,
    pub(super) nvfp4: Pipeline,
    pub(super) nvfp4_rows: Pipeline,
    pub(super) q8_0_scaled: Pipeline,
    pub(super) mxfp8_rows: Pipeline,
    pub(super) router_sigmoid: Pipeline,
    pub(super) pair_situ: Pipeline,
    pub(super) head_f32: Pipeline,
    pub(super) rel_flash: Pipeline,
    pub(super) rel_fold: Pipeline,
}

impl Pipelines {
    pub(crate) fn new(library: &Library) -> Result<Self, DeviceError> {
        Ok(Self {
            res_mix: library.pipeline("res_mix_normed")?,
            conv_res: library.pipeline("conv_res")?,
            qk_clip: library.pipeline("qk_clip_append")?,
            loop_control: library.pipeline("loop_control")?,
            out_gate: library.pipeline("out_gate_proj")?,
            nvfp4: library.pipeline("nvfp4_gemv")?,
            nvfp4_rows: library.pipeline("nvfp4_rows_gemv")?,
            q8_0_scaled: library.pipeline("q8_0_scaled_gemv")?,
            mxfp8_rows: library.pipeline("mxfp8_rows_gemv")?,
            router_sigmoid: library.pipeline("router_sigmoid")?,
            pair_situ: library.pipeline("pair_situ_rows")?,
            head_f32: library.pipeline("head_logits_f32")?,
            rel_flash: library.pipeline("rel_flash_f16")?,
            rel_fold: library.pipeline("rel_flash_fold")?,
        })
    }
}
