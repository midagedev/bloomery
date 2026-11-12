//! The head128 program's session over its actual attention-ring body and the common hybrid
//! planner: Flint's files and K-EXAONE's `exaone-moe` ones, the router instance the body
//! monomorphizes at the load apart.

use std::num::NonZeroUsize;

use bloomery_gpu::head::Head;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError, GpuModel};
use bloomery_gpu_flint::Body;
use bloomery_levers::HostCfg;
use gguf::Split;
use model::arch::flint::place::PlanInputs;
use model::placement::Plan;

use super::{Batch, ResidentSequences};

const WHAT: &str = "flint session";

/// A Flint session, with resident slots backed by the body's own stores.
pub type FlintSession = crate::Session<Body>;

macro_rules! flint_batch {
    ($ty:ty) => {
        impl Batch for $ty {
            type Inputs = PlanInputs;
            type Sequences = ResidentSequences;

            const WHAT: &'static str = self::WHAT;

            fn read(file: &Split) -> Result<PlanInputs, GpuError> {
                PlanInputs::read(file).map_err(|e| GpuError::plan(WHAT, e))
            }

            fn configure(inputs: &mut PlanInputs, cfg: &super::HybridCfg) -> Result<(), GpuError> {
                model::arch::flint::place::configure_group(inputs, cfg.group)
                    .map_err(|e| GpuError::plan(WHAT, e))
            }

            fn open_placed(
                file: Split,
                plan: &Plan<'_>,
                inputs: &PlanInputs,
                host: HostCfg,
                residency: Residency,
                slots: NonZeroUsize,
            ) -> Result<GpuModel<Self>, GpuError> {
                Self::open_placed_slots(file, plan, inputs, 0, host, residency, slots)
            }

            fn batch(
                gpu: &Gpu,
                w: &Weights,
                body: &mut Self,
                head: &mut Head,
                ids: &[u32],
                at: u32,
            ) -> Result<bool, GpuError> {
                bloomery_gpu_flint::prefill::prefill(gpu, w, body, head, ids, at)
            }

            fn group(m: &mut GpuModel<Self>, g: usize) -> Result<bool, GpuError> {
                bloomery_gpu_flint::set_prefill_group(m, g)
            }

            fn keepable(m: &GpuModel<Self>, n: u32) -> u32 {
                m.body(WHAT)
                    .map_or(0, |b| b.keepable_at(m.pos(), n.min(m.pos())))
            }

            fn cut(m: &mut GpuModel<Self>, n: u32) -> Result<(), GpuError> {
                if n == 0 {
                    return m.reset();
                }
                let pos = m.pos();
                m.body_parts(WHAT)?.2.stand_for_cut(pos);
                m.rollback(n)
            }
        }
    };
}

flint_batch!(bloomery_gpu_flint::Body);
flint_batch!(bloomery_gpu_flint::ExaoneBody);
