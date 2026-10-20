//! The tiered body's shell: what every body over a [`Hybrid`] host tier
//! holds the same way (mimo2, glm5next, solar2) — the tier, the residency
//! machine's glue, the per-layer taps — and the accessors a gate, a binary
//! or the shared [`TierBody`](crate::host::served::TierBody) impl read them
//! through. A body holds one [`TierShell`] and forwards to it; its layers,
//! stores, scratch and step stay the body's.

use std::sync::Arc;

use cuda_core::DeviceBuffer;

use crate::host::HostExperts;
use crate::host::PassKind;
use crate::host::served::ResidencyParts;
use crate::host::swap::PassReport;
use crate::host::swap_source::{FileSwap, ResidencyGlue};
use crate::hybrid::Hybrid;
use crate::tensor::DeviceTensor;
use crate::{Gpu, GpuError};

/// The host tier, the residency glue and the taps of one tiered body.
pub struct TierShell<E: HostExperts> {
    pub hybrid: Hybrid<E>,
    pub glue: ResidencyGlue,
    /// Each layer's output after the last step, when a gate armed them.
    pub taps: Option<Vec<DeviceBuffer<f32>>>,
    /// A tap's width: the layer output's values.
    pub embd: usize,
    /// What the shell's refusals name (the body's `WHAT`).
    pub what: &'static str,
}

impl<E: HostExperts> TierShell<E> {
    /// The shell of a body whose layers are `embd` values wide in a tap,
    /// named `what` in its refusals. The taps start disarmed.
    #[must_use]
    pub fn new(
        hybrid: Hybrid<E>,
        glue: ResidencyGlue,
        embd: usize,
        what: &'static str,
    ) -> TierShell<E> {
        TierShell {
            hybrid,
            glue,
            taps: None,
            embd,
            what,
        }
    }

    /// The host tier.
    #[must_use]
    pub fn hybrid(&self) -> &Hybrid<E> {
        &self.hybrid
    }

    /// The host tier, for a gate's instrument or test seam.
    pub fn hybrid_mut(&mut self) -> &mut Hybrid<E> {
        &mut self.hybrid
    }

    /// Keep every residency boundary's report from now on, at most `passes`
    /// between two takes.
    pub fn log_residency(&mut self, passes: usize) {
        self.glue.log_passes(passes);
    }

    /// The residency machine's source, when the load runs one.
    #[must_use]
    pub fn residency_source(&self) -> Option<&FileSwap> {
        self.glue.source()
    }

    /// The residency boundaries' reports since the last take, in order.
    pub fn take_residency_passes(&mut self) -> Vec<(PassKind, PassReport)> {
        self.glue.take_passes()
    }

    /// Arm (or disarm) `layers` taps of `embd` values. The caller drops its
    /// captured chains first: a capture holds the tap copies it was taken
    /// with.
    pub fn set_taps(&mut self, gpu: &Gpu, layers: usize, on: bool) -> Result<(), GpuError> {
        self.taps = if on {
            Some(
                (0..layers)
                    .map(|_| DeviceBuffer::zeroed(gpu.stream(), self.embd))
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else {
            None
        };
        Ok(())
    }

    /// Every layer's output after the last step, `embd` a layer. Blocking;
    /// refused by name when the taps are not armed.
    pub fn taps(&self, gpu: &Gpu) -> Result<Vec<f32>, GpuError> {
        let taps = self.taps.as_ref().ok_or(GpuError::State {
            what: self.what,
            missing: "armed taps (set_taps)",
        })?;
        let mut out = Vec::with_capacity(taps.len() * self.embd);
        for t in taps {
            out.extend(t.to_host_vec(gpu.stream())?);
        }
        Ok(out)
    }

    /// The residency parts the shared [`TierBody`](crate::host::served::TierBody)
    /// impl drives, over the body's slot map card copy `slots`. Always
    /// `Some`: a load with no machine still runs the tier's call-order
    /// checks ([`HostTier::swap_at`](crate::host::HostTier::swap_at)), and
    /// the glue's calls do nothing else.
    pub fn residency_parts<'a>(
        &'a mut self,
        slots: &'a Arc<DeviceTensor<u32>>,
    ) -> Option<ResidencyParts<'a, E>> {
        Some(ResidencyParts {
            glue: &mut self.glue,
            hybrid: &mut self.hybrid,
            slots,
        })
    }
}
