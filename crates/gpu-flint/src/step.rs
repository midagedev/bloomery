//! One-token walk: common attention, then the dense or routed block, and the output head.

use bloomery_gpu::attention_h128::{Binding, Cache, Kernels, Scratch};
use bloomery_gpu::head::Head;
use bloomery_gpu::host::StepLeg;
use bloomery_gpu::host::run::HostRun;
use bloomery_gpu::rope_table::RopeRows;
use bloomery_gpu::routed_block::RouterEntry;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use bloomery_gpu_deepseek41::shared_routed;
use cuda_core::DeviceBuffer;
use runtime::layer::{FfnKind, Layer};
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};

use crate::body::{FfnBody, FlintBody, StepRows, shape};

pub(crate) fn enqueue<R: RouterEntry>(
    gpu: &Gpu,
    w: &Weights,
    body: &mut FlintBody<R>,
    head: &mut Head,
) -> Result<(), GpuError> {
    let FlintBody {
        shell,
        cfg,
        bindings,
        stores,
        k,
        attention,
        s,
        rope,
        ffn,
        ..
    } = body;
    let mut port = StepLeg::new(gpu.stream(), &mut shell.hybrid);
    let mut program = Step {
        gpu,
        w,
        cfg,
        bindings,
        stores,
        k,
        attention,
        s,
        rope,
        ffn,
        taps: shell.taps.as_deref_mut(),
        head,
    };
    sched::walk(
        Overlap {
            units: 1,
            cols: 1,
            port: PortKind::Step,
        },
        program.cfg.len(),
        &mut port,
        &mut program,
    )
}

struct Step<'a, R: RouterEntry> {
    gpu: &'a Gpu,
    w: &'a Weights,
    cfg: &'a [Layer],
    bindings: &'a [Binding],
    stores: &'a mut [Cache],
    k: &'a Kernels,
    attention: &'a mut Scratch,
    s: &'a mut StepRows,
    rope: &'a RopeRows,
    ffn: &'a mut FfnBody<R>,
    taps: Option<&'a mut [DeviceBuffer<f32>]>,
    head: &'a mut Head,
}

impl<R: RouterEntry> Step<'_, R> {
    fn tap(&mut self, l: usize) -> Result<(), GpuError> {
        let last = l + 1 == self.cfg.len();
        if let Some(taps) = self.taps.as_deref_mut() {
            let out = if last {
                &*self.head.input_mut()
            } else {
                &self.s.x
            };
            taps[l].copy_from_device_async(out, self.gpu.stream())?;
        }
        Ok(())
    }
}

impl<'a, R: RouterEntry> LayerProgram for Step<'a, R> {
    type Port = StepLeg<'a, HostRun>;

    fn begin(&mut self, unit: usize) -> Result<(), GpuError> {
        one(unit)
    }

    fn front(&mut self, port: &mut Self::Port, at: At) -> Result<(), GpuError> {
        one(at.unit)?;
        let l = at.layer;
        // SAFETY: refresh stages the one position and its count, position + 1.
        unsafe {
            self.bindings[l].enqueue(
                self.gpu,
                self.w,
                self.k,
                self.attention,
                &mut self.stores[l],
                &self.s.x,
                &mut self.s.x1,
                &self.s.pos,
                &self.s.n_keys,
                1,
                self.rope,
            )?;
        }
        match self.cfg[l].ffn {
            FfnKind::Moe => {
                shared_routed::front(self.gpu, self.w, self.ffn, &self.s.x1, (port.hybrid(), l))
            }
            FfnKind::Dense => {
                let dest = if l + 1 == self.cfg.len() {
                    self.head.input_mut()
                } else {
                    &mut self.s.x
                };
                shared_routed::dense(
                    self.gpu,
                    self.w,
                    self.ffn,
                    l,
                    (&self.s.x1, &mut self.s.xn, &mut self.s.delta),
                    dest,
                )?;
                self.tap(l)
            }
        }
    }

    fn shadow(&mut self, port: &mut Self::Port, at: At) -> Result<(), GpuError> {
        one(at.unit)?;
        if !self.cfg[at.layer].host_leg() {
            return Ok(());
        }
        shared_routed::shadow(
            self.gpu,
            self.w,
            self.ffn,
            &self.s.x1,
            (port.hybrid().boundary(), at.layer),
        )
    }

    fn back(&mut self, port: &mut Self::Port, at: At) -> Result<(), GpuError> {
        one(at.unit)?;
        let l = at.layer;
        if !self.cfg[l].host_leg() {
            return Ok(());
        }
        let dest = if l + 1 == self.cfg.len() {
            self.head.input_mut()
        } else {
            &mut self.s.x
        };
        shared_routed::back(self.gpu, self.ffn, port.hybrid(), l, dest)?;
        self.tap(l)?;
        port.hybrid().row_enqueued(l, at.unit)
    }

    fn end(&mut self, unit: usize) -> Result<(), GpuError> {
        one(unit)?;
        self.head.enqueue(self.gpu, self.w)
    }
}

fn one(unit: usize) -> Result<(), GpuError> {
    if unit == 0 {
        Ok(())
    } else {
        Err(shape(format!("step walk has one unit, not unit {unit}")))
    }
}
