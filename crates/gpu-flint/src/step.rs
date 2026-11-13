//! The step's walk: common attention, then the dense or routed block, and the output head, for one
//! row (the one-token step) or the rows of a verify pass, one layer apart on the one stream.

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

use crate::body::{FfnBody, FlintBody, PASS_ROWS, StepRows, shape};

/// The one-token step: one unit into `head`.
pub(crate) fn enqueue<R: RouterEntry>(
    gpu: &Gpu,
    w: &Weights,
    body: &mut FlintBody<R>,
    head: &mut Head,
) -> Result<(), GpuError> {
    walk(gpu, w, body, std::slice::from_mut(head))
}

/// The walk of `heads.len()` rows, row `r` into `heads[r]`: each row runs the step's launches in the
/// step's order over its own buffers, the rows one layer apart (`runtime::sched`'s point `(2, 1,
/// Step)`, [`bloomery_gpu::hybrid::Chain::Pair`]) so the host serves one row's layer while the card
/// runs the other's front and shadow. Row 1's attention at a layer reads the key row 0 wrote at that
/// layer just before it, and row 0 reads nothing row 1 writes, so each row's logits and every store
/// row are what two steps in turn write. What a row writes in one part and reads in a later one is
/// its own: the running rows and position words ([`StepRows`]), the routed block's step buffers, card
/// sums and the host boundary's row ([`shared_routed::front_of`] and its two). The attention's other
/// buffers are written and read inside one front, so the rows share them.
pub(crate) fn walk<R: RouterEntry>(
    gpu: &Gpu,
    w: &Weights,
    body: &mut FlintBody<R>,
    heads: &mut [Head],
) -> Result<(), GpuError> {
    let units = heads.len();
    if units == 0 || units > PASS_ROWS {
        return Err(shape(format!(
            "a walk of {units} rows; the load holds 1 to {PASS_ROWS}"
        )));
    }
    if units > 1 && body.shell.taps.is_some() {
        return Err(shape(
            "a walk of several rows with the taps armed: they copy the step's one row",
        ));
    }
    let FlintBody {
        shell,
        cfg,
        bindings,
        stores,
        k,
        attention,
        s,
        rows,
        rope,
        ffn,
        draft,
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
        rows,
        rope,
        ffn,
        taps: shell.taps.as_deref_mut(),
        heads,
    };
    sched::walk(
        Overlap {
            units,
            cols: 1,
            port: PortKind::Step,
        },
        program.cfg.len(),
        &mut port,
        &mut program,
    )?;
    let heads = program.heads;
    match draft.as_mut() {
        Some(d) => d.keep_rows(gpu, heads),
        None => Ok(()),
    }
}

struct Step<'a, R: RouterEntry> {
    gpu: &'a Gpu,
    w: &'a Weights,
    cfg: &'a [Layer],
    bindings: &'a [Binding],
    stores: &'a mut [Cache],
    k: &'a Kernels,
    attention: &'a mut Scratch,
    /// Row 0's buffers, the step's own.
    s: &'a mut StepRows,
    /// The buffers of rows 1.. of a pass.
    rows: &'a mut [StepRows],
    rope: &'a RopeRows,
    ffn: &'a mut FfnBody<R>,
    taps: Option<&'a mut [DeviceBuffer<f32>]>,
    heads: &'a mut [Head],
}

/// Row `unit`'s buffers: row 0's are `s`, the rows after it `rows`'.
fn row_of<'b>(
    s: &'b mut StepRows,
    rows: &'b mut [StepRows],
    unit: usize,
) -> Result<&'b mut StepRows, GpuError> {
    let armed = rows.len() + 1;
    match unit {
        0 => Ok(s),
        u => rows
            .get_mut(u - 1)
            .ok_or_else(|| shape(format!("row {u} of {armed} armed"))),
    }
}

impl<R: RouterEntry> Step<'_, R> {
    /// Whether `unit` is a row of this walk.
    fn check(&self, unit: usize) -> Result<(), GpuError> {
        if unit < self.heads.len() {
            Ok(())
        } else {
            Err(shape(format!(
                "walk of {} rows has no unit {unit}",
                self.heads.len()
            )))
        }
    }

    /// The taps copy the step's one row, so only a one-row walk has them armed.
    fn tap(&mut self, l: usize) -> Result<(), GpuError> {
        let last = l + 1 == self.cfg.len();
        if let Some(taps) = self.taps.as_deref_mut() {
            let out = if last {
                &*self.heads[0].input_mut()
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
        self.check(unit)
    }

    fn front(&mut self, port: &mut Self::Port, at: At) -> Result<(), GpuError> {
        self.check(at.unit)?;
        let l = at.layer;
        let last = l + 1 == self.cfg.len();
        let r = row_of(self.s, self.rows, at.unit)?;
        // SAFETY: the plan stages the row's position and its count, position + 1.
        unsafe {
            self.bindings[l].enqueue(
                self.gpu,
                self.w,
                self.k,
                self.attention,
                &mut self.stores[l],
                &r.x,
                &mut r.x1,
                &r.pos,
                &r.n_keys,
                1,
                self.rope,
            )?;
        }
        match self.cfg[l].ffn {
            FfnKind::Moe => shared_routed::front_of(
                self.gpu,
                self.w,
                self.ffn,
                &r.x1,
                (port.hybrid(), l, at.unit),
            ),
            FfnKind::Dense => {
                let dest = if last {
                    self.heads[at.unit].input_mut()
                } else {
                    &mut r.x
                };
                shared_routed::dense(
                    self.gpu,
                    self.w,
                    self.ffn,
                    l,
                    (&r.x1, &mut r.xn, &mut r.delta),
                    dest,
                )?;
                self.tap(l)
            }
        }
    }

    fn shadow(&mut self, port: &mut Self::Port, at: At) -> Result<(), GpuError> {
        self.check(at.unit)?;
        if !self.cfg[at.layer].host_leg() {
            return Ok(());
        }
        let r = row_of(self.s, self.rows, at.unit)?;
        shared_routed::shadow_of(
            self.gpu,
            self.w,
            self.ffn,
            &r.x1,
            (port.hybrid().boundary(), at.layer, at.unit),
        )
    }

    fn back(&mut self, port: &mut Self::Port, at: At) -> Result<(), GpuError> {
        self.check(at.unit)?;
        let l = at.layer;
        if !self.cfg[l].host_leg() {
            return Ok(());
        }
        let r = row_of(self.s, self.rows, at.unit)?;
        let dest = if l + 1 == self.cfg.len() {
            self.heads[at.unit].input_mut()
        } else {
            &mut r.x
        };
        shared_routed::back_of(self.gpu, self.ffn, port.hybrid(), (l, at.unit), dest)?;
        self.tap(l)?;
        port.hybrid().row_enqueued(l, at.unit)
    }

    fn end(&mut self, unit: usize) -> Result<(), GpuError> {
        self.check(unit)?;
        self.heads[unit].enqueue(self.gpu, self.w)
    }
}
