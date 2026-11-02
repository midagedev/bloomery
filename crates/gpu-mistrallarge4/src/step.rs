//! One token's walk: each layer's attention, then the dense lead's block whole
//! in its front or a routed layer's front, shadow and back around its host
//! leg, then the softcapped head. A gate's span runs only its layers (the
//! walk passes the others, which then have no host leg) and the head only
//! when the span reaches the model's end.
//!
//! A verify's rows walk the same program one layer apart, a unit a row
//! ([`walk`], `crate::body::rows`): each part binds its row's buffers first,
//! and the routed block's parts run on the row's own step buffers and host
//! boundary row, so every row runs the step's launches over buffers of its
//! own.

use bloomery_gpu::arch::hybrid::{Acts, GqaCx};
use bloomery_gpu::draft_taps::DraftTaps;
use bloomery_gpu::head::Head;
use bloomery_gpu::host::StepLeg;
use bloomery_gpu::host::run::HostRun;
use bloomery_gpu::routed_block::{RouteIn, RouterEntry};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError, copy_row};
use bloomery_gpu_deepseek41::shared_routed::{back_of, front_of, shadow_of};
use cuda_core::DeviceBuffer;
use model::arch::names;
use runtime::layer::Layer as LayerKind;
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};

use crate::body::rows::RowSet;
use crate::body::{Body, Forced, Kernels, LayerStore, Perm, STEP, Scratch, Shape, Taps, attention};
use crate::ffn::{self, Routed};
use crate::tensors::Layer;

/// Enqueue the step's chain of `body` over `w`, the head's last.
pub(crate) fn enqueue(
    gpu: &Gpu,
    w: &Weights,
    body: &mut Body,
    head: &mut Head,
) -> Result<(), GpuError> {
    walk(gpu, w, body, std::slice::from_mut(head))
}

/// Enqueue the walk of `heads.len()` rows of `body` over `w`, one layer apart
/// (`runtime::sched`'s point `(rows, 1, Step)`), row `r`'s head `heads[r]`:
/// one row is the step, two a verify's pass, whose rows' inputs
/// `Rows::plan_rows` wrote. Row 0's buffers are bound again before the walk
/// returns, whatever it returns. A pass of several rows with the taps armed
/// or a gate's forced list standing is refused by name, nothing enqueued: the
/// taps copy the step's one row and the forced list stands for its one pick.
pub(crate) fn walk(
    gpu: &Gpu,
    w: &Weights,
    body: &mut Body,
    heads: &mut [Head],
) -> Result<(), GpuError> {
    let units = heads.len();
    if units > 1 && body.taps.is_some() {
        return Err(GpuError::shape(
            STEP,
            format!("a walk of {units} rows with the taps armed: they copy the step's one row"),
        ));
    }
    if units > 1 && body.forced.is_some() {
        return Err(GpuError::shape(
            STEP,
            format!(
                "a walk of {units} rows with a forced selection list: it stands for the step's one pick"
            ),
        ));
    }
    let Body {
        sh,
        layers,
        cfg,
        stores,
        k,
        acts,
        perm,
        s,
        x,
        taps,
        forced,
        feat,
        ffn,
        shell,
        pass_rows,
        ..
    } = body;
    let n_layer = cfg.len();
    let mut port = StepLeg::new(gpu.stream(), &mut shell.hybrid);
    let mut program = Step {
        gpu,
        w,
        sh: *sh,
        layers,
        cfg,
        stores,
        k,
        acts,
        perm,
        s,
        x,
        ffn,
        taps: taps.as_mut(),
        forced: forced.as_ref(),
        feat: feat.as_mut().map(|f| &mut f.step),
        heads,
        rows: pass_rows,
    };
    let walked = sched::walk(
        Overlap {
            units,
            cols: 1,
            port: PortKind::Step,
        },
        n_layer,
        &mut port,
        &mut program,
    );
    let bound = program.bind(0);
    walked.and(bound)
}

struct Step<'a> {
    gpu: &'a Gpu,
    w: &'a Weights,
    sh: Shape,
    layers: &'a [Layer],
    cfg: &'a [LayerKind],
    stores: &'a mut [LayerStore],
    k: &'a Kernels,
    acts: &'a mut Acts,
    perm: &'a Perm,
    s: &'a mut Scratch,
    x: &'a mut DeviceBuffer<f32>,
    ffn: &'a mut Routed,
    taps: Option<&'a mut Taps>,
    /// A gate's forced list, which stands for its layer's pick (a one-row walk's only).
    forced: Option<&'a Forced>,
    /// The feature tap's step and verify rows, when a draft attached it.
    feat: Option<&'a mut DraftTaps>,
    /// A head a row: the step's one, a verify's two.
    heads: &'a mut [Head],
    /// The rows past row 0 a verify binds in its place.
    rows: &'a mut RowSet,
}

impl Step<'_> {
    /// Bind unit `unit`'s row buffers in the program's fields
    /// ([`RowSet::bind`]); a unit past the walk's heads is refused by name.
    fn bind(&mut self, unit: usize) -> Result<(), GpuError> {
        if unit >= self.heads.len() {
            return Err(GpuError::shape(
                STEP,
                format!("unit {unit} of a walk of {} rows", self.heads.len()),
            ));
        }
        self.rows.bind(unit, self.s, self.x, self.stores)
    }

    /// Whether the chain runs layer `l` (the taps' span, every layer without
    /// one).
    fn runs(&self, l: usize) -> bool {
        self.taps.as_deref().is_none_or(|t| t.runs(l))
    }

    /// The row of `unit` entering layer `l` (`unit`'s running row, bound now; the head's input
    /// for the layer count) copied into the feature tap's row `unit` at the slot of `l`, when
    /// `l` is tapped. A copy node of the chain, captured with it.
    fn enter(&mut self, l: usize, unit: usize) -> Result<(), GpuError> {
        let Some(tap) = self.feat.as_deref_mut() else {
            return Ok(());
        };
        let Some(slot) = tap.slot(l) else {
            return Ok(());
        };
        let stream = self.gpu.stream();
        if l == self.sh.n_layer {
            tap.enter_at(stream, slot, self.heads[unit].input(), (unit, 1))
        } else {
            tap.enter_at(stream, slot, &*self.x, (unit, 1))
        }
    }

    /// Layer `l`'s attention half and its residual add, into `ffn_inp`: the
    /// gate's row ahead of the layer first, when one replaces the running
    /// row there.
    fn attend(&mut self, l: usize) -> Result<(), GpuError> {
        let gpu = self.gpu;
        let stream = gpu.stream();
        let sh = self.sh;
        if let Some((_, row)) = self
            .taps
            .as_deref()
            .and_then(|t| t.inject.iter().find(|(at, _)| *at == l))
        {
            copy_row(stream, row, self.x, sh.hidden)?;
        }
        let cx = GqaCx {
            what: STEP,
            gpu,
            w: self.w,
            kgemv: &self.k.kgemv,
            g32: &self.k.g32,
            flash: &self.k.flash_gqa,
            mma: false,
            out_gate: None,
            hidden: sh.hidden,
            eps: sh.eps,
            n_kv: 1,
            ctx: sh.ctx,
        };
        attention(
            gpu,
            self.w,
            self.k,
            sh,
            &cx,
            &self.layers[l],
            l,
            self.stores,
            self.acts,
            self.perm,
            self.s,
            self.x,
            self.forced.filter(|f| f.layer == l).map(|f| &f.list),
            gpu.layer_sink(l)?,
        )?;
        gpu.elem()
            .enqueue_add(stream, self.x, &self.s.r, sh.hidden, &mut self.s.ffn_inp)?;
        // The attention's norm row: the dense lead's own `rms_norm` reuses
        // the same scratch, so its tap copies before the block runs.
        if let Some(t) = self.taps.as_deref_mut() {
            t.layers[l].snap_attention(stream, self.s, sh)?;
        }
        Ok(())
    }

    /// Layer `l`'s compared rows once its output is written: the attention
    /// side's, the residual and the output row.
    fn snap(&mut self, l: usize) -> Result<(), GpuError> {
        let Some(t) = self.taps.as_deref_mut() else {
            return Ok(());
        };
        let a = &self.layers[l];
        let list = self.stores[a.kind.indexer].list();
        let cache = (a.kind.cache == l)
            .then(|| self.stores[l].latent())
            .flatten();
        let out = if l + 1 == self.sh.n_layer {
            self.heads[0].input()
        } else {
            &*self.x
        };
        t.layers[l].snap(self.gpu.stream(), self.s, out, list, cache, self.sh)
    }

    /// A routed layer's routing for its tap: the taps' router over the
    /// step's normed row (the step's own launch, the same bits), and the ids
    /// the step picked.
    fn snap_route(&mut self, normed: &DeviceBuffer<f32>, l: usize) -> Result<(), GpuError> {
        let Some(t) = self.taps.as_deref_mut() else {
            return Ok(());
        };
        let stream = self.gpu.stream();
        let r = RouteIn {
            w: self.w.f32_tensor(STEP, &names::ffn_gate_inp(l))?,
            bias: None,
            scale: 1.0,
            fault: self.gpu.layer_sink(l)?,
        };
        t.router.route(stream, normed, &r, &mut t.route)?;
        t.layers[l].snap_route(stream, &t.route, self.ffn.route_ids())
    }
}

impl<'a> LayerProgram for Step<'a> {
    type Port = StepLeg<'a, HostRun>;

    fn begin(&mut self, unit: usize) -> Result<(), GpuError> {
        self.bind(unit)
    }

    fn host_leg(&self, at: At) -> bool {
        self.cfg[at.layer].host_leg() && self.runs(at.layer)
    }

    /// The attention, then the dense lead's block whole or a routed layer's
    /// block up to its go.
    fn front(&mut self, port: &mut Self::Port, at: At) -> Result<(), GpuError> {
        self.bind(at.unit)?;
        let l = at.layer;
        if !self.runs(l) {
            return Ok(());
        }
        self.enter(l, at.unit)?;
        self.attend(l)?;
        if self.cfg[l].host_leg() {
            front_of(
                self.gpu,
                self.w,
                self.ffn,
                &self.s.ffn_inp,
                (port.hybrid(), l, at.unit),
            )?;
            return self.snap_route(port.hybrid().boundary().normed(), l);
        }
        // The layer's output: the row's head's input on the model's last
        // layer, the running row on every other.
        let dest = if l + 1 == self.sh.n_layer {
            self.heads[at.unit].input_mut()
        } else {
            &mut *self.x
        };
        ffn::dense(
            self.gpu,
            self.w,
            self.ffn,
            l,
            (&self.s.ffn_inp, &mut self.s.xn, &mut self.s.ffn_out),
            dest,
        )?;
        if let Some(t) = self.taps.as_deref_mut() {
            t.layers[l].snap_dense(self.gpu.stream(), self.s, self.sh)?;
        }
        self.snap(l)
    }

    /// The shared expert and the card's slots under the host leg.
    fn shadow(&mut self, port: &mut Self::Port, at: At) -> Result<(), GpuError> {
        self.bind(at.unit)?;
        if !self.host_leg(at) {
            return Ok(());
        }
        shadow_of(
            self.gpu,
            self.w,
            self.ffn,
            &self.s.ffn_inp,
            (port.hybrid().boundary(), at.layer, at.unit),
        )
    }

    /// The wait and the join into the layer's output, then the taps of the
    /// join's terms, then the host's serve.
    fn back(&mut self, port: &mut Self::Port, at: At) -> Result<(), GpuError> {
        self.bind(at.unit)?;
        let l = at.layer;
        if !self.host_leg(at) {
            return Ok(());
        }
        let gpu = self.gpu;
        let dest = if l + 1 == self.sh.n_layer {
            self.heads[at.unit].input_mut()
        } else {
            &mut *self.x
        };
        back_of(gpu, self.ffn, port.hybrid(), (l, at.unit), dest)?;
        if let Some(t) = self.taps.as_deref_mut() {
            let terms = self.ffn.join_terms(port.hybrid().boundary(), l)?;
            t.layers[l].snap_join(gpu, &terms, self.sh)?;
        }
        self.snap(l)?;
        port.hybrid().row_enqueued(l, at.unit)
    }

    /// The head, with the file's softcap between the projection and the
    /// argmax, when the walk reached the model's end.
    fn end(&mut self, unit: usize) -> Result<(), GpuError> {
        self.bind(unit)?;
        if !self.runs(self.sh.n_layer - 1) {
            return Ok(());
        }
        self.enter(self.sh.n_layer, unit)?;
        let (gpu, softcap) = (self.gpu, self.sh.softcap);
        let head = &mut self.heads[unit];
        let n = head.n_vocab() * head.m();
        // The projection and the argmax are the head's, of the file's type;
        // the softcap is the one launch between them.
        head.enqueue_with_epilogue(gpu, self.w, |logits| {
            // 0 is no cap: the reference graph caps only on a nonzero bound.
            if softcap > 0.0 {
                gpu.elem()
                    .enqueue_softcap_rows(gpu.stream(), logits, n, softcap)
            } else {
                Ok(())
            }
        })
    }
}
