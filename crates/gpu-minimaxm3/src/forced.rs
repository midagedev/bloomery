//! One layer alone at one position on the rows the caller gives ([`Body::forced_row`]): the
//! teacher-forced arm a gate runs on the oracle's own layer inputs, and the row it holds against
//! that oracle's taps.
//!
//! The row runs the step's launches of that layer in the step's order, eager on the engine stream,
//! and reads back what a gate compares: the residual after the attention sub-layer, the
//! feed-forward block's increment, the routed block's picks and the layer's output. The layer's
//! store holds what the earlier positions of the same run wrote: the caller resets the model
//! before position 0 of each layer's run.
//!
//! With `ffn_x`, the feed-forward sub-layer reads those rows in place of the attention
//! sub-layer's (its residual too): the oracle's own feed-forward input, so the router reads the
//! oracle's own normed row and a difference in its picks is a difference in the router's
//! arithmetic, not in what it read.

use bloomery_gpu::arch::hybrid::GqaCx;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use runtime::layer::FfnKind;

use crate::body::Body;
use crate::ffn;
use crate::step::{AttendParts, FLASH_MMA, attend_layer};

/// What the errors name.
const WHAT: &str = "minimaxm3 forced_row";

/// A routed block's picks at one position.
#[derive(Clone, Debug)]
pub struct ForcedRoute {
    /// The router's picks, in rank order.
    pub ids: Vec<u32>,
    /// The layer's selection bias, the values the router adds to the scores
    /// before it ranks them: the ranking the oracle's picks are read at.
    pub bias: Vec<f32>,
}

/// One layer at one position, read back.
#[derive(Clone, Debug)]
pub struct ForcedRow {
    /// The residual after the attention sub-layer: the feed-forward block's input.
    pub ffn_in: Vec<f32>,
    /// The feed-forward block's increment: its routed sum and the shared expert's, the layer's
    /// output less its input.
    pub ffn_out: Vec<f32>,
    /// A routed block's picks and its selection bias; `None` for a dense block.
    pub route: Option<ForcedRoute>,
    /// The layer's output.
    pub out: Vec<f32>,
}

impl Body {
    /// Layer `l` alone at position `pos` on the rows `x` (`n_embd` values), eager and blocking.
    /// With `ffn_x`, the feed-forward sub-layer reads those rows in place of the attention
    /// sub-layer's. Refused by name: a layer or a position outside the load, rows of another
    /// length. A fault a launch raised is the call's error.
    pub fn forced_row(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        l: usize,
        pos: u32,
        x: &[f32],
        ffn_x: Option<&[f32]>,
    ) -> Result<ForcedRow, GpuError> {
        let ctx = self.ctx();
        let n = self.dims.embd;
        if l >= self.cfg.len() {
            return Err(shape(format!(
                "layer {l} past the {} loaded",
                self.cfg.len()
            )));
        }
        if pos as usize >= ctx {
            return Err(shape(format!("position {pos} in stores of {ctx}")));
        }
        if x.len() != n || ffn_x.is_some_and(|f| f.len() != n) {
            return Err(shape(format!(
                "rows of {} and {:?} values, want {n}",
                x.len(),
                ffn_x.map(<[f32]>::len)
            )));
        }
        let stream = gpu.stream();
        let Body {
            shell,
            dims,
            cfg,
            stores,
            s,
            k,
            rope,
            shape: msa,
            ffn: routed,
            ..
        } = self;
        // The layer's input and its position: what the step's own refresh writes.
        s.x.copy_from_host(stream, x)?;
        s.pos.copy_from_host(stream, &[pos])?;
        s.n_keys.copy_from_host(stream, &[pos + 1])?;
        let cx = GqaCx {
            what: WHAT,
            gpu,
            w,
            kgemv: &k.kgemv,
            g32: &k.g32,
            flash: &k.flash,
            mma: FLASH_MMA,
            out_gate: None,
            hidden: dims.embd,
            eps: dims.rms_eps,
            n_kv: dims.kv_heads,
            ctx: dims.ctx,
        };
        let parts = AttendParts {
            gpu,
            w,
            k,
            rope,
            msa,
            dims,
            cx: &cx,
        };
        attend_layer(&parts, stores, s, l)?;
        // The feed-forward block's input: the oracle's own rows when it gave them, the attention
        // sub-layer's residual otherwise.
        let ffn_in = match ffn_x {
            Some(f) => {
                s.forced_x.copy_from_host(stream, f)?;
                &s.forced_x
            }
            None => &s.x1,
        };
        // The block's increment is read against the rows it actually read: the oracle's own when
        // it gave them, the attention sub-layer's residual otherwise.
        let ffn_in_row = ffn_in.to_host_vec(stream)?;
        let route = match cfg[l].ffn {
            FfnKind::Dense => {
                ffn::dense(
                    gpu,
                    w,
                    routed,
                    l,
                    (ffn_in, &mut s.xn, &mut s.delta),
                    &mut s.x,
                )?;
                None
            }
            FfnKind::Moe => {
                shell.hybrid.open_step(stream, 1, 1)?;
                ffn::front(gpu, w, routed, ffn_in, (&mut shell.hybrid, l))?;
                ffn::shadow(gpu, w, routed, ffn_in, (shell.hybrid.boundary(), l))?;
                ffn::back(gpu, routed, &shell.hybrid, l, &mut s.x)?;
                shell.hybrid.row_enqueued(l, 0)?;
                let ids = routed.route_ids().to_host_vec(stream)?;
                let bias = routed.route_bias(w, l)?.to_host_vec(stream)?;
                Some(ForcedRoute { ids, bias })
            }
        };
        let out = s.x.to_host_vec(stream)?;
        let ffn_out = out.iter().zip(&ffn_in_row).map(|(o, i)| o - i).collect();
        if let Some(fault) = gpu.fault()? {
            return Err(GpuError::fault(WHAT, fault));
        }
        Ok(ForcedRow {
            ffn_in: ffn_in_row,
            ffn_out,
            route,
            out,
        })
    }
}

/// A shape the forced row refuses, by name.
fn shape(detail: impl Into<String>) -> GpuError {
    GpuError::Shape {
        what: WHAT,
        detail: detail.into(),
    }
}
