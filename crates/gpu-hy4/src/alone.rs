//! A block of the step run alone on rows the caller hands in: the end-to-end gate holds each block
//! on the vendor's own inputs this way, so a block's distance from the vendor's node is its own
//! rounding and not what the layers before it left. Each run leaves the stores and the scratch
//! where its launches put them: the model is reset before its next step (a step refuses until
//! then).

use bloomery_gpu::hc_ident::{HC, HeadArgs};
use bloomery_gpu::routed_block::RouterEntry;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use model::arch::hy4::names::{self, Sub};

use crate::body::{Body, Router, Store};
use crate::step;

/// What the runs' refusals name.
const ALONE: &str = "hy4::Body alone";

/// A routed block's picks: the expert ids in rank order and their weights.
pub type Picks = (Vec<u32>, Vec<f32>);

/// A sparse layer's selection: the cache rows its flash reads, in list order, and every pool's
/// score (the open pool's slot and those past it are not scores).
pub type Selection = (Vec<u32>, Vec<f32>);

/// The identity mix of one sub-layer run alone: the normed fold its sub-layer reads, and the
/// streams after its post.
pub struct MixAlone {
    pub normed: Vec<f32>,
    pub streams: Vec<f32>,
}

impl Body {
    /// Layer `l`'s attention run alone over `rows`, the vendor's normed rows (`attn_norm`) of
    /// positions `0..rows.len()`: the output at the last position, after the sandwich norm, and on
    /// a sparse layer the selector's own rows and pool scores there ([`Body::selection`]). Every
    /// position's row is appended to the layer's store as a step appends it. With `rows_read`, the
    /// flash at the last position reads those cache rows in place of the selector's list (the
    /// vendor's selection, so a pool the two engines rank on either side of a tie does not move
    /// the output).
    ///
    /// # Errors
    /// [`GpuError`] for a layer past the model, an empty or too long `rows`, a row not `hidden`
    /// wide, `rows_read` on a window layer or longer than the list, or from the launches.
    pub fn attention_alone(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        l: usize,
        rows: &[Vec<f32>],
        rows_read: Option<&[u32]>,
    ) -> Result<(Vec<f32>, Option<Selection>), GpuError> {
        self.alone_layer(l)?;
        if rows.is_empty() || rows.len() > self.sh.ctx {
            return Err(GpuError::shape(
                ALONE,
                format!("{} rows in stores of {}", rows.len(), self.sh.ctx),
            ));
        }
        if rows_read.is_some() && !self.layers[l].dsa {
            return Err(GpuError::shape(
                ALONE,
                format!("layer {l} is a window layer: it reads no list"),
            ));
        }
        let last = rows.len() - 1;
        let mut ours = None;
        for (t, row) in rows.iter().enumerate() {
            let pos =
                u32::try_from(t).map_err(|_| GpuError::shape(ALONE, "a position past u32"))?;
            ours = self.attend(gpu, w, l, (row, pos), (t == last).then_some(rows_read))?;
        }
        Ok((self.s.out.to_host_vec(gpu.stream())?, ours))
    }

    /// Layer `l`'s attention at position `pos` on the normed row `row`, over the stores as they
    /// stand (a prompt batch's, or earlier runs'): the output after the sandwich norm and, on a
    /// sparse layer, the selector's rows and scores there. The position's row is appended to the
    /// layer's store as a step appends it.
    ///
    /// # Errors
    /// [`GpuError`] for a layer past the model, a position past the context, a row not `hidden`
    /// wide, or from the launches.
    pub fn attention_at(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        l: usize,
        at: (&[f32], u32),
    ) -> Result<(Vec<f32>, Option<Selection>), GpuError> {
        self.alone_layer(l)?;
        let ours = self.attend(gpu, w, l, at, Some(None))?;
        Ok((self.s.out.to_host_vec(gpu.stream())?, ours))
    }

    /// One position of layer `l`'s attention: the normed row `row` at `pos` appended to the store,
    /// then the read into `s.out`. With `select` (a run's last position), a sparse layer's
    /// selector rows and scores there, its flash reading the rows `select` names in place of the
    /// selector's list when it names some.
    fn attend(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        l: usize,
        (row, pos): (&[f32], u32),
        select: Option<Option<&[u32]>>,
    ) -> Result<Option<Selection>, GpuError> {
        let stream = gpu.stream();
        self.alone_row(row)?;
        if pos as usize >= self.sh.ctx {
            return Err(GpuError::shape(
                ALONE,
                format!("position {pos} in stores of {}", self.sh.ctx),
            ));
        }
        let fault = gpu.layer_sink(l)?;
        self.write_position(stream, pos)?;
        self.s.xn.copy_from_host(stream, row)?;
        step::attention_keys(gpu, w, self, l, fault)?;
        let mut ours = None;
        if let Some(read) = select
            && self.layers[l].dsa
        {
            ours = Some(self.selection(gpu, l)?);
            if let Some(read) = read {
                self.read_rows(gpu, l, read)?;
            }
        }
        step::attention_read(gpu, w, self, l, fault)?;
        Ok(ours)
    }

    /// Sparse layer `l`'s list replaced by `read`, the cache rows its flash reads next.
    fn read_rows(&mut self, gpu: &Gpu, l: usize, read: &[u32]) -> Result<(), GpuError> {
        let stream = gpu.stream();
        let top_k = self.sh.top_k;
        let Some(Store::Dsa(d)) = self.stores.get_mut(l) else {
            return Err(GpuError::shape(
                ALONE,
                format!("layer {l} is not a sparse layer"),
            ));
        };
        if read.is_empty() || read.len() > top_k {
            return Err(GpuError::shape(
                ALONE,
                format!("{} rows to read, the list holds 1 to {top_k}", read.len()),
            ));
        }
        let mut list = read.to_vec();
        list.resize(top_k, 0);
        d.list.copy_from_host(stream, &list)?;
        let n =
            u32::try_from(read.len()).map_err(|_| GpuError::shape(ALONE, "a count past u32"))?;
        d.n_sel.copy_from_host(stream, &[n])?;
        Ok(())
    }

    /// Layer `l`'s feed-forward block run alone on the normed row `xn` (the vendor's `ffn_norm`):
    /// its output after the sandwich norm, and a routed block's picks. Touches no store.
    ///
    /// # Errors
    /// [`GpuError`] for a layer past the model, a row not `hidden` wide, or from the launches.
    pub fn ffn_alone(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        l: usize,
        xn: &[f32],
    ) -> Result<(Vec<f32>, Option<Picks>), GpuError> {
        let stream = gpu.stream();
        self.alone_layer(l)?;
        self.alone_row(xn)?;
        let fault = gpu.layer_sink(l)?;
        self.s.xn.copy_from_host(stream, xn)?;
        step::ffn(gpu, w, self, l, fault)?;
        let out = self.s.out.to_host_vec(stream)?;
        let picks = match &self.routes[l] {
            Some(r) => Some((
                Router::out_ids(r).to_host_vec(stream)?,
                Router::out_weights(r).to_host_vec(stream)?,
            )),
            None => None,
        };
        Ok((out, picks))
    }

    /// Sub-layer `sub` of layer `l`'s identity mix run alone: the pre over `streams` (the four
    /// streams the sub-layer reads) and the norm its sub-layer reads (`attn_norm`, `ffn_norm`);
    /// then the post of the sub-layer's output `out` onto the same streams.
    ///
    /// # Errors
    /// [`GpuError`] for a layer past the model, `streams` not four rows of `hidden` or `out` not
    /// one, or from the launches.
    pub fn mix_alone(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        (l, sub): (usize, Sub),
        streams: &[f32],
        out: &[f32],
    ) -> Result<MixAlone, GpuError> {
        let stream = gpu.stream();
        self.alone_layer(l)?;
        self.alone_streams(streams)?;
        self.alone_row(out)?;
        let fault = gpu.layer_sink(l)?;
        self.s.hs[0].copy_from_host(stream, streams)?;
        step::pre(gpu, w, self, (l, sub), 0, fault)?;
        let gain = match sub {
            Sub::Attn => names::attn_norm(l),
            Sub::Ffn => names::ffn_norm(l),
        };
        step::norm(gpu, w, self, &gain)?;
        let normed = self.s.xn.to_host_vec(stream)?;
        self.s.out.copy_from_host(stream, out)?;
        let mut cur = 0;
        step::post(gpu, self, &mut cur)?;
        Ok(MixAlone {
            normed,
            streams: self.s.hs[cur].to_host_vec(stream)?,
        })
    }

    /// Layer `l`'s engram site run alone over `streams`, with the rows and mask the last step's
    /// refresh gathered: the streams after the site's delta.
    ///
    /// # Errors
    /// [`GpuError`] for a layer past the model or with no site, `streams` not four rows of
    /// `hidden`, or from the launches.
    pub fn engram_alone(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        l: usize,
        streams: &[f32],
    ) -> Result<Vec<f32>, GpuError> {
        let stream = gpu.stream();
        self.alone_layer(l)?;
        if self.layers[l].engram.is_none() {
            return Err(GpuError::shape(
                ALONE,
                format!("layer {l} has no engram site"),
            ));
        }
        self.alone_streams(streams)?;
        let fault = gpu.layer_sink(l)?;
        self.s.hs[0].copy_from_host(stream, streams)?;
        let mut cur = 0;
        step::engram(gpu, w, self, l, &mut cur, fault)?;
        Ok(self.s.hs[cur].to_host_vec(stream)?)
    }

    /// The head's fold of `streams` (the last layer's four output streams) alone: the row the
    /// output head reads.
    ///
    /// # Errors
    /// [`GpuError`] for `streams` not four rows of `hidden`, or from the launch.
    pub fn head_fold_alone(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        streams: &[f32],
    ) -> Result<Vec<f32>, GpuError> {
        let stream = gpu.stream();
        self.alone_streams(streams)?;
        self.held = u32::MAX;
        self.s.hs[0].copy_from_host(stream, streams)?;
        let sh = self.sh;
        self.k.hc.enqueue_head(
            stream,
            HeadArgs {
                x: &self.s.hs[0],
                w: w.f32_buf(ALONE, names::OUTPUT_HC_FN)?,
                scale: w.f32_buf(ALONE, names::OUTPUT_HC_SCALE)?,
                base: w.f32_buf(ALONE, names::OUTPUT_HC_BASE)?,
                eps_norm: sh.eps,
                eps_hc: sh.hc_eps,
                n: sh.hidden,
                m: 1,
                y: &mut self.s.y,
                fault: gpu.unlabelled_sink(),
            },
        )?;
        Ok(self.s.y.to_host_vec(stream)?)
    }

    /// Sparse layer `l`'s selection at the position its last run stood at: the cache rows the
    /// flash read (the kept pools' rows and the open pool's, in list order), and every pool's
    /// score the selector ranked (the open pool's slot and those past it are not scores).
    ///
    /// # Errors
    /// [`GpuError`] for a layer past the model or a window layer, or from the readback.
    pub fn selection(&self, gpu: &Gpu, l: usize) -> Result<Selection, GpuError> {
        let stream = gpu.stream();
        match self.stores.get(l) {
            Some(Store::Dsa(d)) => {
                let n = d.n_sel.to_host_vec(stream)?[0] as usize;
                let list = d.list.to_host_vec(stream)?;
                let rows = list.get(..n).ok_or_else(|| {
                    GpuError::shape(
                        ALONE,
                        format!("a count of {n} past the list's {}", list.len()),
                    )
                })?;
                Ok((rows.to_vec(), d.scores.to_host_vec(stream)?))
            }
            _ => Err(GpuError::shape(
                ALONE,
                format!("layer {l} is not a sparse layer of the model"),
            )),
        }
    }

    /// Refuses a layer past the model and marks the stores as left by an alone run: the next step
    /// is refused until a reset.
    fn alone_layer(&mut self, l: usize) -> Result<(), GpuError> {
        if l >= self.layers.len() {
            return Err(GpuError::shape(
                ALONE,
                format!("layer {l} of {}", self.layers.len()),
            ));
        }
        self.held = u32::MAX;
        Ok(())
    }

    fn alone_row(&self, row: &[f32]) -> Result<(), GpuError> {
        if row.len() != self.sh.hidden {
            return Err(GpuError::shape(
                ALONE,
                format!(
                    "a row of {} values, the model's is {}",
                    row.len(),
                    self.sh.hidden
                ),
            ));
        }
        Ok(())
    }

    fn alone_streams(&self, streams: &[f32]) -> Result<(), GpuError> {
        if streams.len() != HC * self.sh.hidden {
            return Err(GpuError::shape(
                ALONE,
                format!(
                    "streams of {} values, the model's four are {}",
                    streams.len(),
                    HC * self.sh.hidden
                ),
            ));
        }
        Ok(())
    }
}
