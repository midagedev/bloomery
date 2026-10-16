//! The verify path: a load opened with [`GpuModel::open_verifying`] runs a
//! draft's proposed rows as one pass of engine's prompt body
//! ([`Body::enqueue_pass`], every row's head through [`head_rows`]) and
//! keeps the first `1..=m` of them at its commit — the Mamba-2 layers' scan
//! state is the one store a pass overwrites in place, and each layer's
//! lanes ([`Verify`]) hold the state after every row but the pass's last,
//! so a commit of `k < m` rows copies row `k − 1`'s state back, one device
//! copy a layer. The conv ring is keyed by position and the attention K/V
//! planes by row: a position re-run overwrites its own ring slot and plane
//! row before anything reads them, so neither needs a copy.

use super::body::{Body, Layer};
use crate::head::Head;
use crate::hybrid::Chain;
use crate::mamba2::STATE;
use crate::model::{MAX_PASS_ROWS, Rollback, RowHeads, Rows};
use crate::tensor::{Pitched, dtod_2d};
use crate::weights::Weights;
use crate::{Gpu, GpuError, Window};
use cuda_core::{CudaStream, DeviceBuffer};

const WHAT: &str = "nemotron_h_moe::verify";

/// A verify of drafted rows still waiting for its commit: the position of
/// its first row and its rows' ids (its row count is the ids' length).
#[derive(Clone, Debug)]
pub(super) struct Pending {
    pos0: u32,
    ids: Vec<u32>,
}

/// The verifying load's own buffers: one lanes buffer a Mamba-2 layer (none
/// a layer of another kind), `rows − 1` states each, and the verify waiting
/// for its commit.
pub(super) struct Verify {
    /// The most rows a pass of this load keeps lanes for: the count the load
    /// was opened with.
    pub(super) rows: usize,
    /// One buffer a layer, in layer order: a Mamba-2 layer's `rows − 1`
    /// states of `state` f32, `None` a layer of another kind.
    lanes: Vec<Option<DeviceBuffer<f32>>>,
    /// One Mamba-2 layer's scan state, in f32: `d_inner · 128`, the length
    /// of a lanes row and of the state a commit restores.
    state: usize,
    /// The verify waiting for its commit.
    pub(super) pending: Option<Pending>,
}

impl Verify {
    /// The lanes for passes of up to `rows` rows over `layers`, each
    /// Mamba-2 layer `rows − 1` states of `d_inner · [`STATE`]` f32.
    /// Load-time only.
    pub(super) fn new(
        stream: &CudaStream,
        what: &'static str,
        layers: &[Layer],
        rows: usize,
        d_inner: usize,
    ) -> Result<Verify, GpuError> {
        let state = d_inner * STATE;
        let lane = |m: bool| -> Result<Option<DeviceBuffer<f32>>, GpuError> {
            if !m {
                return Ok(None);
            }
            Ok(Some(DeviceBuffer::zeroed(
                stream,
                (rows - 1).checked_mul(state).ok_or_else(|| {
                    GpuError::shape(what, format!("{rows} − 1 rows of {state} f32 pass usize"))
                })?,
            )?))
        };
        let lanes = layers
            .iter()
            .map(|l| lane(matches!(l, Layer::Mamba(_))))
            .collect::<Result<Vec<_>, GpuError>>()?;
        Ok(Verify {
            rows,
            lanes,
            state,
            pending: None,
        })
    }

    /// The device bytes the lanes hold.
    pub(super) fn bytes(&self) -> usize {
        self.lanes
            .iter()
            .flatten()
            .map(DeviceBuffer::num_bytes)
            .sum()
    }

    /// Layer `l`'s lanes buffer; `None` for a layer of another kind.
    pub(super) fn lane(&mut self, l: usize) -> Option<&mut DeviceBuffer<f32>> {
        self.lanes.get_mut(l).and_then(Option::as_mut)
    }
}

impl Body {
    /// The verify waiting for its commit: the position of its first row and
    /// its row count. `None` on a plain load, and when none waits.
    #[must_use]
    pub fn verify_waiting(&self) -> Option<(u32, usize)> {
        self.verify
            .as_ref()
            .and_then(|v| v.pending.as_ref().map(|p| (p.pos0, p.ids.len())))
    }

    /// Refused by name while a verify waits for its commit: its rows' states
    /// sit in lanes the committed scan state does not hold yet.
    pub(super) fn refuse_waiting(&self, what: &'static str, doing: &str) -> Result<(), GpuError> {
        if let Some((pos0, rows)) = self.verify_waiting() {
            return Err(GpuError::shape(
                what,
                format!(
                    "{doing} while the verify of {rows} rows at {pos0} waits for its commit \
                     (GpuModel::rollback keeps 1..={rows})"
                ),
            ));
        }
        Ok(())
    }
}

/// Row `r`'s head over the pass's rows: row `r` of the last layer's output
/// copied into `heads[r]` and the head run — a pass swaps its two residual
/// streams once a layer from `x[0]`, so the last layer's output is
/// `x[layer count % 2]` ([`super::prefill::Prompt::last_x`]), the row the
/// pass's own last-row head reads.
pub(super) fn head_rows(
    gpu: &Gpu,
    w: &Weights,
    x: &DeviceBuffer<f32>,
    hidden: usize,
    heads: &mut [Head],
) -> Result<(), GpuError> {
    let f = size_of::<f32>();
    for (r, head) in heads.iter_mut().enumerate() {
        let row = Window::<f32>::of(x, r * hidden * f, hidden)?;
        head.input_mut()
            .copy_from_device_async(&row, gpu.stream())?;
        head.enqueue(gpu, w)?;
    }
    Ok(())
}

impl Rows for Body {
    const MAX_ROWS: usize = MAX_PASS_ROWS;
    /// No host tier serves a replay; the pass is the prompt body's own
    /// launches at `m` rows.
    const CHAIN: Chain = Chain::Step;
    const HEADS: RowHeads = RowHeads::PerRow;

    /// The pass's word record in one copy ([`Prompt::plan_words`]), the
    /// record the captured pass replays, and the verify recorded as waiting
    /// for its commit. Refused by name while a verify waits, on a plain
    /// load (it cannot take rows back), and past the load's rows.
    fn plan_rows(&mut self, stream: &CudaStream, tokens: &[u32], pos: u32) -> Result<(), GpuError> {
        const WHAT: &str = "nemotron_h_moe::Body::plan_rows";
        self.refuse_waiting(WHAT, "a pass")?;
        let m = tokens.len();
        if !(2..=MAX_PASS_ROWS).contains(&m) {
            return Err(GpuError::shape(
                WHAT,
                format!("a pass of {m} rows; a verify holds 2..={MAX_PASS_ROWS}"),
            ));
        }
        let Some(v) = self.verify.as_mut() else {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a plain load's pass of {m} rows: it cannot take rows back \
                     (GpuModel::open_verifying loads the lanes)"
                ),
            ));
        };
        if m > v.rows {
            return Err(GpuError::shape(
                WHAT,
                format!("a pass of {m} rows; the load's lanes hold 2..={}", v.rows),
            ));
        }
        self.prompt.plan_words(stream, &self.sh, tokens, pos)?;
        v.pending = Some(Pending {
            pos0: pos,
            ids: tokens.to_vec(),
        });
        self.held = pos
            + u32::try_from(m).map_err(|_| GpuError::shape(WHAT, format!("{m} rows past u32")))?;
        Ok(())
    }

    /// The pass the last [`Rows::plan_rows`] planned — engine's
    /// [`Body::enqueue_pass`] over the planned ids with no head of its own,
    /// then every row through its own head ([`head_rows`]). A capture
    /// records the launches only, so the planned record may be another
    /// pass's or none: the replay reads the record the plan before it wrote.
    fn enqueue_rows(&mut self, gpu: &Gpu, w: &Weights, heads: &mut [Head]) -> Result<(), GpuError> {
        const WHAT: &str = "nemotron_h_moe::Body::enqueue_rows";
        let m = heads.len();
        let planned = match self.verify.as_ref().and_then(|v| v.pending.as_ref()) {
            Some(p) if p.ids.len() == m => (p.ids.clone(), p.pos0),
            _ if crate::capturing(gpu.stream())? => (vec![0; m], 0),
            _ => {
                return Err(GpuError::state(
                    WHAT,
                    "a record planned for this pass's rows (plan_rows before the pass)",
                ));
            }
        };
        self.enqueue_pass(gpu, w, (&planned.0, planned.1), None)?;
        head_rows(
            gpu,
            w,
            self.prompt.last_x(self.layers.len()),
            self.sh.hidden,
            heads,
        )
    }
}

impl Body {
    /// The first `kept` rows of the verify waiting: row `kept − 1`'s state
    /// copied back over each Mamba-2 layer's scan state, one device copy a
    /// layer. Never inside a capture.
    fn keep_rows(&mut self, stream: &CudaStream, kept: usize) -> Result<(), GpuError> {
        let Body { layers, verify, .. } = self;
        let v = verify
            .as_mut()
            .ok_or(GpuError::state(WHAT, "a verifying load's lanes"))?;
        let state = v.state;
        let at = (kept - 1) * state * size_of::<f32>();
        let shape = Pitched {
            width: state * size_of::<f32>(),
            rows: 1,
            dev_pitch: state * size_of::<f32>(),
            host_pitch: state * size_of::<f32>(),
        };
        for (l, layer) in layers.iter_mut().enumerate() {
            if let Layer::Mamba(m) = layer
                && let Some(lane) = v.lane(l)
            {
                dtod_2d(stream, (lane, at), (&mut m.store.state, 0), shape)?;
            }
        }
        Ok(())
    }

    /// The positions taken back with no verify waiting: nothing at the
    /// stores' held count, any other position refused by name — the scan
    /// state holds no earlier position but a verifying commit's lanes.
    fn take_back(&mut self, pos: u32) -> Result<(), GpuError> {
        if pos == self.held {
            return Ok(());
        }
        Err(GpuError::shape(
            WHAT,
            format!(
                "back to position {pos}: the recurrent stores hold {} and no copy lies behind \
                 them (reset, or a verifying load's commit)",
                self.held
            ),
        ))
    }
}

impl Rollback for Body {
    /// A verifying load's commit is a device copy, so the cardless rollback
    /// refuses it by name; a plain load moves nothing and keeps the
    /// cardless rule of [`Rollback::rollback_on`].
    fn rollback(&mut self, pos: u32) -> Result<(), GpuError> {
        if self.verify.is_some() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "back to position {pos} without the card: a load that verifies drafted rows \
                     commits by a device copy (GpuModel::rollback)"
                ),
            ));
        }
        self.take_back(pos)
    }

    /// With a verify waiting: its commit — any `1..=m` of its rows kept,
    /// `k < m` by one device copy of row `k − 1`'s state a Mamba-2 layer,
    /// `k = m` by nothing (the state after the last row is the state's), the
    /// verify then no longer waiting. With none waiting: nothing at the
    /// stores' held count, any other position refused by name before
    /// anything moves.
    fn rollback_on(&mut self, gpu: &Gpu, pos: u32) -> Result<(), GpuError> {
        let Some((pos0, rows)) = self.verify_waiting() else {
            return self.take_back(pos);
        };
        if pos <= pos0 || pos > pos0 + u32::try_from(rows).unwrap_or(u32::MAX) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "back to position {pos}: the verify of {rows} rows at {pos0} keeps 1..={rows} \
                     (row 0 always), and the scan state holds no earlier position"
                ),
            ));
        }
        let kept = (pos - pos0) as usize;
        if kept < rows {
            self.keep_rows(gpu.stream(), kept)?;
        }
        if let Some(v) = self.verify.as_mut() {
            v.pending = None;
        }
        self.held = pos;
        Ok(())
    }
}
