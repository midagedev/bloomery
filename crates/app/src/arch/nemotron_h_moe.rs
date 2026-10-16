//! The Nemotron 3.5 Bolt session: [`Body`] behind the session's traits —
//! its prompt schedule (the batched pass call, [`GpuModel::prefill`]) and
//! its keep rule: the Mamba-2 scan state holds no position behind the
//! model's but a verifying commit's lanes, so a cut keeps the empty model,
//! the model's position, and — while a verify waits for its commit — its
//! rows. A load opened with
//! [`GpuModel::open_verifying`](bloomery_gpu::GpuModel::open_verifying) is
//! a [`Verify`](runtime::Verify) target through [`Session::from_model`].

use bloomery_gpu::GpuError;
use bloomery_gpu::NemotronHMoeModel;
use bloomery_gpu::arch::nemotron_h_moe::Body;
use runtime::seqstate::{Kept, Why};

use crate::{Keep, Prompt};

const WHAT: &str = "nemotron_h_moe session";

impl Prompt for Body {
    /// The batched pass call ([`GpuModel::prefill`]): passes of the most ids
    /// the prompt buffers hold, the argmax after the last id.
    fn prompt(m: &mut NemotronHMoeModel, ids: &[u32]) -> Result<u32, GpuError> {
        m.prefill(ids)
    }
}

impl Keep for Body {
    /// [`Keep::kept`]'s position: the model's, the empty model's, and any
    /// position past the first of the verify waiting for its commit.
    fn keepable(m: &NemotronHMoeModel, n: u32) -> u32 {
        <Body as Keep>::kept(m, n).at
    }

    /// What a cut keeps and why: everything at the model's position, the
    /// empty model at 0, a waiting verify's rows (its commit, by the lanes'
    /// copy), nothing anywhere else — the scan state holds no earlier
    /// position but a verifying commit's lanes.
    fn kept(m: &NemotronHMoeModel, n: u32) -> Kept {
        let pos = m.pos();
        let waiting = m.body(WHAT).ok().and_then(|b| b.verify_waiting());
        let at = match (n, waiting) {
            (0, _) => 0,
            (n, Some((pos0, rows))) if n > pos0 => n.min(pos0 + rows as u32),
            (n, _) if n >= pos => pos,
            _ => 0,
        };
        let why = if at == pos && n > 0 {
            Why::Current
        } else if n == 0 {
            Why::Empty
        } else {
            Why::Rule
        };
        Kept {
            asked: n,
            held: pos,
            at,
            why,
        }
    }

    /// Back to empty at 0 — a reset; nothing taken back at the model's
    /// position with no verify waiting; else the model's rollback
    /// ([`Rollback`](bloomery_gpu::model::Rollback)): a waiting verify's
    /// kept rows, any other position refused by name.
    fn cut(m: &mut NemotronHMoeModel, n: u32) -> Result<(), GpuError> {
        let waiting = m.body(WHAT).ok().and_then(|b| b.verify_waiting());
        match (n, waiting) {
            (0, _) => m.reset(),
            (n, None) if n == m.pos() => Ok(()),
            (n, _) => m.rollback(n),
        }
    }
}
