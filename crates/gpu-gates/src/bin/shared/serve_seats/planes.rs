//! The seat side of a plane state ([`bloomery_gpu::checkpoint::planes`]): the
//! prompt-cache methods of a seat whose body is a [`PlaneBody`] and whose
//! session holds no draft beside it. The seat gives its model and, for the
//! copy off the request path, its save lane ([`seq_lane`]); this owner turns
//! them into the server's saved state ([`Cached`], side `()`), begins its
//! copy ([`begin`]) and puts it back ([`put_back`]).
//!
//! A state is of one body's kind ([`PlaneSeq`]); a state another seat took is
//! refused by name before the model is reset.

use std::sync::Arc;

use bloomery_gpu::GpuModel;
use bloomery_gpu::checkpoint::planes::{
    PlaneBody, PlaneSeq, seq_resume, seq_resume_prefix, seq_save, seq_save_async, seq_save_bytes,
};
use bloomery_gpu::checkpoint::saved::SaveLane;
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::bind::{Cached, Landing, PutBack, begin, put_back};
use serve::{Pending, Saved};

/// The model a state is put back on: the selected sequence is the model's
/// live one, reset before the put.
struct Selected<'m, B: PlaneBody>(&'m mut GpuModel<B>);

impl<B: PlaneBody> PutBack for Selected<'_, B> {
    type Body = PlaneSeq<B>;
    type Side = ();

    fn takes(&self, _side: &()) -> Result<(), GateError> {
        Ok(())
    }

    fn reset_selected(&mut self) -> Result<(), GateError> {
        Ok(self.0.reset()?)
    }

    fn unpark(&mut self, _side: &()) -> Result<(), GateError> {
        Ok(())
    }
}

/// The host bytes a snapshot would hold ([`seq_save_bytes`]), for
/// `Seat::snapshot_size`.
pub fn snapshot_size<B: PlaneBody>(m: &mut GpuModel<B>) -> Result<Option<u64>, GateError> {
    Ok(Some(seq_save_bytes(m)?))
}

/// The sequence state ([`seq_save`]) as the prompt cache holds it, taken on
/// the request path.
pub fn snapshot<B: PlaneBody>(m: &mut GpuModel<B>) -> Result<Arc<dyn Saved>, GateError> {
    Ok(Arc::new(Cached::<PlaneSeq<B>, ()>::landed(
        seq_save(m)?,
        (),
    )))
}

/// The state's copy begun on `lane` ([`seq_save_async`]); a staging pool with
/// no room is the server's `NoRoom` ([`begin`]).
pub fn snapshot_begin<B: PlaneBody>(
    m: &mut GpuModel<B>,
    lane: &SaveLane,
) -> Result<Pending, GateError> {
    let copy = seq_save_async(m, lane)?
        .map(|(state, pending)| (state, Arc::new(pending) as Arc<dyn Landing>));
    begin(copy, || Ok(()))
}

/// `state` put back ([`seq_resume`]) after the model's reset; `whose` names
/// the seat's body in the refusal of another seat's state.
pub fn resume<B: PlaneBody>(
    m: &mut GpuModel<B>,
    state: &dyn Saved,
    whose: &str,
) -> Result<(), GateError> {
    put_back(&mut Selected(m), state, whose, |g, body| {
        seq_resume(g.0, body).map(|()| body.positions())
    })
    .map(drop)
}

/// The first `n` positions of `state` that it can be cut to put back after
/// the model's reset ([`seq_resume_prefix`]); the positions the model holds
/// after.
pub fn resume_prefix<B: PlaneBody>(
    m: &mut GpuModel<B>,
    state: &dyn Saved,
    n: usize,
    whose: &str,
) -> Result<usize, GateError> {
    let n = u32::try_from(n).unwrap_or(u32::MAX);
    put_back(&mut Selected(m), state, whose, |g, body| {
        seq_resume_prefix(g.0, body, n)
    })
}
