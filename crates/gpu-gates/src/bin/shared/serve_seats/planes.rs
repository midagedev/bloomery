//! The seat side of a plane state ([`bloomery_gpu::checkpoint::planes`]): the
//! prompt-cache methods of a seat whose body is a [`PlaneBody`] and whose
//! session holds no draft beside it. The seat gives its model and, for the
//! copy off the request path, its save lane ([`seq_lane`]); this owner turns
//! them into the server's saved state ([`Cached`], side `()`), begins its
//! copy ([`begin`]) and puts it back ([`put_back`]).
//!
//! A state is of one body's kind ([`PlaneSeq`]); a state another seat took is
//! refused by name before the model is reset.
//!
//! A slot file (`Seat::save_state`, `Seat::restore_state`) is the state in the
//! byte form of `bloomery_gpu::checkpoint::bytes` ([`save_state`],
//! [`restore_state`]): a refusal by name is the request's, a failure of the
//! stream the file's, any other the engine's.

use std::io::{Read, Write};
use std::sync::Arc;

use bloomery_gpu::GpuModel;
use bloomery_gpu::checkpoint::bytes::{Put, Take};
use bloomery_gpu::checkpoint::planes::{
    PlaneBody, PlaneSeq, seq_put, seq_restore, seq_resume, seq_resume_prefix, seq_save,
    seq_save_async, seq_save_bytes,
};
use bloomery_gpu::checkpoint::saved::SaveLane;
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::bind::{Cached, Landing, PutBack, begin, byte_failure, put_back};
use serve::{Pending, Saved, StateError};

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

/// The selected sequence's state written to `out` in the byte form `PlaneSeq::put` writes
/// ([`seq_put`]: copied from the card a bounded chunk at a time, no state held on the host), the
/// model unchanged; the bytes written.
pub fn save_state<B: PlaneBody>(
    m: &mut GpuModel<B>,
    out: &mut dyn Write,
) -> Result<u64, StateError> {
    let mut w = Put::new(out);
    seq_put(m, &mut w).map_err(byte_failure)?;
    w.flush().map_err(byte_failure)?;
    Ok(w.bytes())
}

/// The state [`save_state`] wrote, read from `input` and put into the selected sequence as it is
/// read ([`seq_restore`]): refused by name, nothing on the card touched, for a stream that is not
/// such a state of this load in its header or its layout against the model's stores; a stream that
/// ends or fails after them, or goes on past the state, leaves the model reset. The positions
/// restored.
pub fn restore_state<B: PlaneBody>(
    m: &mut GpuModel<B>,
    input: &mut dyn Read,
) -> Result<usize, StateError> {
    let mut r = Take::new(input);
    let n = seq_restore(m, &mut r).map_err(byte_failure)?;
    Ok(n as usize)
}
