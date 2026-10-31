//! A Gated DeltaNet layer's launches bound to its stores, over the rows of
//! several sequences in one launch each (`crate::seq_rows`): the pass of
//! resident slots of any body whose layers keep a delta store
//! ([`RecStore`]). The conv runs over each sequence's ring; the delta step
//! over each sequence's state, in place (`gdn_delta`) or, a verify, recorded
//! into each sequence's record after its carry (`gdn_delta_recovery`). Each
//! sequence's rows are the launches of its rows alone, bit for bit, so a
//! pass of `n` sequences runs these two launches once where one launch a
//! sequence ran `2·n`.

use std::ops::Range;

use cuda_core::{CudaStream, DeviceBuffer};

use crate::GpuError;
use crate::fault::FaultSink;
use crate::linear::conv::{ConvSeq, ConvSeqsArgs};
use crate::linear::delta::{DeltaRecord, DeltaSeq, DeltaSeqsArgs};
use crate::linear::{LinearKernels, LinearShape};
use crate::recur::RecStore;

/// One sequence of [`enqueue_gdn_seqs`]: its rows of the launches, its lane
/// word, its layer's store, and for a verify the store's stamps and record
/// (`None`: the step in place).
pub(crate) struct GdnSeq<'a> {
    pub(crate) rows: Range<usize>,
    pub(crate) lane: &'a DeviceBuffer<u32>,
    pub(crate) rec: &'a mut RecStore,
    pub(crate) verify: Option<(&'a mut DeviceBuffer<u32>, &'a mut DeviceBuffer<f32>)>,
}

/// [`enqueue_gdn_seqs`]'s arguments over every sequence's rows: the conv's
/// (`ConvArgs`' with F32 channels, without the ring) and the delta step's
/// output `o`.
pub(crate) struct GdnSeqsArgs<'a> {
    pub(crate) x: &'a DeviceBuffer<f32>,
    pub(crate) b_raw: &'a DeviceBuffer<f32>,
    pub(crate) a_raw: &'a DeviceBuffer<f32>,
    pub(crate) ba_pitch: usize,
    pub(crate) w: &'a DeviceBuffer<f32>,
    pub(crate) dt_bias: &'a DeviceBuffer<f32>,
    pub(crate) ssm_a: &'a DeviceBuffer<f32>,
    pub(crate) pos: &'a DeviceBuffer<u32>,
    pub(crate) shape: LinearShape,
    pub(crate) eps: f32,
    pub(crate) fault: FaultSink,
    pub(crate) conv: &'a mut DeviceBuffer<f32>,
    pub(crate) beta: &'a mut DeviceBuffer<f32>,
    pub(crate) decay: &'a mut DeviceBuffer<f32>,
    pub(crate) o: &'a mut DeviceBuffer<f32>,
}

/// Enqueue the conv of every sequence of `seqs` over its store's ring into
/// `a.conv`, `a.beta` and `a.decay`, then the delta step of every sequence
/// over its store's state into `a.o` (module doc): two launches. Refused by
/// name: what the conv's and the step's launches over several sequences
/// refuse (`ConvKernels::enqueue_conv_prep_seqs`,
/// `DeltaKernels::enqueue_delta_seqs`), and stores of more than one lane
/// count. Asynchronous and capturable; builds its sequence tables on the
/// host.
pub(crate) fn enqueue_gdn_seqs(
    lin: &LinearKernels,
    stream: &CudaStream,
    a: GdnSeqsArgs<'_>,
    seqs: &mut [GdnSeq<'_>],
) -> Result<(), GpuError> {
    const WHAT: &str = "linear::enqueue_gdn_seqs";
    let GdnSeqsArgs {
        x,
        b_raw,
        a_raw,
        ba_pitch,
        w,
        dt_bias,
        ssm_a,
        pos,
        shape,
        eps,
        fault,
        conv,
        beta,
        decay,
        o,
    } = a;
    let lanes = match seqs {
        [first, rest @ ..] if rest.iter().all(|q| q.rec.lanes == first.rec.lanes) => {
            first.rec.lanes
        }
        _ => {
            return Err(GpuError::shape(
                WHAT,
                "sequences whose delta stores hold one lane count",
            ));
        }
    };
    let rings: Vec<ConvSeq<'_>> = seqs
        .iter_mut()
        .map(|q| ConvSeq {
            rows: q.rows.clone(),
            ring: &mut q.rec.ring,
        })
        .collect();
    lin.conv.enqueue_conv_prep_seqs(
        stream,
        ConvSeqsArgs {
            x,
            b_raw,
            a_raw,
            ba_pitch,
            w,
            dt_bias,
            ssm_a,
            pos,
            shape,
            eps,
            fault,
            y: &mut *conv,
            beta: &mut *beta,
            decay: &mut *decay,
            seqs: &rings,
        },
    )?;
    drop(rings);
    let states: Vec<DeltaSeq<'_>> = seqs
        .iter_mut()
        .map(|q| DeltaSeq {
            rows: q.rows.clone(),
            lane: q.lane,
            state: &mut q.rec.state,
            verify: q.verify.as_mut().map(|(stamp, record)| DeltaRecord {
                carry: q.rec.carry.as_ref(),
                stamp,
                record,
            }),
        })
        .collect();
    lin.delta.enqueue_delta_seqs(
        stream,
        DeltaSeqsArgs {
            qkv: conv,
            beta,
            decay,
            pos,
            lanes,
            shape,
            fault,
            o,
            seqs: &states,
        },
    )
}
