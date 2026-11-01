//! A Gated DeltaNet layer's delta step bound to its store ([`RecStore::gdn`]:
//! one stamped lane, the carry and the verify record), the one owner every
//! body's delta layer calls: one sequence's step in place, its verify
//! recorded into the record, or its prompt form ([`enqueue_gdn`], [`GdnRun`]);
//! the commit's replay table over a sequence's stores ([`recovery_table`]);
//! and the rows of several sequences in one launch each (`crate::seq_rows`,
//! [`enqueue_gdn_seqs`]): the pass of resident slots. The conv runs over each
//! sequence's ring; the delta step over each sequence's state, in place or, a
//! verify, recorded into each sequence's record, each after its carry. Each
//! sequence's rows are the launches of its rows alone, bit for bit, so a
//! pass of `n` sequences runs these two launches once where one launch a
//! sequence ran `2·n`.

use std::ops::Range;

use cuda_core::{CudaStream, DeviceBuffer};

use crate::GpuError;
use crate::fault::FaultSink;
use crate::linear::conv::{ConvSeq, ConvSeqsArgs};
use crate::linear::delta::{
    DeltaArgs, DeltaLanesArgs, DeltaSeq, DeltaSeqsArgs, DeltaStamped, RecoveryStore, RecoveryTable,
};
use crate::linear::{LinearKernels, LinearShape};
use crate::recur::RecStore;

/// How [`enqueue_gdn`] runs a call's rows over its store.
pub(crate) enum GdnRun<'w> {
    /// In place: the state after the last row back into the lane.
    Step,
    /// A verify: each row recorded into the store's record, the lane kept
    /// for the commit's replay ([`recovery_table`]).
    Verify,
    /// The prompt form over `work` ([`crate::linear::delta::prompt_work_len`]),
    /// the call's last chunk kept out of the lane when `keep`.
    Prompt {
        work: &'w mut DeviceBuffer<u32>,
        keep: bool,
    },
}

/// One call's delta step over a store: the conv's rows `qkv`, `beta` and
/// `decay` of `m` rows at positions `pos`, the lane word, and the output `o`.
pub(crate) struct GdnCall<'a> {
    pub(crate) qkv: &'a DeviceBuffer<f32>,
    pub(crate) beta: &'a DeviceBuffer<f32>,
    pub(crate) decay: &'a DeviceBuffer<f32>,
    pub(crate) lane: &'a DeviceBuffer<u32>,
    pub(crate) pos: &'a DeviceBuffer<u32>,
    pub(crate) shape: LinearShape,
    pub(crate) m: usize,
    pub(crate) fault: FaultSink,
    pub(crate) o: &'a mut DeviceBuffer<f32>,
}

/// Enqueue `a`'s delta step over `rec` as `run` asks, after the rows `rec`'s
/// carry holds for the call's first position: `enqueue_delta_lanes` in
/// place, `enqueue_delta_recovery` into the record, or
/// `enqueue_delta_prompt`. Refused by name: a store without stamps, a
/// prompt form or a verify over a store without its carry or record, and
/// what each launch refuses. Asynchronous; the step and the verify are
/// capturable.
pub(crate) fn enqueue_gdn(
    lin: &LinearKernels,
    stream: &CudaStream,
    a: GdnCall<'_>,
    rec: &mut RecStore,
    run: GdnRun<'_>,
) -> Result<(), GpuError> {
    const WHAT: &str = "linear::enqueue_gdn";
    let RecStore {
        state,
        lanes,
        stamp,
        carry,
        record,
        ..
    } = rec;
    let stamp = stamp
        .as_mut()
        .ok_or(GpuError::state(WHAT, "a store of stamped lanes"))?;
    let args = DeltaLanesArgs {
        delta: DeltaArgs {
            qkv: a.qkv,
            beta: a.beta,
            decay: a.decay,
            lane: a.lane,
            lane_at: 0,
            lanes: *lanes,
            shape: a.shape,
            m: a.m,
            fault: a.fault,
            o: a.o,
            state,
        },
        each: matches!(run, GdnRun::Verify),
        pos: a.pos,
        stamp,
    };
    let delta = &lin.delta;
    match run {
        GdnRun::Step => delta.enqueue_delta_lanes(stream, args, carry.as_ref()),
        GdnRun::Verify => {
            let record = record
                .as_mut()
                .ok_or(GpuError::state(WHAT, "the store's verify record"))?;
            delta.enqueue_delta_recovery(stream, args, carry.as_ref(), record)
        }
        GdnRun::Prompt { work, keep } => {
            let carry = carry
                .as_mut()
                .ok_or(GpuError::state(WHAT, "the store's carry"))?;
            delta.enqueue_delta_prompt(stream, args, carry, work, keep)
        }
    }
}

/// The commit's replay table over a sequence's delta stores `stores` (each
/// `(layer, store)`, its faults under that layer) of `shape`, records of
/// `rows` rows, one lane each ([`RecoveryTable::new`]); a store without its
/// stamps or record is refused by name.
///
/// # Safety
///
/// [`RecoveryTable::new`]'s: the stores stay allocated at their addresses
/// while any launch or captured graph using the table can run, and the
/// replay is the only launch on them while it runs.
pub(crate) unsafe fn recovery_table<'a>(
    stream: &CudaStream,
    (shape, rows): (LinearShape, usize),
    stores: impl IntoIterator<Item = (usize, &'a RecStore)>,
) -> Result<RecoveryTable, GpuError> {
    const WHAT: &str = "linear::recovery_table";
    let rows_of = stores
        .into_iter()
        .map(|(layer, r)| {
            Ok(RecoveryStore {
                layer,
                state: &r.state,
                stamp: r
                    .stamp
                    .as_ref()
                    .ok_or(GpuError::state(WHAT, "a store of stamped lanes"))?,
                carry: r.carry.as_ref(),
                record: r
                    .record
                    .as_ref()
                    .ok_or(GpuError::state(WHAT, "the store's verify record"))?,
            })
        })
        .collect::<Result<Vec<_>, GpuError>>()?;
    // SAFETY: the caller's contract (above) is the table's.
    unsafe { RecoveryTable::new(stream, shape, (1, rows), &rows_of) }
}

/// One sequence of [`enqueue_gdn_seqs`]: its rows of the launches, its lane
/// word, its layer's store, and whether its rows are a verify recorded into
/// the store's record (else the step in place).
pub(crate) struct GdnSeq<'a> {
    pub(crate) rows: Range<usize>,
    pub(crate) lane: &'a DeviceBuffer<u32>,
    pub(crate) rec: &'a mut RecStore,
    pub(crate) verify: bool,
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
/// `DeltaKernels::enqueue_delta_seqs`), stores of more than one lane count,
/// and a store without its stamps or a verify's without its record.
/// Asynchronous and capturable; builds its sequence tables on the host.
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
    let states = seqs
        .iter_mut()
        .map(|q| {
            let RecStore {
                state,
                stamp,
                carry,
                record,
                ..
            } = &mut *q.rec;
            Ok(DeltaSeq {
                rows: q.rows.clone(),
                lane: q.lane,
                state,
                stamped: DeltaStamped {
                    carry: carry.as_ref(),
                    stamp: stamp
                        .as_mut()
                        .ok_or(GpuError::state(WHAT, "a store of stamped lanes"))?,
                    record: if q.verify {
                        Some(
                            record
                                .as_mut()
                                .ok_or(GpuError::state(WHAT, "the store's verify record"))?,
                        )
                    } else {
                        None
                    },
                },
            })
        })
        .collect::<Result<Vec<_>, GpuError>>()?;
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
