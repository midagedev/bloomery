//! Qwen3.8's resident state on the card: each layer's store, and the arena
//! every layer of one walk shares (they run in turn) for up to `rows` tokens
//! — the decode step's one, an eager pass's up to [`PASS_ROWS`]. Everything
//! is allocated once at load. Buffers are token-major (`[rows][width]`)
//! unless their comment names another layout; a unit of `m <= rows` tokens
//! uses the first `m` rows.
//!
//! The store sizes are `runtime::stores`'s rules, which the plan counts by:
//! the constants the card's kernels are built for are held to them here, and
//! the load holds the stores it allocated to the rules' bytes
//! ([`store_rule_bytes`]).

use super::plan38::{GDN, geo};
use super::router::{RouterDims, RouterOut};
use super::scratch::{
    Dims, IN_IDS, IN_POS0, Inbox, Io, KvPlanes, LANE, RecStore, param_view, put_input,
};
use crate::GpuError;
use crate::flash_gqa::{partials_ms_len, partials_v_len_256};
use crate::hc_gated::HcScratch;
use crate::linear::{self, LinearShape};
use crate::ple;
use crate::qsa::{self, QsaScratch};
use cuda_core::{CudaStream, DeviceBuffer};
use runtime::hc_gated::Geometry;
use runtime::stores;
use std::mem::ManuallyDrop;

/// The most tokens an eager pass walks: the m-column kernels' width.
pub(super) const PASS_ROWS: usize = 8;

// The card's rings keep the rows the plan counts them by.
const _: () = assert!(
    linear::PASS_ROWS == stores::PASS_ROWS
        && linear::RING_ROWS == stores::conv_ring_rows(linear::CONV_TAPS)
        && ple::RING_ROWS == stores::ple_ring_rows(ple::TAPS, ple::DILATION)
        && PASS_ROWS == qsa::MAX_ROWS
        && PASS_ROWS <= stores::PASS_ROWS
);

/// A layer's store: the delta rule's recurrent state and conv ring, or a
/// selecting attention layer's K/V planes with its raw and pooled indexer
/// keys (`[ctx][IDX_DIM]` and `[pools][IDX_DIM]` f16).
pub(super) enum Store38 {
    Rec(RecStore),
    Qsa {
        kv: KvPlanes,
        raw: DeviceBuffer<u16>,
        pooled: DeviceBuffer<u16>,
    },
}

impl Store38 {
    /// A zeroed delta store (one lane).
    pub(super) fn rec(stream: &CudaStream) -> Result<Store38, GpuError> {
        Ok(Store38::Rec(RecStore::new(stream, GDN, 1)?))
    }

    /// A zeroed selecting store of `ctx` positions over the planes `d` cuts.
    pub(super) fn qsa(stream: &CudaStream, d: &Dims) -> Result<Store38, GpuError> {
        Ok(Store38::Qsa {
            kv: KvPlanes::new(stream, d)?,
            raw: DeviceBuffer::zeroed(stream, d.ctx * geo::IDX_DIM)?,
            pooled: DeviceBuffer::zeroed(stream, qsa::pools_for(d.ctx) * geo::IDX_DIM)?,
        })
    }

    /// Device bytes.
    pub(super) fn bytes(&self) -> usize {
        match self {
            Store38::Rec(r) => r.bytes(),
            Store38::Qsa { kv, raw, pooled } => kv.bytes() + raw.num_bytes() + pooled.num_bytes(),
        }
    }
}

/// The bytes `runtime::stores` counts for a delta store and for a selecting
/// store at `ctx` positions: what the load's allocations must equal.
pub(super) fn store_rule_bytes(ctx: usize) -> (u64, u64) {
    (
        stores::recurrent_bytes(GDN.n_v, GDN.n_k, linear::HEAD, linear::CONV_TAPS),
        stores::selecting_bytes(geo::N_KV, geo::HEAD, geo::IDX_DIM, geo::POOL, ctx),
    )
}

/// The dims the shared scratch types take ([`KvPlanes`]): the planes' heads
/// and `ctx`, the router's instance.
pub(super) fn dims(router: RouterDims, ctx: usize) -> Dims {
    Dims {
        hidden: geo::HIDDEN,
        n_head: geo::N_HEAD,
        n_kv: geo::N_KV,
        head: geo::HEAD,
        q_rows: geo::Q_ROWS,
        ff: geo::FF,
        router,
        lin: Some(GDN),
        ctx,
    }
}

/// A delta layer's intermediates.
pub(super) struct Gdn38 {
    /// The q·k·v channels and `z`.
    pub(super) x: DeviceBuffer<f32>,
    pub(super) z: DeviceBuffer<f32>,
    /// The joined β·α gemv's output, row-major `[2·n_v][m]` (β's rows
    /// first): at one row it is β then α, which the conv reads in place.
    pub(super) ba: DeviceBuffer<f32>,
    /// β and α token-major, copied out of `ba` at more than one row.
    pub(super) b: DeviceBuffer<f32>,
    pub(super) a: DeviceBuffer<f32>,
    pub(super) conv: DeviceBuffer<f32>,
    pub(super) beta: DeviceBuffer<f32>,
    pub(super) decay: DeviceBuffer<f32>,
    pub(super) o: DeviceBuffer<f32>,
}

/// A selecting attention layer's intermediates.
pub(super) struct Qsa38 {
    /// The query projection's `[q | gate]` rows, the keys and values.
    pub(super) qg: DeviceBuffer<f32>,
    pub(super) k: DeviceBuffer<f32>,
    pub(super) v: DeviceBuffer<f32>,
    /// The normed and turned queries.
    pub(super) q: DeviceBuffer<f32>,
    /// The flash's partials, over a list's width.
    pub(super) part_v: DeviceBuffer<f32>,
    pub(super) part_ms: DeviceBuffer<f32>,
    /// The indexer's raw keys and queries, row-major as the f32 gemv writes
    /// them (`[IDX_DIM][m]`, `[IDX_HEADS·IDX_DIM][m]`), and the queries
    /// token-major, copied out at more than one row.
    pub(super) kr: DeviceBuffer<f32>,
    pub(super) qr: DeviceBuffer<f32>,
    pub(super) qi: DeviceBuffer<f32>,
    /// The selection's queries, scores and lists.
    pub(super) sel: QsaScratch,
}

/// The PLE site's intermediates: the rows the host gathered (`e`), the key
/// and value projections, the gate's outputs.
pub(super) struct Ple38 {
    pub(super) e: DeviceBuffer<f32>,
    pub(super) key: DeviceBuffer<f32>,
    pub(super) value: DeviceBuffer<f32>,
    pub(super) gv: DeviceBuffer<f32>,
    pub(super) ngv: DeviceBuffer<f32>,
    pub(super) gate: DeviceBuffer<f32>,
}

/// The arena of one walk for up to `rows` tokens.
pub(super) struct Arena38 {
    pub(super) rows: usize,
    /// The embedding rows, and each row's position and live key count, which
    /// the embedding launch writes.
    pub(super) emb: DeviceBuffer<f32>,
    pub(super) pos: DeviceBuffer<u32>,
    pub(super) n_keys: DeviceBuffer<u32>,
    /// The residual streams, `[rows][STREAMS][HIDDEN]`, in two buffers: the
    /// PLE site writes the streams into the other one, and the walk goes on
    /// from there.
    pub(super) res: [DeviceBuffer<f32>; 2],
    /// A mix's output (the mixer's input) and a sub-layer's output (the next
    /// combine's input).
    pub(super) mixed: DeviceBuffer<f32>,
    pub(super) y: DeviceBuffer<f32>,
    pub(super) hc: HcScratch,
    /// The mixer's rows before the output projection: the gated norm's, or
    /// the attention's under its gate.
    pub(super) attn: DeviceBuffer<f32>,
    /// The flash's output.
    pub(super) flash: DeviceBuffer<f32>,
    pub(super) gdn: Gdn38,
    pub(super) qsa: Qsa38,
    pub(super) ple: Ple38,
    /// The router's results, `N_USED + 1` slots a token (the shared expert's
    /// last), and the card's places the handoff writes (no routed expert
    /// lives on the card; the places are all the host's).
    pub(super) route: RouterOut,
    pub(super) sel: DeviceBuffer<u32>,
    /// The feed-forward block's input on an eager pass, the shared expert's
    /// gate, up, SwiGLU and output.
    pub(super) ffn_x: DeviceBuffer<f32>,
    pub(super) sh_g: DeviceBuffer<f32>,
    pub(super) sh_u: DeviceBuffer<f32>,
    pub(super) sh_h: DeviceBuffer<f32>,
    pub(super) sh_y: DeviceBuffer<f32>,
}

impl Arena38 {
    /// The arena for up to `rows` (1..=[`PASS_ROWS`]) tokens over caches of
    /// `ctx` positions. Load-time only.
    pub(super) fn new(
        stream: &CudaStream,
        router: RouterDims,
        rows: usize,
        ctx: usize,
    ) -> Result<Arena38, GpuError> {
        if !(1..=PASS_ROWS).contains(&rows) {
            return Err(GpuError::shape(
                "qwen4exp::Arena38::new",
                format!("{rows} rows; an arena holds 1..={PASS_ROWS}"),
            ));
        }
        let f = |n: usize| DeviceBuffer::<f32>::zeroed(stream, rows * n);
        let u = |n: usize| DeviceBuffer::<u32>::zeroed(stream, rows * n);
        let (h, wide) = (geo::HIDDEN, geo::STREAMS * geo::HIDDEN);
        let s: LinearShape = GDN;
        let width = qsa::list_width(geo::KEPT);
        let geometry = Geometry::new(geo::STREAMS as u32, geo::RANK as u32, h as u32)
            .map_err(|e| GpuError::shape("qwen4exp::Arena38::new", e.to_string()))?;
        Ok(Arena38 {
            rows,
            emb: f(h)?,
            pos: u(1)?,
            n_keys: u(1)?,
            res: [f(wide)?, f(wide)?],
            mixed: f(h)?,
            y: f(h)?,
            hc: HcScratch::new(stream, geometry)?,
            attn: f(geo::ATTN)?,
            flash: f(geo::ATTN)?,
            gdn: Gdn38 {
                x: f(s.channels())?,
                z: f(s.n_v * linear::HEAD)?,
                ba: f(2 * s.n_v)?,
                b: f(s.n_v)?,
                a: f(s.n_v)?,
                conv: f(s.channels())?,
                beta: f(s.n_v)?,
                decay: f(s.n_v)?,
                o: f(s.n_v * linear::HEAD)?,
            },
            qsa: Qsa38 {
                qg: f(geo::Q_ROWS)?,
                k: f(geo::KV)?,
                v: f(geo::KV)?,
                q: f(geo::ATTN)?,
                part_v: DeviceBuffer::zeroed(stream, partials_v_len_256(rows, geo::N_HEAD, width))?,
                part_ms: DeviceBuffer::zeroed(stream, partials_ms_len(rows, geo::N_HEAD, width))?,
                kr: f(geo::IDX_DIM)?,
                qr: f(geo::IDX_HEADS * geo::IDX_DIM)?,
                qi: f(geo::IDX_HEADS * geo::IDX_DIM)?,
                sel: QsaScratch::new(stream, rows, ctx, geo::KEPT)?,
            },
            ple: Ple38 {
                e: f(h)?,
                key: f(wide)?,
                value: f(h)?,
                gv: f(wide)?,
                ngv: f(wide)?,
                gate: f(geo::STREAMS)?,
            },
            route: RouterOut::with_tokens(stream, router, rows)?,
            sel: u(geo::N_USED)?,
            ffn_x: f(h)?,
            sh_g: f(geo::FF)?,
            sh_u: f(geo::FF)?,
            sh_h: f(geo::FF)?,
            sh_y: f(h)?,
        })
    }

    /// Device bytes.
    pub(super) fn bytes(&self) -> usize {
        let g = &self.gdn;
        let q = &self.qsa;
        let p = &self.ple;
        let f32s = [
            &self.emb,
            &self.res[0],
            &self.res[1],
            &self.mixed,
            &self.y,
            &self.attn,
            &self.flash,
            &g.x,
            &g.z,
            &g.ba,
            &g.b,
            &g.a,
            &g.conv,
            &g.beta,
            &g.decay,
            &g.o,
            &q.qg,
            &q.k,
            &q.v,
            &q.q,
            &q.part_v,
            &q.part_ms,
            &q.kr,
            &q.qr,
            &q.qi,
            &q.sel.q_out,
            &q.sel.scores,
            &p.e,
            &p.key,
            &p.value,
            &p.gv,
            &p.ngv,
            &p.gate,
            &self.ffn_x,
            &self.sh_g,
            &self.sh_u,
            &self.sh_h,
            &self.sh_y,
        ];
        let hc = [
            &self.hc.xn,
            &self.hc.dpart,
            &self.hc.ipart,
            &self.hc.lo,
            &self.hc.wgt,
        ];
        f32s.iter()
            .chain(hc.iter())
            .map(|b| b.num_bytes())
            .sum::<usize>()
            + [
                &self.pos,
                &self.n_keys,
                &self.sel,
                &q.sel.list,
                &q.sel.n_sel,
            ]
            .iter()
            .map(|b| b.num_bytes())
            .sum::<usize>()
            + self.route.bytes()
    }
}

/// An eager pass's input record — its first position, up to [`PASS_ROWS`]
/// ids and the lane word — and the windows a pass of `m` rows reads.
pub(super) struct PassRecord {
    ids: Vec<ManuallyDrop<DeviceBuffer<u32>>>,
    pos0: ManuallyDrop<DeviceBuffer<u32>>,
    lane: ManuallyDrop<DeviceBuffer<u32>>,
    inbox: Inbox,
}

/// The lane word's offset in a pass's record, after its ids.
const PR_LANE: usize = IN_IDS + PASS_ROWS;
/// Words of a pass's record.
const PR_WORDS: usize = PR_LANE + 1;

impl PassRecord {
    /// A zeroed record. Load-time only.
    pub(super) fn new(stream: &CudaStream) -> Result<PassRecord, GpuError> {
        let inbox = Inbox::new(stream, PR_WORDS)?;
        // SAFETY: every window lies inside the inbox's `PR_WORDS` device words
        // (`IN_IDS + m <= PR_LANE < PR_WORDS`, `IN_POS0 < PR_WORDS`), and the
        // inbox moves into the struct beside them (a move of the handle, not
        // of the allocation), where it outlives them.
        let (ids, pos0, lane) = unsafe {
            (
                (1..=PASS_ROWS)
                    .map(|m| param_view::<u32>(inbox.dev(), IN_IDS, m))
                    .collect(),
                param_view::<u32>(inbox.dev(), IN_POS0, 1),
                param_view::<u32>(inbox.dev(), PR_LANE, 1),
            )
        };
        Ok(PassRecord {
            ids,
            pos0,
            lane,
            inbox,
        })
    }

    /// Write `tokens` from position `pos` and the lane word, and enqueue the
    /// record's copy. Asynchronous: the pass behind it reads it.
    pub(super) fn write(
        &mut self,
        stream: &CudaStream,
        tokens: &[u32],
        pos: u32,
    ) -> Result<(), GpuError> {
        if !(1..=PASS_ROWS).contains(&tokens.len()) {
            return Err(GpuError::shape(
                "qwen4exp::PassRecord::write",
                format!("{} rows; a pass takes 1..={PASS_ROWS}", tokens.len()),
            ));
        }
        let host = self.inbox.host_mut()?;
        put_input(host, tokens, pos)?;
        host[PR_LANE] = LANE;
        self.inbox.upload(stream, PR_WORDS)
    }

    /// The input of a pass of `m` rows, as its first launch reads it.
    pub(super) fn io(&self, m: usize) -> Result<Io<'_>, GpuError> {
        let ids = m
            .checked_sub(1)
            .and_then(|i| self.ids.get(i))
            .ok_or_else(|| {
                GpuError::shape(
                    "qwen4exp::PassRecord::io",
                    format!("a pass of {m} rows (1..={PASS_ROWS})"),
                )
            })?;
        Ok(Io {
            ids,
            pos0: &self.pos0,
            first: 0,
            lane: Some(&self.lane),
        })
    }

    /// Device bytes.
    pub(super) fn bytes(&self) -> usize {
        self.inbox.bytes()
    }
}
