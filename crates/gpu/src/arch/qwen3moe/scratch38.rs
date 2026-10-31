//! Qwen3.8's resident state on the card: each layer's store, and the arena
//! every layer of one walk shares (they run in turn) for up to `rows` tokens
//! — the decode step's one, an eager pass's up to [`PASS_ROWS`], a ubatch
//! walk's up to its load-time size ([`Arena38::wide`]). Everything is
//! allocated once at load; a selecting store's addresses are reserved there
//! and its rows backed as a sequence's positions reach them
//! ([`Store38::fit`]). Buffers are token-major (`[rows][width]`)
//! unless their comment names another layout; a unit of `m <= rows` tokens
//! uses the first `m` rows.
//!
//! The store sizes are `runtime::stores`'s rules, which the plan counts by:
//! the constants the card's kernels are built for are held to them here, and
//! the load holds the stores it allocated to the rules' bytes
//! ([`store_rule_bytes`]).

use super::body::KvQ8;
use super::body38::WHAT_SHARE;
use super::plan38::{GDN, Layer38, geo};
use super::scratch::{
    Dims, IN_IDS, IN_POS0, Inbox, Io, KvPlanes, LANE, RecStore, fresh_stamps, param_view, put_input,
};
use crate::GpuError;
use crate::flash_gqa::{listed_partials_ms_len, listed_partials_v_len_256};
use crate::hc_gated::HcScratch;
use crate::linear::{self, LinearShape};
use crate::ple;
use crate::qsa::{self, QsaScratch};
use crate::room::{Room, Roomed, Run, SharedPages};
use crate::route_softmax::{RouterDims, RouterOut};
use crate::site::ByName;
use crate::tensor::WindowMut;
use cuda_core::{CudaStream, DeviceBuffer};
use runtime::hc_gated::Geometry;
use runtime::stores;
use std::sync::Arc;

/// The most tokens an eager pass walks: the m-column kernels' width.
pub(super) const PASS_ROWS: usize = 8;

/// The rows one selecting run of a ubatch walk of up to `rows` tokens takes
/// over caches of `ctx` positions: as many as the plan's selection bytes
/// (`place::SELECT_RUN_BYTES`) hold at the selection's bytes a row
/// ([`QsaScratch::row_bytes`]), in whole groups of the score pass's
/// [`qsa::MAX_ROWS`], at least one group and at most `rows`. Each run uses
/// two selection launches and one attention launch, which needs no
/// partials.
pub(super) fn select_run(rows: usize, ctx: usize) -> usize {
    let row = QsaScratch::row_bytes(ctx, geo::KEPT) as u64;
    let fit = usize::try_from(model::arch::qwen35moe::place::SELECT_RUN_BYTES / row)
        .expect("a quotient of the plan's selection bytes fits a 64-bit usize");
    (fit / qsa::MAX_ROWS * qsa::MAX_ROWS)
        .max(qsa::MAX_ROWS)
        .min(rows)
}

/// The most tokens one router launch of a ubatch walk takes, and so one
/// routed table over them: 2,048 tokens at the router's eleven slots a token
/// fit one GEMM route table, and a 4,096-token ubatch routes in two. The
/// tier's block route runs in runs of this many. So does the card route,
/// unless its plan gives it a run of the whole ubatch
/// (`place::card_route_run`); at ten slots a token, that run fits one table.
pub(super) const ROUTE_ROWS: usize = 2048;

// The card's rings keep the rows the plan counts them by.
const _: () = assert!(
    linear::PASS_ROWS == stores::PASS_ROWS
        && linear::RING_ROWS == stores::conv_ring_rows(linear::CONV_TAPS)
        && ple::RING_ROWS == stores::ple_ring_rows(ple::TAPS, ple::DILATION)
        && qsa::RING_ROWS == stores::index_ring_rows(geo::POOL)
        && PASS_ROWS <= stores::PASS_ROWS
        && ROUTE_ROWS * (geo::N_USED + 1) <= crate::gemm::GEMM_MAX_SLOTS
);

// At the card route's ten slots a token, the default ubatch fills exactly one
// GEMM route table, so a card route of the whole ubatch takes one table.
const _: () = assert!(super::ubatch::UBATCH * geo::N_USED == crate::gemm::GEMM_MAX_SLOTS);

/// Lanes of a delta store's state: one. A verify records its rows beside
/// the state and the commit replays the kept ones into that lane
/// (`linear::delta`'s `enqueue_delta_recovery`).
// PIN(2026-10-30): the lane-rotation test and the gates' planted-lane cases
// (`delta_stamp` on a lane no call wrote) are removed: a delta store has one
// lane, a verify's rows go to its record, and a commit moves no lane word.
pub(super) const LANES: usize = 1;

/// The most rows a verify runs, which a delta store's record holds.
/// `runtime::stores` owns the count.
pub(super) const VERIFY_ROWS: usize = stores::DELTA_VERIFY_ROWS;

/// A layer's store: the delta rule's recurrent state ([`LANES`] lane), its
/// lane's stamp (`linear::delta`'s module doc), conv ring, carry and verify
/// record ([`VERIFY_ROWS`] rows), or a
/// selecting attention layer's K/V planes, its raw indexer key ring
/// (`[qsa::RING_ROWS][IDX_DIM]` f16, `crate::qsa`'s module doc) and its pooled
/// keys (`[pools][IDX_DIM]` f16), the planes and the pooled keys each in a
/// room whose pages are backed as far as the sequence's positions reach
/// ([`Store38::fit`]).
pub(super) enum Store38 {
    Rec {
        rec: RecStore,
        stamp: DeviceBuffer<u32>,
        record: DeviceBuffer<f32>,
    },
    Qsa {
        kv: Roomed<KvPlanes>,
        ring: DeviceBuffer<u16>,
        pooled: Roomed<DeviceBuffer<u16>>,
    },
}

impl Store38 {
    /// A zeroed delta store of [`LANES`] lane with a carry and a record,
    /// the lane stamped at position 0.
    pub(super) fn rec(stream: &CudaStream) -> Result<Store38, GpuError> {
        Ok(Store38::Rec {
            rec: RecStore::carried(stream, GDN.lens(LANES), LANES, GDN.carry_len())?,
            stamp: DeviceBuffer::from_host(stream, &fresh_stamps(LANES))?,
            record: DeviceBuffer::zeroed(
                stream,
                linear::delta::recovery_record_len(GDN, VERIFY_ROWS),
            )?,
        })
    }

    /// A delta store's stamp back to a fresh store's: the lane at position
    /// 0. Never inside a capture.
    pub(super) fn restamp(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        if let Store38::Rec { stamp, .. } = self {
            stamp.copy_from_host(stream, &fresh_stamps(LANES))?;
        }
        Ok(())
    }

    /// A selecting store of `ctx` positions over the planes `d` cuts, its
    /// K/V planes in the load's cache format `kv` (the index keys f16), no
    /// row backed until a fit reaches it, zeroed then on `stream`.
    pub(super) fn qsa(stream: &Arc<CudaStream>, d: &Dims, kv: KvQ8) -> Result<Store38, GpuError> {
        let pools = qsa::pools_for(d.ctx);
        let run = Run {
            at: 0,
            row: geo::IDX_DIM * size_of::<u16>(),
            per: geo::POOL,
            rows: pools,
        };
        Ok(Store38::Qsa {
            kv: KvPlanes::roomed(stream, (d.n_kv, d.ctx, d.head), kv)?,
            ring: DeviceBuffer::zeroed(stream, qsa::RING_ROWS * geo::IDX_DIM)?,
            pooled: Roomed::new(Arc::clone(stream), pools * geo::IDX_DIM, &[run], |[b]| b)?,
        })
    }

    /// Device bytes the store's layout takes: what the plan counts it at.
    pub(super) fn bytes(&self) -> usize {
        match self {
            Store38::Rec { rec, stamp, record } => {
                rec.bytes() + stamp.num_bytes() + record.num_bytes()
            }
            Store38::Qsa { kv, ring, pooled } => kv.bytes() + ring.num_bytes() + pooled.num_bytes(),
        }
    }

    /// Device bytes a selecting store's pages with memory of their own
    /// hold; none for a delta store.
    pub(super) fn mapped(&self) -> usize {
        match self {
            Store38::Rec { .. } => 0,
            Store38::Qsa { kv, pooled, .. } => kv.mapped() + pooled.mapped(),
        }
    }

    /// Device bytes a selecting store's pages hold once all are backed;
    /// none for a delta store.
    pub(super) fn whole(&self) -> usize {
        match self {
            Store38::Rec { .. } => 0,
            Store38::Qsa { kv, pooled, .. } => kv.whole() + pooled.whole(),
        }
    }

    /// A selecting store's rooms; none for a delta store, which holds no
    /// rows by position.
    fn rooms(&mut self) -> impl Iterator<Item = &mut Room> {
        let qsa = match self {
            Store38::Rec { .. } => None,
            Store38::Qsa { kv, pooled, .. } => Some((kv, pooled)),
        };
        qsa.into_iter().flat_map(|(kv, pooled)| {
            (kv.rooms_mut().iter_mut()).chain(pooled.rooms_mut().iter_mut())
        })
    }

    /// A selecting store's rows of the first `positions` positions backed
    /// ([`Room::fit`]). Never inside a capture.
    pub(super) fn fit(&mut self, positions: usize) -> Result<(), GpuError> {
        self.rooms().try_for_each(|r| r.fit(positions))
    }

    /// A selecting store's pages a write from position `pos` on reaches given
    /// memory no other sequence's store maps ([`Room::own_from`]): every
    /// writer's first step after its fit. Never inside a capture.
    pub(super) fn own_from(&mut self, pos: usize) -> Result<(), GpuError> {
        self.rooms().try_for_each(|r| r.own_from(pos))
    }

    /// A selecting store's rows of the first `positions` positions taken
    /// from `from`, the same layer's store of another sequence
    /// ([`Roomed::share`]): its K/V planes' and its pooled keys' pages
    /// below the prefix mapped, the page where it ends copied. Its raw key
    /// ring is a checkpoint store, put back with the recurrent ones. Nothing
    /// for a delta store. Refused by name for stores of two kinds, and as
    /// [`Roomed::share`] refuses. Never inside a capture.
    pub(super) fn share(
        &mut self,
        from: &Store38,
        positions: usize,
    ) -> Result<SharedPages, GpuError> {
        match (self, from) {
            (Store38::Rec { .. }, Store38::Rec { .. }) => Ok(SharedPages::default()),
            (
                Store38::Qsa { kv, pooled, .. },
                Store38::Qsa {
                    kv: kv_from,
                    pooled: pooled_from,
                    ..
                },
            ) => Ok(kv.share(kv_from, positions)? + pooled.share(pooled_from, positions)?),
            _ => Err(GpuError::shape(
                WHAT_SHARE,
                "a store's share from another kind of store",
            )),
        }
    }

    /// Device bytes of a selecting store's pages another sequence's store
    /// maps too; none for a delta store.
    pub(super) fn shared(&self) -> usize {
        match self {
            Store38::Rec { .. } => 0,
            Store38::Qsa { kv, pooled, .. } => kv.shared() + pooled.shared(),
        }
    }

    /// Every row of a selecting store backed: what a read or write of its
    /// whole planes takes.
    pub(super) fn fill(&mut self) -> Result<(), GpuError> {
        self.fit(usize::MAX)
    }

    /// A sequence's end: the memory of a selecting store's pages past the
    /// rows it reached given back ([`Room::settle`]).
    pub(super) fn settle(&mut self) -> Result<(), GpuError> {
        self.rooms().try_for_each(Room::settle)
    }
}

/// The refusal of a delta store without the carry [`Store38::rec`] keeps.
pub(super) fn no_carry() -> GpuError {
    GpuError::state(
        "qwen4exp delta store",
        "a delta store's carry (`Store38::rec` keeps one)",
    )
}

/// The bytes `runtime::stores` counts for a delta store of [`LANES`] lane
/// with its record and for a selecting store at `ctx` positions in the cache format `kv`:
/// what the load's allocations must equal.
pub(super) fn store_rule_bytes(ctx: usize, kv: KvQ8) -> (u64, u64) {
    (
        stores::recurrent_bytes(GDN.n_v, GDN.n_k, linear::HEAD, linear::CONV_TAPS)
            + stores::delta_lane_bytes(GDN.n_v, linear::HEAD, LANES)
            + stores::delta_record_bytes(GDN.n_v, linear::HEAD, VERIFY_ROWS)
            + stores::delta_carry_bytes(GDN.n_v, GDN.n_k, linear::HEAD),
        stores::selecting_bytes(
            geo::N_KV,
            geo::HEAD,
            kv == KvQ8::Q8,
            geo::IDX_DIM,
            geo::POOL,
            ctx,
        ),
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
        router: Some(router),
        lin: Some(GDN),
        ctx,
    }
}

/// A delta layer's intermediates.
pub(super) struct Gdn38 {
    /// The q·k·v channels and `z`.
    pub(super) x: DeviceBuffer<f32>,
    pub(super) z: DeviceBuffer<f32>,
    /// The joined β·α projection token-major, each row's β values then its
    /// α's (`program38::beta_alpha_rows`), which the conv reads in place.
    pub(super) ba: DeviceBuffer<f32>,
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
    /// The normed and turned queries: a pass's rows, none in a ubatch walk's
    /// arena (its flashes norm and turn the queries they read).
    pub(super) q: DeviceBuffer<f32>,
    /// The flash's partials, over a list's width: a pass's rows, none in a
    /// ubatch walk's arena (its listed prefill flash takes none).
    pub(super) part_v: DeviceBuffer<f32>,
    pub(super) part_ms: DeviceBuffer<f32>,
    /// The indexer's raw keys and queries, token-major.
    pub(super) kr: DeviceBuffer<f32>,
    pub(super) qi: DeviceBuffer<f32>,
    /// The selection's queries, scores and lists.
    pub(super) sel: QsaScratch,
}

/// The PLE site's intermediates: the rows the host gathered (`e`) and, for
/// NVFP4 rows the card decodes, their code and scale planes as read
/// (`planes`, 9 bytes a 16 values), the key and value projections, the
/// gate's outputs.
pub(super) struct Ple38 {
    pub(super) e: DeviceBuffer<f32>,
    pub(super) planes: DeviceBuffer<u8>,
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
    /// from there; a walk leaves its last streams in buffer [`final_res`].
    /// A ubatch walk carries its streams as BF16 values in the other buffer,
    /// which its arena allocates at `rows·STREAMS·HIDDEN` halfwords, and
    /// writes F32 values into buffer [`final_res`] only where a PLE site
    /// reads them and where its last combine leaves its last streams.
    pub(super) res: [DeviceBuffer<f32>; 2],
    /// A mix's output (the mixer's input) and a sub-layer's output (the next
    /// combine's input).
    pub(super) mixed: DeviceBuffer<f32>,
    pub(super) y: DeviceBuffer<f32>,
    pub(super) hc: HcScratch,
    /// The mixer's rows before the output projection: the gated norm's, or
    /// the attention's under its gate.
    pub(super) attn: DeviceBuffer<f32>,
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
    /// The by-name projections' forms and staging: the main stream's, and the
    /// branch's, whose projections run beside the main stream's
    /// ([`Branch`](crate::Branch)) and so share none of its.
    pub(super) by: ByName,
    pub(super) by_fork: ByName,
}

/// The buffer of [`Arena38::res`] that holds a walk's last streams under
/// `plans`: a walk starts in buffer 0 and each PLE site past layer 0 moves it
/// to the other one.
pub(super) fn final_res(plans: &[Layer38]) -> usize {
    plans.iter().skip(1).filter(|p| p.ple.is_some()).count() % 2
}

impl Arena38 {
    /// The arena for up to `rows` (1..=[`PASS_ROWS`]) tokens over caches of
    /// `ctx` positions. Load-time only.
    pub(super) fn new(
        stream: &CudaStream,
        router: RouterDims,
        rows: usize,
        ctx: usize,
        by: (ByName, ByName),
    ) -> Result<Arena38, GpuError> {
        if !(1..=PASS_ROWS).contains(&rows) {
            return Err(GpuError::shape(
                "qwen4exp::Arena38::new",
                format!("{rows} rows; an arena holds 1..={PASS_ROWS}"),
            ));
        }
        let route = RouterOut::with_tokens(stream, router, rows)?;
        Arena38::alloc(stream, (rows, None), (rows, rows), route, ctx, by)
    }

    /// A ubatch walk's arena for up to `rows` (1..=`most`) tokens over caches
    /// of `ctx` positions, `fin` the walk's [`final_res`]: every token-major
    /// buffer for `rows` tokens (the stream buffer other than `fin` in BF16), the
    /// selection's scratch for a run of [`select_run`] rows (its selecting
    /// rows run in runs of that many) and no selected-flash queries or
    /// partials, the router's buffers for [`ROUTE_ROWS`] (it routes in runs of that many).
    /// Load-time only.
    pub(super) fn wide(
        stream: &CudaStream,
        router: RouterDims,
        (rows, most, fin): (usize, usize, usize),
        ctx: usize,
        by: (ByName, ByName),
    ) -> Result<Arena38, GpuError> {
        if !(1..=most).contains(&rows) || fin > 1 {
            return Err(GpuError::shape(
                "qwen4exp::Arena38::wide",
                format!(
                    "{rows} rows, final streams {fin}; a ubatch arena holds 1..={most} rows and two stream buffers"
                ),
            ));
        }
        let route = RouterOut::for_ubatch(stream, router, rows.min(ROUTE_ROWS))?;
        let mut a = Arena38::alloc(
            stream,
            (rows, Some(1 - fin)),
            (select_run(rows, ctx), 0),
            route,
            ctx,
            by,
        )?;
        // The prompt form's records take the q·k·v rows' room once the conv
        // has read them (`wide38::gdn`); a short walk needs a little more
        // than the rows.
        let work = linear::delta::prompt_work_len(GDN, rows);
        if a.gdn.x.len() < work {
            a.gdn.x = DeviceBuffer::zeroed(stream, work)?;
        }
        Ok(a)
    }

    /// The buffers for `rows` tokens (stream buffer `bf16`, when named, at a
    /// halfword a value), the selection's for `sel` rows and the selected
    /// flash's queries and partials for `parts`, the router's `route`.
    fn alloc(
        stream: &CudaStream,
        (rows, bf16): (usize, Option<usize>),
        (sel, parts): (usize, usize),
        route: RouterOut,
        ctx: usize,
        (by, by_fork): (ByName, ByName),
    ) -> Result<Arena38, GpuError> {
        let f = |n: usize| DeviceBuffer::<f32>::zeroed(stream, rows * n);
        let u = |n: usize| DeviceBuffer::<u32>::zeroed(stream, rows * n);
        let (h, wide) = (geo::HIDDEN, geo::STREAMS * geo::HIDDEN);
        let s: LinearShape = GDN;
        let width = qsa::list_width(geo::KEPT);
        let geometry = Geometry::new(geo::STREAMS as u32, geo::RANK as u32, h as u32)
            .map_err(|e| GpuError::shape("qwen4exp::Arena38::new", e.to_string()))?;
        let streams = |b: usize| f(if bf16 == Some(b) { wide / 2 } else { wide });
        const _: () = assert!((geo::STREAMS * geo::HIDDEN).is_multiple_of(2));
        Ok(Arena38 {
            rows,
            emb: f(h)?,
            pos: u(1)?,
            n_keys: u(1)?,
            res: [streams(0)?, streams(1)?],
            mixed: f(h)?,
            y: f(h)?,
            hc: HcScratch::new(stream, geometry)?,
            attn: f(geo::ATTN)?,
            gdn: Gdn38 {
                x: f(s.channels())?,
                z: f(s.n_v * linear::HEAD)?,
                ba: f(2 * s.n_v)?,
                conv: f(s.channels())?,
                beta: f(s.n_v)?,
                decay: f(s.n_v)?,
                o: f(s.n_v * linear::HEAD)?,
            },
            qsa: Qsa38 {
                qg: f(geo::Q_ROWS)?,
                k: f(geo::KV)?,
                v: f(geo::KV)?,
                q: DeviceBuffer::zeroed(stream, parts * geo::ATTN)?,
                part_v: DeviceBuffer::zeroed(
                    stream,
                    listed_partials_v_len_256(parts, geo::N_HEAD, width),
                )?,
                part_ms: DeviceBuffer::zeroed(
                    stream,
                    listed_partials_ms_len(parts, geo::N_HEAD, width),
                )?,
                kr: f(geo::IDX_DIM)?,
                qi: f(geo::IDX_HEADS * geo::IDX_DIM)?,
                sel: QsaScratch::new(stream, sel, ctx, geo::KEPT)?,
            },
            ple: Ple38 {
                e: f(h)?,
                planes: DeviceBuffer::zeroed(stream, rows * h / 16 * 9)?,
                key: f(wide)?,
                value: f(h)?,
                gv: f(wide)?,
                ngv: f(wide)?,
                gate: f(geo::STREAMS)?,
            },
            route,
            sel: u(geo::N_USED)?,
            ffn_x: f(h)?,
            sh_g: f(geo::FF)?,
            sh_u: f(geo::FF)?,
            sh_h: f(geo::FF)?,
            sh_y: f(h)?,
            by,
            by_fork,
        })
    }

    /// Device bytes of the by-name projections' forms and staging, which the
    /// plan's arena counts leave out ([`Arena38::bytes`]).
    pub(super) fn by_bytes(&self) -> usize {
        self.by.bytes() + self.by_fork.bytes()
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
            &g.x,
            &g.z,
            &g.ba,
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
            + p.planes.num_bytes()
            + self.route.bytes()
    }
}

/// A gate's per-layer taps of the decode step: each layer's streams after
/// its combine (`STREAMS · HIDDEN`), and its router's logits and slot ids as
/// the step's arena holds them (its first `EXPERTS` logits and `N_USED` ids
/// are the step's row). Allocated when armed, never on the step's path
/// otherwise; a pass writes none.
pub(super) struct Taps38 {
    pub(super) out: Vec<DeviceBuffer<f32>>,
    pub(super) logits: Vec<DeviceBuffer<f32>>,
    pub(super) ids: Vec<DeviceBuffer<u32>>,
}

impl Taps38 {
    /// Zeroed taps for `layers` layers, the route taps as wide as `route`'s
    /// buffers.
    pub(super) fn new(
        stream: &CudaStream,
        layers: usize,
        route: &RouterOut,
    ) -> Result<Taps38, GpuError> {
        let each = |n: usize| {
            (0..layers)
                .map(|_| DeviceBuffer::zeroed(stream, n))
                .collect::<Result<Vec<_>, _>>()
        };
        Ok(Taps38 {
            out: each(geo::STREAMS * geo::HIDDEN)?,
            logits: each(route.logits.len())?,
            ids: (0..layers)
                .map(|_| DeviceBuffer::zeroed(stream, route.ids.len()))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    /// Device bytes.
    pub(super) fn bytes(&self) -> usize {
        self.out.iter().map(DeviceBuffer::num_bytes).sum::<usize>()
            + self
                .logits
                .iter()
                .map(DeviceBuffer::num_bytes)
                .sum::<usize>()
            + self.ids.iter().map(DeviceBuffer::num_bytes).sum::<usize>()
    }
}

/// A ubatch walk's input record — its first position, up to `rows` ids and
/// the lane word — and the window of the ids its last write holds.
pub(super) struct WideRecord {
    rows: usize,
    ids: WindowMut<'static, u32>,
    pos0: WindowMut<'static, u32>,
    lane: WindowMut<'static, u32>,
    inbox: Inbox,
}

impl WideRecord {
    /// A zeroed record for up to `rows` ids. Load-time only.
    pub(super) fn new(stream: &CudaStream, rows: usize) -> Result<WideRecord, GpuError> {
        if rows == 0 {
            return Err(GpuError::shape(
                "qwen4exp::WideRecord::new",
                "a record of no row",
            ));
        }
        let inbox = Inbox::new(stream, IN_IDS + rows + 1)?;
        // SAFETY: every window lies inside the inbox's `IN_IDS + rows + 1`
        // device words, and the inbox moves into the struct beside them (a
        // move of the handle, not of the allocation), where it outlives them.
        let (ids, pos0, lane) = unsafe {
            (
                param_view::<u32>(inbox.dev(), IN_IDS, 1),
                param_view::<u32>(inbox.dev(), IN_POS0, 1),
                param_view::<u32>(inbox.dev(), IN_IDS + rows, 1),
            )
        };
        Ok(WideRecord {
            rows,
            ids,
            pos0,
            lane,
            inbox,
        })
    }

    /// Write `tokens` (1..=`rows`) from position `pos` and the lane word,
    /// enqueue the record's copy, and window its ids. Asynchronous: the walk
    /// behind it reads it.
    pub(super) fn write(
        &mut self,
        stream: &CudaStream,
        tokens: &[u32],
        pos: u32,
    ) -> Result<(), GpuError> {
        let (n, rows) = (tokens.len(), self.rows);
        if !(1..=rows).contains(&n) {
            return Err(GpuError::shape(
                "qwen4exp::WideRecord::write",
                format!("{n} rows; a ubatch takes 1..={rows}"),
            ));
        }
        let host = self.inbox.host_mut()?;
        put_input(host, tokens, pos)?;
        host[IN_IDS + rows] = LANE;
        self.inbox.upload(stream, IN_IDS + rows + 1)?;
        // SAFETY: `IN_IDS + n <= IN_IDS + rows`, inside the inbox's device
        // words, which outlive the window (it lives in the struct beside
        // them).
        self.ids = unsafe { param_view::<u32>(self.inbox.dev(), IN_IDS, n) };
        Ok(())
    }

    /// The input of the last write's ubatch, as its first launch reads it.
    pub(super) fn io(&self) -> Io<'_> {
        Io {
            ids: &self.ids,
            pos0: &self.pos0,
            first: 0,
            lane: Some(&self.lane),
        }
    }

    /// Device bytes.
    pub(super) fn bytes(&self) -> usize {
        self.inbox.bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::{Arena38, ROUTE_ROWS, RouterDims, geo};
    use crate::site::ByName;
    use crate::weights::Weights;
    use model::arch::models::shape::{MoeShape, rules};

    /// Admission counts the actual rooms' whole capacities at a small,
    /// non-pool-aligned context on every slot, and one shared spare page.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_room_admission_counts_every_slots_whole_pages() {
        use super::{KvPlanes, KvQ8, Store38, dims};
        use model::arch::models::{
            Act, Arch, ChatSpec, Ffn, Gqa, LayerSpec, Mixer, ModelSpec, OutGate, PoolRule,
            Residual, Rope, RopeMode, Selector,
        };
        use model::placement::{ModelTensors, RowTableSource, workstation::HostRead};
        let gpu = crate::Gpu::new().expect("CUDA device 0");
        let page = gpu.allocation_granularity().expect("the granularity") as u64;
        let ctx = 17;
        let stream = gpu.stream_handle();
        let rope = Rope {
            mode: RopeMode::NormTail,
            dims: geo::HEAD as u32,
            base: 1e7,
            yarn: None,
            factors: false,
        };
        let spec = ModelSpec {
            arch: Arch::Qwen4Exp,
            hidden: geo::HIDDEN as u32,
            vocab: 2,
            ctx_train: ctx as u32,
            rms_eps: 1e-6,
            layers: vec![LayerSpec {
                mixer: Mixer::Gqa(Gqa {
                    heads: geo::N_HEAD as u32,
                    kv_heads: geo::N_KV as u32,
                    head_dim: geo::HEAD as u32,
                    value_dim: geo::HEAD as u32,
                    rope,
                    qk_norm: true,
                    o_norm: false,
                    out_gate: OutGate::Packed,
                    select: Some(Selector::TokenPool {
                        heads: 4,
                        d: geo::IDX_DIM as u32,
                        top_k: 2048,
                        pool: geo::POOL as u32,
                        rule: PoolRule::Mean { rope },
                    }),
                    window: None,
                    sinks: false,
                    value_scale: None,
                }),
                ffn: Ffn::Dense {
                    ff: 1,
                    act: Act::SwiGlu { limit: None },
                },
                residual: Residual::Plain,
                extras: Vec::new(),
            }],
            mtp: Vec::new(),
            hc: None,
            engram: None,
            encoder: None,
            logit_epilogue: None,
            chat: ChatSpec {
                pre: String::new(),
                template: None,
                tools: None,
                reasoning: None,
            },
        };
        let mut inputs = model::arch::qwen35moe::place::PlanInputs {
            kv: model::arch::qwen35moe::place::KvLayout::of_spec(&spec).expect("the layout"),
            spec,
            model: ModelTensors {
                tensors: Vec::new(),
                layers: 1,
                experts: 0,
                experts_used: 0,
                bf16_words: |_| false,
            },
            room: (u64::MAX, HostRead::Given),
            row_source: RowTableSource::Split,
            arena: None,
        };
        let router = RouterDims::of(MoeShape {
            rule: rules::SOFTMAX_NORM_GATED,
            experts: geo::EXPERTS as u32,
            top_k: geo::N_USED as u32,
        })
        .expect("Qwen3.8's router");
        for kv in [KvQ8::F16, KvQ8::Q8] {
            inputs.kv = inputs.kv.in_q8_if(kv == KvQ8::Q8);
            for slots in [1, 16] {
                let stores: Vec<_> = (0..slots)
                    .map(|_| {
                        Store38::qsa(&stream, &dims(router, ctx), kv).expect("a selecting store")
                    })
                    .collect();
                for drafts in [false, true] {
                    let draft: Vec<_> = (0..if drafts { slots } else { 0 })
                        .map(|_| {
                            KvPlanes::roomed(&stream, (geo::N_KV, ctx, geo::HEAD), kv)
                                .expect("a draft store")
                        })
                        .collect();
                    let raw = runtime::stores::selecting_rows_bytes(
                        geo::N_KV,
                        geo::HEAD,
                        kv == KvQ8::Q8,
                        geo::IDX_DIM,
                        geo::POOL,
                        ctx,
                    ) * slots as u64
                        + draft.iter().map(|s| s.bytes() as u64).sum::<u64>();
                    let whole = stores.iter().map(Store38::whole).sum::<usize>()
                        + draft.iter().map(|s| s.whole()).sum::<usize>();
                    let padding = inputs
                        .room_padding(ctx as u64, slots as u64, drafts, page)
                        .expect("admitted padding");
                    assert_eq!(
                        raw + padding,
                        whole as u64 + page,
                        "{kv:?}, slots {slots}, drafts {drafts}"
                    );
                }
            }
        }
    }

    /// A ubatch walk's arena holds each position the bytes the plan counts
    /// for it (`place::UBATCH_ARENA_TOKEN_BYTES`), no more and no fewer:
    /// between [`ROUTE_ROWS`] and `place::UBATCH_PLANNED` rows its router's
    /// buffers and its selection's run stay one size, so the arena grows by
    /// its token-major rows alone.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_ubatch_arena_holds_a_position_the_bytes_the_plan_counts() {
        let gpu = crate::Gpu::new().expect("CUDA device 0");
        let router = RouterDims::of(MoeShape {
            rule: rules::SOFTMAX_NORM_GATED,
            experts: geo::EXPERTS as u32,
            top_k: geo::N_USED as u32,
        })
        .expect("Qwen3.8's router");
        let w = Weights::default();
        let by = || ByName::for_sites(&gpu, &w, std::iter::empty::<&str>(), 1).expect("no site");
        let bytes = |rows: usize| {
            let a = Arena38::wide(gpu.stream(), router, (rows, rows, 1), 8192, (by(), by()))
                .expect("a ubatch arena");
            a.bytes() as u64
        };
        let plan = model::arch::qwen35moe::place::UBATCH_ARENA_TOKEN_BYTES;
        let (lo, hi) = (
            ROUTE_ROWS,
            model::arch::qwen35moe::place::UBATCH_PLANNED as usize,
        );
        let grown = bytes(hi) - bytes(lo);
        let rows = (hi - lo) as u64;
        eprintln!("ubatch arena: {grown} B over {rows} rows, the plan counts {plan} B a row");
        assert_eq!(grown % rows, 0, "the arena grows by whole rows");
        assert_eq!(grown / rows, plan);
    }
}
