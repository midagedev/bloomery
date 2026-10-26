//! Qwen3.8's resident state on the card: each layer's store, and the arena
//! every layer of one walk shares (they run in turn) for up to `rows` tokens
//! — the decode step's one, an eager pass's up to [`PASS_ROWS`], a ubatch
//! walk's up to its load-time size ([`Arena38::wide`]). Everything is
//! allocated once at load. Buffers are token-major (`[rows][width]`)
//! unless their comment names another layout; a unit of `m <= rows` tokens
//! uses the first `m` rows.
//!
//! The store sizes are `runtime::stores`'s rules, which the plan counts by:
//! the constants the card's kernels are built for are held to them here, and
//! the load holds the stores it allocated to the rules' bytes
//! ([`store_rule_bytes`]).

use super::body::KvQ8;
use super::plan38::{GDN, geo};
use super::router::{RouterDims, RouterOut};
use super::scratch::{
    Dims, IN_IDS, IN_POS0, Inbox, Io, KvPlanes, LANE, RecStore, fresh_stamps, param_view, put_input,
};
use crate::GpuError;
use crate::flash_gqa::{listed_partials_ms_len, listed_partials_v_len_256};
use crate::hc_gated::HcScratch;
use crate::linear::{self, LinearShape};
use crate::ple;
use crate::qsa::{self, QsaScratch};
use crate::site::ByName;
use crate::tensor::WindowMut;
use cuda_core::{CudaStream, DeviceBuffer};
use runtime::hc_gated::Geometry;
use runtime::stores;

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
        && PASS_ROWS <= stores::PASS_ROWS
        && ROUTE_ROWS * (geo::N_USED + 1) <= crate::gemm::GEMM_MAX_SLOTS
);

// At the card route's ten slots a token, the default ubatch fills exactly one
// GEMM route table, so a card route of the whole ubatch takes one table.
const _: () = assert!(super::ubatch::UBATCH * geo::N_USED == crate::gemm::GEMM_MAX_SLOTS);

/// Lanes of a delta store's state: a verify of up to this many rows keeps
/// the state after each of its rows in a lane of its own
/// (`linear::delta`'s `gdn_delta_lanes`). `runtime::stores` owns the count.
pub(super) const LANES: usize = stores::DELTA_LANES;

/// The most rows a verify runs: a row a lane.
pub(super) const VERIFY_ROWS: usize = LANES;

/// A layer's store: the delta rule's recurrent state ([`LANES`] lanes), its
/// lanes' stamps (`linear::delta`'s module doc), conv ring and carry, or a
/// selecting attention layer's K/V planes with its raw and pooled indexer
/// keys (`[ctx][IDX_DIM]` and `[pools][IDX_DIM]` f16).
pub(super) enum Store38 {
    Rec {
        rec: RecStore,
        stamp: DeviceBuffer<u32>,
    },
    Qsa {
        kv: KvPlanes,
        raw: DeviceBuffer<u16>,
        pooled: DeviceBuffer<u16>,
    },
}

impl Store38 {
    /// A zeroed delta store of [`LANES`] lanes with a carry, lane 0 stamped
    /// at position 0 and every other never written.
    pub(super) fn rec(stream: &CudaStream) -> Result<Store38, GpuError> {
        Ok(Store38::Rec {
            rec: RecStore::carried(stream, GDN.lens(LANES), LANES, GDN.carry_len())?,
            stamp: DeviceBuffer::from_host(stream, &fresh_stamps(LANES))?,
        })
    }

    /// A delta store's stamps back to a fresh store's: lane 0 at position
    /// 0, every other never written. Never inside a capture.
    pub(super) fn restamp(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        if let Store38::Rec { stamp, .. } = self {
            stamp.copy_from_host(stream, &fresh_stamps(LANES))?;
        }
        Ok(())
    }

    /// A zeroed selecting store of `ctx` positions over the planes `d` cuts
    /// — f16 planes: the qwen38 family's stores carry no q8_0 form.
    pub(super) fn qsa(stream: &CudaStream, d: &Dims) -> Result<Store38, GpuError> {
        Ok(Store38::Qsa {
            kv: KvPlanes::new(stream, d, KvQ8::F16)?,
            raw: DeviceBuffer::zeroed(stream, d.ctx * geo::IDX_DIM)?,
            pooled: DeviceBuffer::zeroed(stream, qsa::pools_for(d.ctx) * geo::IDX_DIM)?,
        })
    }

    /// Device bytes.
    pub(super) fn bytes(&self) -> usize {
        match self {
            Store38::Rec { rec, stamp } => rec.bytes() + stamp.num_bytes(),
            Store38::Qsa { kv, raw, pooled } => kv.bytes() + raw.num_bytes() + pooled.num_bytes(),
        }
    }
}

/// The refusal of a delta store without the carry [`Store38::rec`] keeps.
pub(super) fn no_carry() -> GpuError {
    GpuError::state(
        "qwen4exp delta store",
        "a delta store's carry (`Store38::rec` keeps one)",
    )
}

/// The bytes `runtime::stores` counts for a delta store of [`LANES`] lanes
/// and for a selecting store at `ctx` positions: what the load's
/// allocations must equal.
pub(super) fn store_rule_bytes(ctx: usize) -> (u64, u64) {
    (
        stores::recurrent_bytes(GDN.n_v, GDN.n_k, linear::HEAD, linear::CONV_TAPS)
            + stores::delta_lane_bytes(GDN.n_v, linear::HEAD, LANES)
            + stores::delta_carry_bytes(GDN.n_v, GDN.n_k, linear::HEAD),
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
    /// The joined β·α projection: β's rows token-major, then α's
    /// (`program38::beta_alpha_rows`), which the conv reads in place.
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
    /// The selected flash's output before its gate: a pass's rows, none in a
    /// ubatch walk's arena (its prefill flash stores into `attn` under the
    /// gate).
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
    /// The by-name projections' forms and staging: the main stream's, and the
    /// branch's, whose projections run beside the main stream's
    /// ([`Branch`](crate::Branch)) and so share none of its.
    pub(super) by: ByName,
    pub(super) by_fork: ByName,
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
        Arena38::alloc(stream, rows, (rows, rows), route, ctx, by)
    }

    /// A ubatch walk's arena for up to `rows` (1..=`most`) tokens over caches
    /// of `ctx` positions: every token-major buffer for `rows` tokens, the
    /// selection's scratch for a run of [`select_run`] rows (its selecting
    /// rows run in runs of that many) and no selected-flash queries, partials
    /// or output, the router's buffers for [`ROUTE_ROWS`] (it routes in runs of that many).
    /// Load-time only.
    pub(super) fn wide(
        stream: &CudaStream,
        router: RouterDims,
        rows: usize,
        most: usize,
        ctx: usize,
        by: (ByName, ByName),
    ) -> Result<Arena38, GpuError> {
        if !(1..=most).contains(&rows) {
            return Err(GpuError::shape(
                "qwen4exp::Arena38::wide",
                format!("{rows} rows; a ubatch arena holds 1..={most}"),
            ));
        }
        let route = RouterOut::for_ubatch(stream, router, rows.min(ROUTE_ROWS))?;
        let mut a = Arena38::alloc(stream, rows, (select_run(rows, ctx), 0), route, ctx, by)?;
        // The prompt form's records take the q·k·v rows' room once the conv
        // has read them (`wide38::gdn`); a short walk needs a little more
        // than the rows.
        let work = linear::delta::prompt_work_len(GDN, rows);
        if a.gdn.x.len() < work {
            a.gdn.x = DeviceBuffer::zeroed(stream, work)?;
        }
        Ok(a)
    }

    /// The buffers for `rows` tokens, the selection's for `sel` rows and the
    /// selected flash's queries, partials and output for `parts`, the
    /// router's `route`.
    fn alloc(
        stream: &CudaStream,
        rows: usize,
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
            flash: DeviceBuffer::zeroed(stream, parts * geo::ATTN)?,
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
            &self.flash,
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
    use super::LANES;
    use crate::linear::delta::{kept_lane, row_lane};

    /// A verify from lane `c` writes its rows into `c, c + 1, …` mod
    /// [`LANES`], every row a lane of its own; keeping `k` rows moves the
    /// word to the lane row `k − 1` wrote, `(c + k − 1) mod LANES`, and
    /// keeping them all leaves it on the last row's (the common rule,
    /// `linear::delta::kept_lane`, at this body's lane count).
    #[test]
    fn a_commit_moves_the_word_to_the_last_kept_row() {
        let lanes_n = LANES as u32;
        for c in 0..lanes_n {
            let lanes: Vec<u32> = (0..lanes_n).map(|j| row_lane(c, j, lanes_n)).collect();
            let mut sorted = lanes.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..lanes_n).collect::<Vec<_>>(), "c {c}");
            assert_eq!(lanes[0], c, "row 0 overwrites the lane it read");
            for k in 1..=lanes_n {
                assert_eq!(
                    kept_lane(c, k, lanes_n),
                    (c + k - 1) % lanes_n,
                    "c {c} k {k}"
                );
                assert_eq!(kept_lane(c, k, lanes_n), lanes[k as usize - 1]);
            }
        }
    }
}
