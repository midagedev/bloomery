//! The row kernels of the batched prefill pass: the gated delta step over a
//! chunk, the attention's append and flash over a chunk's causal queries, and
//! the norm over a chunk. Each row is held to the bits of the decode kernel
//! that runs that row alone.

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Device, Elems, Fault};
use bloomery_metal::MetalBuffer;
use bloomery_metal::geom::{AttnGeom, DeltaGeom};
use bloomery_metal::kernels::{
    AttentionParams, AttentionPrefillParams, AttentionRowsParams, AttnCache, DeltaParams,
    DeltaRowStrides, DeltaRowsParams, DeltaSnaps, HeadMap, NormRowsParams, PARTIALS_PER_HEAD,
};

use super::clauses_attention::{
    CTX, CacheBack, KIND_K, KIND_Q, KIND_V, Scene, Tables, Uploaded, cache_of_for, dequant_k_for,
    dequant_v_for, gain_of, gate_of, head_row_for, load_tables_for, plant_f16_for, read_cache,
    take_qg_for, u, upload_cache,
};
use super::clauses_delta::{Fixed, KEY_HEADS, Params, draw, fixed};
use super::clauses_router;
use super::common::{self, BadArg, Edit, LAYER};
use super::fixture::Fixture;
use super::oracle;
use super::oracle::sigmoid;
use super::oracle_attention::{attend, band, norm_rope, prefill_band};
use super::{ClauseCtx, ClauseRun, Harness};

/// The delta layer's key heads.
const N_K: usize = KEY_HEADS;

/// Lanes of the state the delta clauses hold.
const LANES: usize = 2;

/// One chunk of the delta step: the inputs of `rows` rows from `pos`, and the
/// lane words the state is read and written through.
struct Chunk {
    rows: usize,
    pos: u32,
    lane_words: Vec<u32>,
    lane_at: usize,
    lane_to: usize,
    x: Vec<f32>,
    b: Vec<f32>,
    a: Vec<f32>,
    z: Vec<f32>,
    state: Vec<f32>,
    ring: Vec<f32>,
}

fn elems(n: Option<usize>, what: &str) -> Result<usize, String> {
    n.ok_or_else(|| format!("{what} does not fit"))
}

/// A chunk of `rows` rows from `pos`; the lane words are `[0, 1]` and the
/// step reads word `at` and writes word `to`.
fn chunk(
    delta: &DeltaGeom,
    rng: &mut oracle::Rng,
    rows: usize,
    pos: u32,
    (at, to): (usize, usize),
) -> Result<Chunk, String> {
    let ch = elems(delta.channels(N_K), "channels")?;
    Ok(Chunk {
        rows,
        pos,
        lane_words: vec![0, 1],
        lane_at: at,
        lane_to: to,
        x: draw(rng, rows * ch, 0.3),
        b: draw(rng, rows * delta.value_heads, 1.0),
        a: draw(rng, rows * delta.value_heads, 2.0),
        z: draw(rng, rows * delta.value_heads * delta.head_cols, 1.0),
        state: draw(rng, elems(delta.state_elems(LANES), "state")?, 0.04),
        ring: draw(rng, elems(delta.ring_elems(N_K), "ring")?, 0.05),
    })
}

/// Every window a delta launch binds, uploaded from a chunk.
struct Bufs {
    x: MetalBuffer<f32>,
    conv_w: MetalBuffer<f32>,
    b: MetalBuffer<f32>,
    a: MetalBuffer<f32>,
    dt: MetalBuffer<f32>,
    ssm_a: MetalBuffer<f32>,
    z: MetalBuffer<f32>,
    norm_w: MetalBuffer<f32>,
    lane: MetalBuffer<u32>,
    state: MetalBuffer<f32>,
    ring: MetalBuffer<f32>,
    partials: MetalBuffer<f32>,
    g: MetalBuffer<f32>,
    prep: MetalBuffer<f32>,
}

/// What a delta launch left behind.
struct Out {
    g: Vec<f32>,
    partials: Vec<f32>,
    state: Vec<f32>,
    ring: Vec<f32>,
    fault: Option<Fault>,
}

/// f32s of one row of the numerator.
fn g_row(delta: &DeltaGeom) -> usize {
    delta.value_heads * delta.head_cols
}

/// f32s of one row of the partials.
fn p_row(delta: &DeltaGeom) -> usize {
    delta.value_heads * PARTIALS_PER_HEAD
}

/// f32s of one prepared row of the delta layer.
fn prep_row(delta: &DeltaGeom) -> Result<usize, String> {
    elems(delta.prepared_elems(N_K), "the prepared row")
}

impl Bufs {
    fn new(h: &Harness<'_>, fx: &Fixed, c: &Chunk) -> Result<Self, String> {
        let (device, stream) = (h.device, h.stream);
        let delta = &h.geom().delta;
        let (g_row, p_row) = (g_row(delta), p_row(delta));
        Ok(Self {
            x: device.upload(stream, &c.x).map_err(why)?,
            conv_w: device.upload(stream, &fx.conv_w).map_err(why)?,
            b: device.upload(stream, &c.b).map_err(why)?,
            a: device.upload(stream, &c.a).map_err(why)?,
            dt: device.upload(stream, &fx.dt).map_err(why)?,
            ssm_a: device.upload(stream, &fx.ssm_a).map_err(why)?,
            z: device.upload(stream, &c.z).map_err(why)?,
            norm_w: device.upload(stream, &fx.norm_w).map_err(why)?,
            lane: device.upload(stream, &c.lane_words).map_err(why)?,
            state: device.upload(stream, &c.state).map_err(why)?,
            ring: device.upload(stream, &c.ring).map_err(why)?,
            partials: device
                .upload(stream, &vec![0f32; c.rows * p_row])
                .map_err(why)?,
            g: device
                .upload(stream, &vec![0f32; c.rows * g_row])
                .map_err(why)?,
            prep: device
                .upload(stream, &vec![0f32; c.rows * prep_row(delta)?])
                .map_err(why)?,
        })
    }

    fn read(&self, h: &Harness<'_>, c: &Chunk) -> Result<Out, String> {
        let (device, stream) = (h.device, h.stream);
        let delta = &h.geom().delta;
        let mut out = Out {
            g: vec![0f32; c.rows * g_row(delta)],
            partials: vec![0f32; c.rows * p_row(delta)],
            state: vec![0f32; c.state.len()],
            ring: vec![0f32; c.ring.len()],
            fault: h.fault()?,
        };
        device
            .read(stream, self.g.whole().map_err(why)?, &mut out.g)
            .map_err(why)?;
        device
            .read(
                stream,
                self.partials.whole().map_err(why)?,
                &mut out.partials,
            )
            .map_err(why)?;
        device
            .read(stream, self.state.whole().map_err(why)?, &mut out.state)
            .map_err(why)?;
        device
            .read(stream, self.ring.whole().map_err(why)?, &mut out.ring)
            .map_err(why)?;
        Ok(out)
    }
}

/// The chunk as one rows launch.
fn run_rows(h: &Harness<'_>, fx: &Fixed, c: &Chunk) -> Result<Out, String> {
    run_rows_copied(h, fx, c, &[]).map(|(out, _)| out)
}

/// The lanes and rings a rows launch copied, copy after copy.
struct Copies {
    lane: Vec<f32>,
    ring: Vec<f32>,
}

/// The chunk as one rows launch that copies its stores after the rows
/// `after` (none: no copies). The copies' buffers start NaN, so a copy the
/// launch did not write reads as no number.
fn run_rows_copied(
    h: &Harness<'_>,
    fx: &Fixed,
    c: &Chunk,
    after: &[u32],
) -> Result<(Out, Copies), String> {
    let delta = &h.geom().delta;
    let (g_row, p_row) = (g_row(delta), p_row(delta));
    let mut bufs = Bufs::new(h, fx, c)?;
    let strides = DeltaRowStrides::of(delta, N_K).ok_or("the strides do not fit")?;
    let prep_len = c.rows * prep_row(delta)?;
    let lane_len = after.len() * elems(delta.state_elems(1), "a lane")?;
    let ring_len = after.len() * c.ring.len();
    let poison = |n: usize| -> Result<MetalBuffer<f32>, String> {
        h.device
            .upload(h.stream, &vec![f32::NAN; n.max(1)])
            .map_err(why)
    };
    let mut lane_copies = poison(lane_len)?;
    let mut ring_copies = poison(ring_len)?;
    h.clear_fault()?;
    h.encode(|encoder| {
        let snaps = if after.is_empty() {
            None
        } else {
            Some(DeltaSnaps {
                after,
                lane: lane_copies.window_mut(Elems(0), Elems(lane_len))?,
                ring: ring_copies.window_mut(Elems(0), Elems(ring_len))?,
            })
        };
        h.kernels.delta_rows(
            encoder,
            DeltaRowsParams {
                x: bufs.x.window(Elems(0), Elems(c.x.len()))?,
                conv_w: bufs.conv_w.window(Elems(0), Elems(fx.conv_w.len()))?,
                b_raw: bufs.b.window(Elems(0), Elems(c.b.len()))?,
                a_raw: bufs.a.window(Elems(0), Elems(c.a.len()))?,
                dt_bias: bufs.dt.window(Elems(0), Elems(fx.dt.len()))?,
                ssm_a: bufs.ssm_a.window(Elems(0), Elems(fx.ssm_a.len()))?,
                z: bufs.z.window(Elems(0), Elems(c.z.len()))?,
                norm_w: bufs.norm_w.window(Elems(0), Elems(fx.norm_w.len()))?,
                lane: bufs.lane.window(Elems(0), Elems(c.lane_words.len()))?,
                state: bufs.state.window_mut(Elems(0), Elems(c.state.len()))?,
                ring: bufs.ring.window_mut(Elems(0), Elems(c.ring.len()))?,
                partials: bufs.partials.window_mut(Elems(0), Elems(c.rows * p_row))?,
                g: bufs.g.window_mut(Elems(0), Elems(c.rows * g_row))?,
                prep: bufs.prep.window_mut(Elems(0), Elems(prep_len))?,
                faults: h.device.fault_word(),
                n_v: delta.value_heads,
                head: delta.head_cols,
                taps: delta.conv_taps,
                n_k: N_K,
                map: HeadMap::Tiled,
                lanes: LANES,
                lane_at: c.lane_at,
                lane_to: c.lane_to,
                layer: LAYER,
                rows: c.rows,
                pos: c.pos,
                strides,
                eps: fx.eps,
                snaps,
            },
        )
    })?;
    let out = bufs.read(h, c)?;
    let mut copies = Copies {
        lane: vec![0f32; lane_len],
        ring: vec![0f32; ring_len],
    };
    if !after.is_empty() {
        h.device
            .read(
                h.stream,
                lane_copies.window(Elems(0), Elems(lane_len)).map_err(why)?,
                &mut copies.lane,
            )
            .map_err(why)?;
        h.device
            .read(
                h.stream,
                ring_copies.window(Elems(0), Elems(ring_len)).map_err(why)?,
                &mut copies.ring,
            )
            .map_err(why)?;
    }
    Ok((out, copies))
}

/// The chunk as `rows` stepped launches in one command buffer, row `r` at
/// position `pos + r`. The first step reads the chunk's read lane and writes
/// its write lane; the later steps update the write lane in place, so the
/// state the rows kernel carries in registers is the state the steps carry in
/// memory.
fn run_stepped(h: &Harness<'_>, fx: &Fixed, c: &Chunk) -> Result<Out, String> {
    let delta = &h.geom().delta;
    let (g_row, p_row) = (g_row(delta), p_row(delta));
    let mut bufs = Bufs::new(h, fx, c)?;
    let ch = elems(delta.channels(N_K), "channels")?;
    h.clear_fault()?;
    h.encode(|encoder| {
        for r in 0..c.rows {
            let pos = [c.pos + r as u32];
            let lane_at = if r == 0 { c.lane_at } else { c.lane_to };
            h.kernels.delta(
                encoder,
                DeltaParams {
                    x: bufs.x.window(Elems(r * ch), Elems(ch))?,
                    conv_w: bufs.conv_w.window(Elems(0), Elems(fx.conv_w.len()))?,
                    b_raw: bufs
                        .b
                        .window(Elems(r * delta.value_heads), Elems(delta.value_heads))?,
                    a_raw: bufs
                        .a
                        .window(Elems(r * delta.value_heads), Elems(delta.value_heads))?,
                    dt_bias: bufs.dt.window(Elems(0), Elems(fx.dt.len()))?,
                    ssm_a: bufs.ssm_a.window(Elems(0), Elems(fx.ssm_a.len()))?,
                    z: bufs.z.window(Elems(r * g_row), Elems(g_row))?,
                    norm_w: bufs.norm_w.window(Elems(0), Elems(fx.norm_w.len()))?,
                    lane: bufs.lane.window(Elems(0), Elems(c.lane_words.len()))?,
                    state: bufs.state.window_mut(Elems(0), Elems(c.state.len()))?,
                    ring: bufs.ring.window_mut(Elems(0), Elems(c.ring.len()))?,
                    partials: bufs.partials.window_mut(Elems(r * p_row), Elems(p_row))?,
                    g: bufs.g.window_mut(Elems(r * g_row), Elems(g_row))?,
                    faults: h.device.fault_word(),
                    n_v: delta.value_heads,
                    head: delta.head_cols,
                    taps: delta.conv_taps,
                    n_k: N_K,
                    map: HeadMap::Tiled,
                    lanes: LANES,
                    lane_at,
                    lane_to: c.lane_to,
                    layer: LAYER,
                    pos: &pos,
                    eps: fx.eps,
                },
            )?;
        }
        Ok(())
    })?;
    bufs.read(h, c)
}

/// The first word of `got` that is not the bits of `want`, named.
pub(super) fn same_bits(what: &str, got: &[f32], want: &[f32]) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!("{what}: {} words, not {}", got.len(), want.len()));
    }
    match got
        .iter()
        .zip(want)
        .position(|(g, w)| g.to_bits() != w.to_bits())
    {
        None => Ok(()),
        Some(i) => Err(format!(
            "{what}[{i}] is {:?} ({:#010x}), expected {:?} ({:#010x})",
            got[i],
            got[i].to_bits(),
            want[i],
            want[i].to_bits()
        )),
    }
}

/// The rows launch left what the stepped launches left: the numerator and
/// partials of every row, the whole state, the whole ring and the fault word.
fn same_out(rows: &Out, stepped: &Out) -> Result<(), String> {
    same_bits("g", &rows.g, &stepped.g)?;
    same_bits("partials", &rows.partials, &stepped.partials)?;
    same_bits("state", &rows.state, &stepped.state)?;
    same_bits("ring", &rows.ring, &stepped.ring)?;
    if rows.fault != stepped.fault {
        let show = |f: &Option<Fault>| f.map_or("none".to_string(), |f| f.to_string());
        return Err(format!(
            "the fault word is {}, the stepped launches' {}",
            show(&rows.fault),
            show(&stepped.fault)
        ));
    }
    Ok(())
}

/// The cases of the delta clause: rows, first position, and the read and the
/// write lane word. A first position of 0 puts taps before the first
/// position; 7 and 3 with 12 and 64 rows wrap the ring.
const DELTA_CASES: [(usize, u32, (usize, usize)); 7] = [
    (1, 4, (0, 1)),
    (9, 0, (0, 1)),
    (9, 20, (1, 0)),
    (12, 7, (0, 1)),
    (12, 3, (1, 1)),
    (64, 5, (0, 0)),
    (64, 9, (1, 0)),
];

/// `m` rows are the bits of `m` stepped launches: numerators, partials, state,
/// ring and fault word.
pub fn delta_rows(h: &Harness<'_>, params: Params<'_>) -> Clause {
    let delta = &h.geom().delta;
    let fx = fixed(delta, params)?;
    let mut rng = oracle::Rng::new(0xD0A5);
    let mut parts = Vec::new();
    for (rows, pos, lanes) in DELTA_CASES {
        let c = chunk(delta, &mut rng, rows, pos, lanes)?;
        let a = run_rows(h, &fx, &c)?;
        let b = run_stepped(h, &fx, &c)?;
        if let Some(f) = a.fault {
            return Err(format!("{rows} rows from {pos} raised {f}"));
        }
        same_out(&a, &b).map_err(|e| format!("{rows} rows from {pos}, lanes {lanes:?}: {e}"))?;
        // The state moved: a rows launch that wrote nothing is not the
        // stepped launches' bits by luck of an unchanged lane.
        if a.state == c.state {
            return Err(format!("{rows} rows from {pos} left the state as it was"));
        }
        if rows >= delta.ring_rows && a.ring == c.ring {
            return Err(format!("{rows} rows from {pos} left the ring as it was"));
        }
        parts.push(format!("{rows}@{pos}"));
    }
    Ok(Verdict::Noted(format!(
        "rows@first position {} are the stepped launches' bits",
        parts.join(", ")
    )))
}

/// The first `rows` rows of `c`: the same state, ring and lane words.
fn head_of(delta: &DeltaGeom, c: &Chunk, rows: usize) -> Result<Chunk, String> {
    let ch = elems(delta.channels(N_K), "channels")?;
    Ok(Chunk {
        rows,
        pos: c.pos,
        lane_words: c.lane_words.clone(),
        lane_at: c.lane_at,
        lane_to: c.lane_to,
        x: c.x[..rows * ch].to_vec(),
        b: c.b[..rows * delta.value_heads].to_vec(),
        a: c.a[..rows * delta.value_heads].to_vec(),
        z: c.z[..rows * g_row(delta)].to_vec(),
        state: c.state.clone(),
        ring: c.ring.clone(),
    })
}

/// A copies case: rows, first position, lane words, the copies' rows.
type SnapCase = (usize, u32, (usize, usize), &'static [u32]);

/// The cases of the copies clause: rows, first position, the read and the
/// write lane word, and the rows the copies follow. A copy after row 0 takes
/// every ring slot but one from the ring the chunk found; one after a row
/// past the ring's length takes none; a first position of 0 has slots no
/// position wrote; a copy after the last row is the chunk's own end; a
/// chunk ending at the last u32 position takes its ring slots without the
/// slot arithmetic wrapping.
const SNAP_CASES: [SnapCase; 5] = [
    (12, 7, (0, 1), &[3, 9]),
    (64, 5, (0, 0), &[0, 40]),
    (9, 0, (0, 1), &[1, 8]),
    (20, 3, (1, 0), &[12]),
    (9, u32::MAX - 9, (0, 1), &[8]),
];

/// A rows launch's copies are the bits a chunk ending at their rows leaves:
/// copy `k`'s lane is the write lane, and its ring the ring, of the rows
/// launch over rows `0..=after[k]` of the same inputs; and the copies move
/// none of the launch's own outputs (numerators, partials, state, ring,
/// fault word).
pub fn delta_rows_snap(h: &Harness<'_>, params: Params<'_>) -> Clause {
    let delta = &h.geom().delta;
    let fx = fixed(delta, params)?;
    let lane_len = elems(delta.state_elems(1), "a lane")?;
    let mut rng = oracle::Rng::new(0xD0A7);
    let mut parts = Vec::new();
    for (rows, pos, lanes, after) in SNAP_CASES {
        let c = chunk(delta, &mut rng, rows, pos, lanes)?;
        let at = format!("{rows} rows from {pos}, lanes {lanes:?}, copies after {after:?}");
        let (got, copies) = run_rows_copied(h, &fx, &c, after)?;
        if let Some(f) = got.fault {
            return Err(format!("{at}: raised {f}"));
        }
        let plain = run_rows(h, &fx, &c)?;
        same_out(&got, &plain).map_err(|e| format!("{at}, against no copies: {e}"))?;
        let ring_len = c.ring.len();
        for (k, &r) in after.iter().enumerate() {
            let end = run_rows(h, &fx, &head_of(delta, &c, r as usize + 1)?)?;
            let lane = &end.state[c.lane_to * lane_len..(c.lane_to + 1) * lane_len];
            same_bits(
                &format!("{at}: copy {k}'s lane"),
                &copies.lane[k * lane_len..(k + 1) * lane_len],
                lane,
            )?;
            same_bits(
                &format!("{at}: copy {k}'s ring"),
                &copies.ring[k * ring_len..(k + 1) * ring_len],
                &end.ring,
            )?;
        }
        parts.push(format!("{rows}@{pos} after {after:?}"));
    }
    Ok(Verdict::Noted(format!(
        "rows@first position {}: each copy is the chunk ending at its row, bit for bit, and \
         the launch's own outputs are the launch's without copies",
        parts.join(", ")
    )))
}

/// What one poisoned chunk raises and where the NaN lands.
struct Poison {
    what: &'static str,
    site: FaultSite,
    /// Whether every numerator word is NaN (a lane that cannot be read leaves
    /// no column a number), else only the columns the input reaches.
    everywhere: bool,
    apply: fn(&DeltaGeom, &mut Chunk, usize),
}

fn nan_x(_delta: &DeltaGeom, c: &mut Chunk, ch: usize) {
    // A query channel of key head 0, in row 5: rows 5 to 8 read it as a tap.
    c.x[5 * ch + 100] = f32::NAN;
}

fn nan_value_x(delta: &DeltaGeom, c: &mut Chunk, ch: usize) {
    // A value channel of value head 2, column 9, in row 6: rows 6 to 9 read it
    // as a tap, and only the column block that holds it is unusable.
    c.x[6 * ch + (2 * N_K + 2) * delta.head_cols + 9] = f32::NAN;
}

fn inf_b(delta: &DeltaGeom, c: &mut Chunk, _ch: usize) {
    c.b[8 * delta.value_heads + 3] = f32::INFINITY;
}

fn nan_state(delta: &DeltaGeom, c: &mut Chunk, _ch: usize) {
    // One key of one column of value head 2, in the lane the chunk reads.
    c.state[((2 * delta.head_cols) + 9) * delta.head_cols + 17] = f32::NAN;
}

fn lane_past(_delta: &DeltaGeom, c: &mut Chunk, _ch: usize) {
    c.lane_words = vec![0, 7];
}

/// A row with a non-finite input raises the decode's site and is NaN on the
/// columns the stepped launches make NaN; the other rows are unaffected.
pub fn delta_rows_nan(h: &Harness<'_>, params: Params<'_>) -> Clause {
    let delta = &h.geom().delta;
    let fx = fixed(delta, params)?;
    let ch = elems(delta.channels(N_K), "channels")?;
    let cases = [
        Poison {
            what: "a NaN conv input",
            site: FaultSite::LinearConv,
            everywhere: false,
            apply: nan_x,
        },
        Poison {
            what: "a NaN value conv input",
            site: FaultSite::LinearConv,
            everywhere: false,
            apply: nan_value_x,
        },
        Poison {
            what: "an infinite β input",
            site: FaultSite::LinearConv,
            everywhere: false,
            apply: inf_b,
        },
        Poison {
            what: "a NaN state key",
            site: FaultSite::DeltaLane,
            everywhere: false,
            apply: nan_state,
        },
        Poison {
            what: "a write lane word past the lanes",
            site: FaultSite::DeltaLane,
            everywhere: true,
            apply: lane_past,
        },
    ];
    let mut parts = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        // In place: the stepped launches then carry the state in memory the
        // way the rows launch carries it in registers.
        let mut rng = oracle::Rng::new(0xD0A6 + i as u64);
        let mut c = chunk(delta, &mut rng, 12, 7, (0, 0))?;
        if case.what.starts_with("a write lane") {
            c.lane_at = 0;
            c.lane_to = 1;
        }
        (case.apply)(delta, &mut c, ch);
        let a = run_rows(h, &fx, &c)?;
        let b = run_stepped(h, &fx, &c)?;
        same_out(&a, &b).map_err(|e| format!("{}: {e}", case.what))?;
        match a.fault {
            Some(f) if f.layer == LAYER && f.sites & (1 << case.site as u32) != 0 => {}
            Some(f) => {
                return Err(format!(
                    "{} raised {f}, not {}",
                    case.what,
                    case.site.name()
                ));
            }
            None => return Err(format!("{} raised nothing", case.what)),
        }
        let nans = a.g.iter().filter(|v| v.is_nan()).count();
        if nans == 0 {
            return Err(format!("{} left no NaN in the numerator", case.what));
        }
        if (nans == a.g.len()) != case.everywhere {
            return Err(format!(
                "{} made {nans} of {} numerator words NaN",
                case.what,
                a.g.len()
            ));
        }
        parts.push(format!("{} ({nans} NaN words)", case.what));
    }
    Ok(Verdict::Noted(format!(
        "each is the stepped launches' fault word, numerators and state: {}",
        parts.join(", ")
    )))
}

/// One chunk of the attention clauses: the first position, the rows and the
/// split-K segments of every row's flash.
struct AttnCase {
    from: usize,
    rows: usize,
    segments: usize,
}

/// A first position of 0 starts on an empty cache; 37 reads thirty-seven
/// planted rows. Three segments split 33 keys; five split 70.
const ATTN_CASES: [AttnCase; 6] = [
    AttnCase {
        from: 0,
        rows: 1,
        segments: 1,
    },
    AttnCase {
        from: 0,
        rows: 9,
        segments: 1,
    },
    AttnCase {
        from: 0,
        rows: 33,
        segments: 3,
    },
    AttnCase {
        from: 37,
        rows: 1,
        segments: 5,
    },
    AttnCase {
        from: 37,
        rows: 9,
        segments: 5,
    },
    AttnCase {
        from: 37,
        rows: 33,
        segments: 5,
    },
];

/// The projections of a chunk, rows end to end: every row's query and gate,
/// raw keys and raw values differ from every other row's.
struct AttnIn {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

fn attn_inputs(attn: &AttnGeom, rows: usize) -> Result<AttnIn, String> {
    let q_n = attn.qg_len().ok_or("the query length wraps")?;
    let kv_n = attn.kv_heads * attn.head;
    let mut inp = AttnIn {
        q: vec![0f32; rows * q_n],
        k: vec![0f32; rows * kv_n],
        v: vec![0f32; rows * kv_n],
    };
    for r in 0..rows {
        for head in 0..attn.query_heads {
            let at = r * q_n + head * attn.qg_stride;
            inp.q[at..at + attn.head].copy_from_slice(&head_row_for(
                attn,
                u(r * attn.query_heads + head),
                KIND_Q,
            ));
            if attn.gated {
                for dim in 0..attn.head {
                    inp.q[at + attn.head + dim] = gate_of(u(dim + r));
                }
            }
        }
        for kv in 0..attn.kv_heads {
            let at = r * kv_n + kv * attn.head;
            inp.k[at..at + attn.head].copy_from_slice(&head_row_for(
                attn,
                u(r * attn.kv_heads + kv),
                KIND_K,
            ));
            inp.v[at..at + attn.head].copy_from_slice(&head_row_for(
                attn,
                u(r * attn.kv_heads + kv),
                KIND_V,
            ));
        }
    }
    Ok(inp)
}

/// What a chunk's attention left: the outputs row after row, the cache, the
/// fault word and the fold tickets.
struct AttnSeen {
    out: Vec<f32>,
    cache: CacheBack,
    fault: Option<Fault>,
    ticket: Vec<u32>,
}

/// The buffers of one chunk: the cache planted with the rows before `from`,
/// the projections, the tables, and the outputs, partials and tickets.
struct AttnBufs {
    q: MetalBuffer<f32>,
    k: MetalBuffer<f32>,
    v: MetalBuffer<f32>,
    q_gain: MetalBuffer<f32>,
    k_gain: MetalBuffer<f32>,
    rope: MetalBuffer<f32>,
    partials: MetalBuffer<f32>,
    out: MetalBuffer<f32>,
    ticket: MetalBuffer<u32>,
    cache: Uploaded,
}

impl AttnBufs {
    fn new(
        h: &Harness<'_>,
        tables: &Tables,
        case: &AttnCase,
        inp: &AttnIn,
        ctx: usize,
    ) -> Result<Self, String> {
        let (device, stream) = (h.device, h.stream);
        let attn = &h.geom().attn;
        let out_n = attn.out_len().ok_or("the output length wraps")?;
        let scene = Scene {
            segments: case.segments,
            ctx,
            pos: u(case.from),
            n_keys: u(case.from),
            prefill: case.from,
            q8: false,
        };
        let planted = plant_f16_for(attn, &scene)?;
        let partial_n = attn
            .partials_len(case.segments)
            .ok_or("the partial length wraps")?;
        Ok(Self {
            q: device.upload(stream, &inp.q).map_err(why)?,
            k: device.upload(stream, &inp.k).map_err(why)?,
            v: device.upload(stream, &inp.v).map_err(why)?,
            q_gain: device.upload(stream, &tables.q_gain).map_err(why)?,
            k_gain: device.upload(stream, &tables.k_gain).map_err(why)?,
            rope: device
                .upload(stream, &tables.rope[..ctx * attn.rot])
                .map_err(why)?,
            partials: device
                .upload(stream, &vec![0f32; case.rows * partial_n])
                .map_err(why)?,
            out: device
                .upload(stream, &vec![0f32; case.rows * out_n])
                .map_err(why)?,
            ticket: device
                .upload(stream, &vec![0u32; case.rows * attn.kv_heads])
                .map_err(why)?,
            cache: upload_cache(device, stream, &planted)?,
        })
    }

    fn read(&self, h: &Harness<'_>, case: &AttnCase) -> Result<AttnSeen, String> {
        let (device, stream) = (h.device, h.stream);
        let attn = &h.geom().attn;
        let out_n = attn.out_len().ok_or("the output length wraps")?;
        let mut out = vec![0f32; case.rows * out_n];
        device
            .read(stream, self.out.whole().map_err(why)?, &mut out)
            .map_err(why)?;
        let mut ticket = vec![1u32; case.rows * attn.kv_heads];
        device
            .read(stream, self.ticket.whole().map_err(why)?, &mut ticket)
            .map_err(why)?;
        Ok(AttnSeen {
            out,
            cache: read_cache(device, stream, &self.cache)?,
            fault: h.fault()?,
            ticket,
        })
    }
}

/// The chunk as one rows launch.
fn attn_rows_run(
    h: &Harness<'_>,
    tables: &Tables,
    case: &AttnCase,
    inp: &AttnIn,
    ctx: usize,
) -> Result<AttnSeen, String> {
    let attn = &h.geom().attn;
    let mut b = AttnBufs::new(h, tables, case, inp, ctx)?;
    let out_n = attn.out_len().ok_or("the output length wraps")?;
    let q_n = attn.qg_len().ok_or("the query length wraps")?;
    let partial_n = attn
        .partials_len(case.segments)
        .ok_or("the partial length wraps")?;
    let kv_n = attn.kv_heads * attn.head;
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.attention_rows(
            encoder,
            AttentionRowsParams {
                head_dim: attn.head,
                query_heads: attn.query_heads,
                kv_heads: attn.kv_heads,
                segments: case.segments,
                ctx,
                eps: tables.eps,
                layer: LAYER,
                from: u(case.from),
                rows: case.rows,
                q: b.q.window(Elems(0), Elems(inp.q.len()))?,
                q_stride: q_n,
                k: b.k.window(Elems(0), Elems(inp.k.len()))?,
                k_stride: kv_n,
                v: b.v.window(Elems(0), Elems(inp.v.len()))?,
                v_stride: kv_n,
                q_gain: gain_of(attn, &b.q_gain, 0)?,
                k_gain: gain_of(attn, &b.k_gain, 0)?,
                rope: b.rope.window(Elems(0), Elems(ctx * attn.rot))?,
                cache: cache_of_for(attn, &mut b.cache, ctx)?,
                partials: b
                    .partials
                    .window_mut(Elems(0), Elems(case.rows * partial_n))?,
                out: b.out.window_mut(Elems(0), Elems(case.rows * out_n))?,
                out_stride: out_n,
                fault: h.device.fault_word(),
                ticket: b
                    .ticket
                    .window_mut(Elems(0), Elems(case.rows * attn.kv_heads))?,
            },
        )
    })?;
    b.read(h, case)
}

/// The chunk as `rows` decode launches in one command buffer, row `r` at
/// position `from + r` over `from + r + 1` keys, the same segment count.
fn attn_stepped_run(
    h: &Harness<'_>,
    tables: &Tables,
    case: &AttnCase,
    inp: &AttnIn,
) -> Result<AttnSeen, String> {
    let attn = &h.geom().attn;
    let mut b = AttnBufs::new(h, tables, case, inp, CTX)?;
    let out_n = attn.out_len().ok_or("the output length wraps")?;
    let q_n = attn.qg_len().ok_or("the query length wraps")?;
    let partial_n = attn
        .partials_len(case.segments)
        .ok_or("the partial length wraps")?;
    let kv_n = attn.kv_heads * attn.head;
    h.clear_fault()?;
    h.encode(|encoder| {
        for r in 0..case.rows {
            h.kernels.attention_flash(
                encoder,
                AttentionParams {
                    head_dim: attn.head,
                    query_heads: attn.query_heads,
                    kv_heads: attn.kv_heads,
                    segments: case.segments,
                    ctx: CTX,
                    eps: tables.eps,
                    layer: LAYER,
                    pos: u(case.from + r),
                    n_keys: u(case.from + r + 1),
                    q: b.q.window(Elems(r * q_n), Elems(q_n))?,
                    k: b.k.window(Elems(r * kv_n), Elems(kv_n))?,
                    v: b.v.window(Elems(r * kv_n), Elems(kv_n))?,
                    q_gain: gain_of(attn, &b.q_gain, 0)?,
                    k_gain: gain_of(attn, &b.k_gain, 0)?,
                    rope: b.rope.window(Elems(0), Elems(CTX * attn.rot))?,
                    cache: cache_of_for(attn, &mut b.cache, CTX)?,
                    partials: b
                        .partials
                        .window_mut(Elems(r * partial_n), Elems(partial_n))?,
                    out: b.out.window_mut(Elems(r * out_n), Elems(out_n))?,
                    fault: h.device.fault_word(),
                    ticket: b
                        .ticket
                        .window_mut(Elems(r * attn.kv_heads), Elems(attn.kv_heads))?,
                },
            )?;
        }
        Ok(())
    })?;
    b.read(h, case)
}

/// The first word of `got` that differs from `want`'s, named.
fn same_words<T: PartialEq + Copy + core::fmt::Debug>(
    what: &str,
    got: &[T],
    want: &[T],
) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!("{what}: {} words, not {}", got.len(), want.len()));
    }
    match common::first_diff(got, want) {
        None => Ok(()),
        Some((i, g, w)) => Err(format!("{what}[{i}] is {g:?}, the decode launch's {w:?}")),
    }
}

/// Every cache word, holes included, is the decode launches'.
fn same_cache(rows: &CacheBack, stepped: &CacheBack) -> Result<(), String> {
    match (rows, stepped) {
        (CacheBack::F16 { k: rk, v: rv }, CacheBack::F16 { k: sk, v: sv }) => {
            same_words("k cache", rk, sk)?;
            same_words("v cache", rv, sv)
        }
        _ => Err("the rows launch and the decode launches ran on different caches".to_string()),
    }
}

/// The key and value rows the chunk appended are the decode launches' bits:
/// the key normed and roped at its absolute position, the value raw, each
/// stored as the f16 the decode stores, and the rows around them untouched.
pub fn norm_rope_rows(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let attn = &h.geom().attn;
    let tables = load_tables_for(attn, fixture)?;
    let mut parts = Vec::new();
    for case in &ATTN_CASES {
        let inp = attn_inputs(attn, case.rows)?;
        let a = attn_rows_run(h, &tables, case, &inp, CTX)?;
        let b = attn_stepped_run(h, &tables, case, &inp)?;
        same_cache(&a.cache, &b.cache)
            .map_err(|e| format!("{} rows from {}: {e}", case.rows, case.from))?;
        parts.push(format!("{}@{}", case.rows, case.from));
    }
    Ok(Verdict::Noted(format!(
        "rows@first position {} append the decode launches' cache rows",
        parts.join(", ")
    )))
}

/// Each row of the chunk's attention is the bits of the decode launch at its
/// position: the gated output, the fault word, and the tickets the folds reset.
pub fn attn_rows_v0(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let attn = &h.geom().attn;
    let tables = load_tables_for(attn, fixture)?;
    let mut parts = Vec::new();
    for case in &ATTN_CASES {
        let inp = attn_inputs(attn, case.rows)?;
        let a = attn_rows_run(h, &tables, case, &inp, CTX)?;
        let b = attn_stepped_run(h, &tables, case, &inp)?;
        let at = format!("{} rows from {}", case.rows, case.from);
        if let Some(f) = a.fault {
            return Err(format!("{at} raised {f}"));
        }
        if a.out.iter().any(|v| v.is_nan()) {
            return Err(format!("{at} wrote a NaN"));
        }
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        same_words("out", &bits(&a.out), &bits(&b.out)).map_err(|e| format!("{at}: {e}"))?;
        if a.fault != b.fault {
            return Err(format!("{at}: the fault words differ"));
        }
        if a.ticket.iter().any(|&t| t != 0) {
            return Err(format!(
                "{at}: a fold's ticket reads {:?}, not zero",
                a.ticket
            ));
        }
        parts.push(format!("{}@{}/{}", case.rows, case.from, case.segments));
    }
    Ok(Verdict::Noted(format!(
        "rows@first position/segments {} are the decode launches' bits",
        parts.join(", ")
    )))
}

/// Cache rows of the prefill clauses: the longest chunk and its first
/// position, rounded up to a tile.
const PREFILL_CTX: usize = 352;

/// The prefill cases: the first position and the rows. A first position of 0
/// starts on an empty cache; 37 reads thirty-seven planted rows. A threadgroup
/// takes 32 consecutive (position, query head) rows of a key head, so 1, 7, 8
/// and 9 rows end the chunk at different offsets of the last block; a
/// position's heads straddle two SIMD groups when the group size does not
/// divide eight. 64, 257 and 300 rows cover the diagonal tile at every offset
/// of the block, 257 the first position past a 256-row cut.
const PREFILL_CASES: [(usize, usize); 14] = [
    (0, 1),
    (0, 7),
    (0, 8),
    (0, 9),
    (0, 64),
    (0, 257),
    (0, 300),
    (37, 1),
    (37, 7),
    (37, 8),
    (37, 9),
    (37, 64),
    (37, 257),
    (37, 300),
];

/// Split-K segments of the decode flash the prefill is compared with.
const PREFILL_V0_SEGMENTS: usize = 5;

/// The chunk as one prefill launch.
fn attn_prefill_run(
    h: &Harness<'_>,
    tables: &Tables,
    case: &AttnCase,
    inp: &AttnIn,
    ctx: usize,
) -> Result<AttnSeen, String> {
    let attn = &h.geom().attn;
    let mut b = AttnBufs::new(h, tables, case, inp, ctx)?;
    let out_n = attn.out_len().ok_or("the output length wraps")?;
    let q_n = attn.qg_len().ok_or("the query length wraps")?;
    let kv_n = attn.kv_heads * attn.head;
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.attention_prefill(
            encoder,
            AttentionPrefillParams {
                head_dim: attn.head,
                query_heads: attn.query_heads,
                kv_heads: attn.kv_heads,
                ctx,
                eps: tables.eps,
                layer: LAYER,
                from: u(case.from),
                rows: case.rows,
                q: b.q.window(Elems(0), Elems(inp.q.len()))?,
                q_stride: q_n,
                k: b.k.window(Elems(0), Elems(inp.k.len()))?,
                k_stride: kv_n,
                v: b.v.window(Elems(0), Elems(inp.v.len()))?,
                v_stride: kv_n,
                q_gain: gain_of(attn, &b.q_gain, 0)?,
                k_gain: gain_of(attn, &b.k_gain, 0)?,
                rope: b.rope.window(Elems(0), Elems(ctx * attn.rot))?,
                cache: cache_of_for(attn, &mut b.cache, ctx)?,
                out: b.out.window_mut(Elems(0), Elems(case.rows * out_n))?,
                out_stride: out_n,
                fault: h.device.fault_word(),
            },
        )
    })?;
    b.read(h, case)
}

/// The prefill launch and the decode flash's rows launch on one chunk.
fn prefill_pair(
    h: &Harness<'_>,
    tables: &Tables,
    case: &AttnCase,
    inp: &AttnIn,
) -> Result<(AttnSeen, AttnSeen), String> {
    let pre = attn_prefill_run(h, tables, case, inp, PREFILL_CTX)?;
    let v0 = attn_rows_run(h, tables, case, inp, PREFILL_CTX)?;
    Ok((pre, v0))
}

/// The worst errors of a chunk's outputs over the (row, head) pairs a clause
/// checks, each as a ratio to its band.
#[derive(Default)]
struct Worst {
    /// The prefill against the f64 referee, over the prefill band.
    prefill: f64,
    /// The decode flash against the f64 referee, over the decode band.
    decode: f64,
    /// The prefill against the decode flash, over the two bands.
    between: f64,
}

/// Each (row, head) pair `pick` names: the prefill output, the decode flash's
/// and the f64 attention of the cache the launch left, with the prefill band
/// and the decode band.
fn prefill_bands(
    attn: &AttnGeom,
    tables: &Tables,
    case: &AttnCase,
    inp: &AttnIn,
    (pre, v0): (&AttnSeen, &AttnSeen),
    pick: impl Fn(usize, usize) -> bool,
) -> Result<Worst, String> {
    let q_n = attn.qg_len().ok_or("the query length wraps")?;
    let out_n = attn.out_len().ok_or("the output length wraps")?;
    let end = case.from + case.rows;
    let mut keys = Vec::new();
    let mut vals = Vec::new();
    for kv in 0..attn.kv_heads {
        keys.push(
            (0..end)
                .map(|i| dequant_k_for(attn, &pre.cache, PREFILL_CTX, kv, i))
                .collect::<Vec<_>>(),
        );
        vals.push(
            (0..end)
                .map(|i| dequant_v_for(attn, &pre.cache, PREFILL_CTX, kv, i))
                .collect::<Vec<_>>(),
        );
    }
    let gain = &tables.q_gain;
    let mut worst = Worst::default();
    for r in 0..case.rows {
        let pos = case.from + r;
        let rope = tables
            .rope
            .get(pos * attn.rot..pos * attn.rot + attn.rot)
            .ok_or("the rope row is past the table")?;
        for head in 0..attn.query_heads {
            if !pick(r, head) {
                continue;
            }
            let kv = head / (attn.query_heads / attn.kv_heads);
            let qrow = &inp.q[r * q_n..(r + 1) * q_n];
            let qn = norm_rope(take_qg_for(attn, qrow, head, 0)?, gain, tables.eps, rope)?;
            let gate = if attn.gated {
                Some(take_qg_for(attn, qrow, head, attn.head)?)
            } else {
                None
            };
            let (k, v) = (&keys[kv][..=pos], &vals[kv][..=pos]);
            let refer = attend(&qn, k, v)?;
            let bp = prefill_band(&qn, k, v, gate)?;
            let bd = band(&qn, k, v, gate, case.segments)?;
            for dim in 0..attn.head {
                let at = r * out_n + head * attn.head + dim;
                let (gp, g0) = (pre.out[at], v0.out[at]);
                let want = gate.map_or(refer[dim], |g| refer[dim] * sigmoid(f64::from(g[dim])));
                let here = format!(
                    "row {r} (position {pos}) head {head} dim {dim}: prefill {gp}, decode flash {g0}, f64 attention {want}"
                );
                if !gp.is_finite() || !g0.is_finite() {
                    return Err(format!("{here}: not finite"));
                }
                let (dp, d0) = ((f64::from(gp) - want).abs(), (f64::from(g0) - want).abs());
                let between = (f64::from(gp) - f64::from(g0)).abs();
                if dp > bp[dim] || d0 > bd[dim] || between > bp[dim] + bd[dim] {
                    return Err(format!(
                        "{here}: |prefill - f64| {dp} over band {}, |decode - f64| {d0} over band {}, |prefill - decode| {between}",
                        bp[dim], bd[dim]
                    ));
                }
                worst.prefill = worst.prefill.max(common::ratio(dp, bp[dim]));
                worst.decode = worst.decode.max(common::ratio(d0, bd[dim]));
                worst.between = worst.between.max(common::ratio(between, bp[dim] + bd[dim]));
            }
        }
    }
    Ok(worst)
}

/// The prefill attention of a chunk is within its band of the f64 attention of
/// the cache it read, and so within the two bands of the decode flash's rows:
/// every row and head, with the query heads reading their key head.
pub fn attn_prefill(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let attn = &h.geom().attn;
    let tables = load_tables_for(attn, fixture)?;
    let mut parts = Vec::new();
    let mut worst = Worst::default();
    for (from, rows) in PREFILL_CASES {
        let case = AttnCase {
            from,
            rows,
            segments: PREFILL_V0_SEGMENTS,
        };
        let at = format!("{rows} rows from {from}");
        let inp = attn_inputs(attn, rows)?;
        let (pre, v0) = prefill_pair(h, &tables, &case, &inp)?;
        for (what, seen) in [("prefill", &pre), ("decode flash", &v0)] {
            if let Some(f) = seen.fault {
                return Err(format!("{at}: the {what} raised {f}"));
            }
        }
        same_cache(&pre.cache, &v0.cache).map_err(|e| format!("{at}: {e}"))?;
        let w = prefill_bands(attn, &tables, &case, &inp, (&pre, &v0), |_, _| true)
            .map_err(|e| format!("{at}: {e}"))?;
        worst.prefill = worst.prefill.max(w.prefill);
        worst.decode = worst.decode.max(w.decode);
        worst.between = worst.between.max(w.between);
        parts.push(format!("{rows}@{from}"));
    }
    Ok(Verdict::Noted(format!(
        "rows@first position {} are within the prefill band of the f64 attention (worst ratio {:.3}) and within the two bands of the decode flash (worst ratio {:.3}; the decode flash itself {:.3} of its band)",
        parts.join(", "),
        worst.prefill,
        worst.between,
        worst.decode
    )))
}

/// The fault word names the layer's `site` and no other.
fn only_site(fault: Option<Fault>, site: FaultSite) -> bool {
    matches!(fault, Some(f) if f.layer == LAYER && f.code == site as u32 && f.sites == 1u32 << (site as u32))
}

/// The prefill rows of a chunk with a key row that is not finite, a gate that
/// is not finite and a query past f16's range: the decode flash's NaN and
/// fault word, never a plausible value.
pub fn attn_prefill_nan(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let attn = &h.geom().attn;
    let tables = load_tables_for(attn, fixture)?;
    let q_n = attn.qg_len().ok_or("the query length wraps")?;
    let out_n = attn.out_len().ok_or("the output length wraps")?;
    let kv_n = attn.kv_heads * attn.head;
    let case = AttnCase {
        from: 37,
        rows: 40,
        segments: PREFILL_V0_SEGMENTS,
    };

    // A key row that is not finite, in key head 1 at the chunk's row 20
    // (position 57, in the tile of keys 48 to 63): the rows of that key head
    // from there on are NaN; the rows whose own position is under 48 and
    // those of key head 0 are in band.
    let mut inp = attn_inputs(attn, case.rows)?;
    inp.k[20 * kv_n + attn.head + 5] = f32::NAN;
    let (pre, v0) = prefill_pair(h, &tables, &case, &inp)?;
    if pre.fault != v0.fault {
        return Err(format!(
            "a non-finite key: the prefill's fault word is {:?}, the decode flash's {:?}",
            pre.fault, v0.fault
        ));
    }
    // Key head 1 is read by the query heads `gqa..2 * gqa`.
    let gqa = attn.query_heads / attn.kv_heads;
    for r in 20..case.rows {
        for head in gqa..2 * gqa {
            let at = r * out_n + head * attn.head;
            if v0.out[at..at + attn.head].iter().any(|v| !v.is_nan()) {
                return Err(format!(
                    "the decode flash left row {r} head {head} finite past a non-finite key"
                ));
            }
            if pre.out[at..at + attn.head].iter().any(|v| !v.is_nan()) {
                return Err(format!(
                    "the prefill left row {r} head {head} finite past a non-finite key"
                ));
            }
        }
    }
    prefill_bands(attn, &tables, &case, &inp, (&pre, &v0), |r, head| {
        !(gqa..2 * gqa).contains(&head) || r < 11
    })
    .map_err(|e| format!("a non-finite key: {e}"))?;

    // A gate that is not finite: the product site, and that value NaN, in
    // both. A query with no gate has none.
    if attn.gated {
        let mut inp = attn_inputs(attn, case.rows)?;
        let (gr, gh, gd) = (3usize, 2usize, 7usize);
        inp.q[gr * q_n + gh * attn.qg_stride + attn.head + gd] = f32::NAN;
        let (pre, v0) = prefill_pair(h, &tables, &case, &inp)?;
        if !only_site(pre.fault, FaultSite::F32Product)
            || !only_site(v0.fault, FaultSite::F32Product)
        {
            return Err(format!(
                "a non-finite gate: the prefill's fault word is {:?}, the decode flash's {:?}, the product site alone is expected",
                pre.fault, v0.fault
            ));
        }
        let at = gr * out_n + gh * attn.head + gd;
        if !pre.out[at].is_nan() || !v0.out[at].is_nan() {
            return Err("a non-finite gate left its value finite".to_string());
        }
        prefill_bands(attn, &tables, &case, &inp, (&pre, &v0), |r, head| {
            (r, head) != (gr, gh)
        })
        .map_err(|e| format!("a non-finite gate: {e}"))?;
    }

    // A query that is not finite: that row is NaN and raises nothing; the rows
    // beside it in its SIMD group (the position's other heads, the next
    // position's) are in band, as the decode flash's.
    let (qr, qh, qd) = (20usize, 2usize, 5usize);
    let mut inp = attn_inputs(attn, case.rows)?;
    inp.q[qr * q_n + qh * attn.qg_stride + qd] = f32::NAN;
    let (pre, v0) = prefill_pair(h, &tables, &case, &inp)?;
    if pre.fault.is_some() {
        return Err(format!(
            "a non-finite query: the prefill's fault word is {:?}, none is expected",
            pre.fault
        ));
    }
    let at = qr * out_n + qh * attn.head;
    if pre.out[at..at + attn.head].iter().any(|v| !v.is_nan()) {
        return Err("a non-finite query left its row finite".to_string());
    }
    prefill_bands(attn, &tables, &case, &inp, (&pre, &v0), |r, head| {
        (r, head) != (qr, qh)
    })
    .map_err(|e| format!("a non-finite query: {e}"))?;

    // A query past f16's range (a gain of 1e5): the product site, every row NaN.
    let big = Tables {
        eps: tables.eps,
        q_gain: vec![1e5; attn.head],
        k_gain: tables.k_gain.clone(),
        rope: tables.rope.clone(),
        note: tables.note,
    };
    let inp = attn_inputs(attn, case.rows)?;
    let pre = attn_prefill_run(h, &big, &case, &inp, PREFILL_CTX)?;
    if !only_site(pre.fault, FaultSite::F32Product) {
        return Err(format!(
            "a query past f16's range: the fault word is {:?}, the product site alone is expected",
            pre.fault
        ));
    }
    if pre.out.iter().any(|v| !v.is_nan()) {
        return Err("a query past f16's range left a finite value".to_string());
    }
    Ok(Verdict::Noted(
        "a key row that is not finite makes NaN the rows of its key head from its position on and leaves the rows before its tile in band; a gate that is not finite (of a query that has one) raises the product site and is NaN in both; a query that is not finite makes NaN its own row and no other; a query past f16's range raises the product site and every row is NaN"
            .to_string(),
    ))
}

/// A launch of the norm over `rows` rows, the words around each output row
/// checked as they were, and the fault word as the launch left it.
fn norm_rows_run(
    h: &Harness<'_>,
    x: &[f32],
    gain: &[f32],
    rows: usize,
    (x_stride, y_stride): (usize, usize),
    eps: f32,
) -> Result<(Vec<f32>, Option<Fault>), String> {
    let (device, stream) = (h.device, h.stream);
    let k = h.geom().hidden;
    let sentinel = f32::from_bits(super::SENTINEL_BITS);
    let x_b = device.upload(stream, x).map_err(why)?;
    let gain_b = device.upload(stream, gain).map_err(why)?;
    let mut y_b = device
        .upload(stream, &vec![sentinel; rows * y_stride])
        .map_err(why)?;
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.norm_rows(
            encoder,
            NormRowsParams {
                rows,
                cols: k,
                x: x_b.window(Elems(0), Elems(x.len()))?,
                x_stride,
                gain: gain_b.window(Elems(0), Elems(gain.len()))?,
                y: y_b.window_mut(Elems(0), Elems(rows * y_stride))?,
                y_stride,
                eps,
                fault: device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    let mut y = vec![0f32; rows * y_stride];
    device
        .read(stream, y_b.whole().map_err(why)?, &mut y)
        .map_err(why)?;
    for r in 0..rows {
        let pad = &y[r * y_stride + k..(r + 1) * y_stride];
        if pad.iter().any(|v| v.to_bits() != super::SENTINEL_BITS) {
            return Err(format!(
                "row {r}: a word past the row's {k} columns was written"
            ));
        }
    }
    Ok((y, h.fault()?))
}

/// Row `x` as the decode router carries it. The router run with one nonzero
/// weight in each routed row returns, as that row's logit, the one product
/// `(x · gain)[c] · scale` the router's dot adds to zeros, so the logits of
/// the launches over each block of columns are the router's normed input.
fn router_normed(h: &Harness<'_>, x: &[f32], gain: &[f32], eps: f32) -> Result<Vec<f32>, String> {
    let (k, experts) = (h.geom().hidden, h.geom().router.experts);
    let shared = vec![0f32; k];
    let mut out = vec![0f32; k];
    for c0 in (0..k).step_by(experts) {
        let mut routed = vec![0f32; experts * k];
        for e in 0..experts {
            routed[e * k + c0 + e] = 1.0;
        }
        let run = clauses_router::run(h, &routed, &shared, x, gain, eps)?;
        out[c0..c0 + experts].copy_from_slice(&run.logits[..experts]);
    }
    Ok(out)
}

/// Each row is the bits of the decode router's normed input of that row, for
/// dense and for strided rows.
pub fn norm_rows(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let k = h.geom().hidden;
    let eps = fixture.rms_eps()?;
    let gain = clauses_router::gain_for(&h.geom().router, fixture)?;
    let mut parts = Vec::new();
    for (rows, strides) in [(1usize, (k, k)), (9, (k, k)), (33, (k + 4, k + 8))] {
        let data: Vec<Vec<f32>> = (0..rows)
            .map(|r| oracle::activations(k, 0x40A0 + r as u64, 3.7))
            .collect();
        let x = common::strided(&data, strides.0, f32::NAN);
        let (y, fault) = norm_rows_run(h, &x, &gain, rows, strides, eps)?;
        if let Some(f) = fault {
            return Err(format!("{rows} rows raised {f}"));
        }
        for (r, row) in data.iter().enumerate() {
            let want = router_normed(h, row, &gain, eps)?;
            let got = &y[r * strides.1..r * strides.1 + k];
            same_bits("norm", got, &want).map_err(|e| {
                format!("{rows} rows, row {r}: {e}").replace("stepped launch", "router")
            })?;
        }
        parts.push(rows.to_string());
    }
    Ok(Verdict::Noted(format!(
        "{} rows are the decode router's normed input, bit for bit",
        parts.join(", ")
    )))
}

/// A row the norm cannot take is NaN alone and raises the norm site; a gain
/// that is not finite raises it and moves only its column.
pub fn norm_rows_nan(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let k = h.geom().hidden;
    let eps = 0.0f32;
    let gain = clauses_router::gain_for(&h.geom().router, fixture)?;
    let rows = 9usize;
    let clean: Vec<Vec<f32>> = (0..rows)
        .map(|r| oracle::activations(k, 0x40B0 + r as u64, 3.7))
        .collect();
    let (want, fault) = norm_rows_run(
        h,
        &common::strided(&clean, k, f32::NAN),
        &gain,
        rows,
        (k, k),
        eps,
    )?;
    if let Some(f) = fault {
        return Err(format!("the clean chunk raised {f}"));
    }
    let row_is = |y: &[f32], r: usize| y[r * k..(r + 1) * k].to_vec();
    let nan_alone = |y: &[f32], bad: usize, what: &str| -> Result<(), String> {
        for r in 0..rows {
            if r == bad {
                if !y[r * k..(r + 1) * k].iter().all(|v| v.is_nan()) {
                    return Err(format!("{what}: row {r} is not all NaN"));
                }
            } else {
                same_bits("norm", &row_is(y, r), &row_is(&want, r))
                    .map_err(|e| format!("{what}: row {r} moved: {e}"))?;
            }
        }
        Ok(())
    };
    let mut zero = clean.clone();
    zero[3] = vec![0.0; k];
    let (y, fault) = norm_rows_run(
        h,
        &common::strided(&zero, k, f32::NAN),
        &gain,
        rows,
        (k, k),
        eps,
    )?;
    nan_alone(&y, 3, "a zero row with eps 0")?;
    expect_norm_fault(fault, "a zero row with eps 0")?;

    let mut nan = clean.clone();
    nan[5][700] = f32::NAN;
    let (y, fault) = norm_rows_run(
        h,
        &common::strided(&nan, k, f32::NAN),
        &gain,
        rows,
        (k, k),
        eps,
    )?;
    nan_alone(&y, 5, "a NaN input")?;
    expect_norm_fault(fault, "a NaN input")?;

    let mut bad_gain = gain.clone();
    bad_gain[11] = f32::INFINITY;
    let (y, fault) = norm_rows_run(
        h,
        &common::strided(&clean, k, f32::NAN),
        &bad_gain,
        rows,
        (k, k),
        eps,
    )?;
    expect_norm_fault(fault, "an infinite gain")?;
    for r in 0..rows {
        for c in 0..k {
            let got = y[r * k + c];
            if c == 11 {
                if got.is_finite() {
                    return Err(format!("an infinite gain left row {r} column 11 finite"));
                }
            } else if got.to_bits() != want[r * k + c].to_bits() {
                return Err(format!("an infinite gain moved row {r} column {c}"));
            }
        }
    }

    let (y, fault) = norm_rows_run(
        h,
        &common::strided(&clean, k, f32::NAN),
        &gain,
        rows,
        (k, k),
        f32::INFINITY,
    )?;
    expect_norm_fault(fault, "an infinite epsilon")?;
    if !y.iter().all(|v| v.is_nan()) {
        return Err("an infinite epsilon left a row that is not NaN".to_string());
    }
    Ok(Verdict::Noted(
        "a zero row with eps 0 and a NaN row are NaN alone, an infinite gain moves one column, an infinite eps is every row NaN, each raising the norm site"
            .to_string(),
    ))
}

fn expect_norm_fault(fault: Option<Fault>, what: &str) -> Result<(), String> {
    match fault {
        Some(f) if f.layer == LAYER && f.sites == 1 << FaultSite::NormQuant as u32 => Ok(()),
        Some(f) => Err(format!(
            "{what} raised {f}, not norm_quant in layer {LAYER}"
        )),
        None => Err(format!("{what} raised nothing")),
    }
}

/// The buffers of the delta rows refusal table, sized for [`REF_ROWS`] rows.
struct DeltaRef {
    ch: usize,
    x: MetalBuffer<f32>,
    conv_w: MetalBuffer<f32>,
    b: MetalBuffer<f32>,
    a: MetalBuffer<f32>,
    dt: MetalBuffer<f32>,
    ssm_a: MetalBuffer<f32>,
    z: MetalBuffer<f32>,
    norm_w: MetalBuffer<f32>,
    lane: MetalBuffer<u32>,
    state: MetalBuffer<f32>,
    ring: MetalBuffer<f32>,
    partials: MetalBuffer<f32>,
    g: MetalBuffer<f32>,
    prep: MetalBuffer<f32>,
}

/// Rows of the well-formed refusal-table launches.
const REF_ROWS: usize = 4;

/// The launch of one delta case: the well-formed words, with the ones a case
/// edits.
#[derive(Clone, Copy)]
struct DeltaOver {
    n_k: usize,
    rows: usize,
    pos: u32,
    x_stride: usize,
    z_stride: usize,
    x_len: usize,
    g_len: usize,
    partials_len: usize,
    lane_len: usize,
    lane_at: usize,
    lane_to: usize,
    prep_len: usize,
    prep_at: usize,
}

impl DeltaOver {
    fn whole(delta: &DeltaGeom, ch: usize, prep: usize) -> Self {
        let (g_row, p_row) = (g_row(delta), p_row(delta));
        Self {
            n_k: N_K,
            rows: REF_ROWS,
            pos: 3,
            x_stride: ch,
            z_stride: g_row,
            x_len: REF_ROWS * ch,
            g_len: REF_ROWS * g_row,
            partials_len: REF_ROWS * p_row,
            lane_len: 2,
            lane_at: 0,
            lane_to: 1,
            prep_len: REF_ROWS * prep,
            prep_at: 0,
        }
    }
}

fn delta_ref_launch(
    h: &Harness<'_>,
    b: &mut DeltaRef,
    o: &DeltaOver,
    encoder: &mut bloomery_metal::encode::ComputeEncoder<'_>,
) -> Result<(), bloomery_device::DeviceError> {
    let ch = b.ch;
    let delta = &h.geom().delta;
    let (g_row, p_row) = (g_row(delta), p_row(delta));
    h.kernels.delta_rows(
        encoder,
        DeltaRowsParams {
            x: b.x.window(Elems(0), Elems(o.x_len))?,
            conv_w: b.conv_w.window(Elems(0), Elems(delta.conv_taps * ch))?,
            b_raw: b.b.window(Elems(0), Elems(REF_ROWS * delta.value_heads))?,
            a_raw: b.a.window(Elems(0), Elems(REF_ROWS * delta.value_heads))?,
            dt_bias: b.dt.window(Elems(0), Elems(delta.value_heads))?,
            ssm_a: b.ssm_a.window(Elems(0), Elems(delta.value_heads))?,
            z: b.z.window(Elems(0), Elems(REF_ROWS * g_row))?,
            norm_w: b.norm_w.window(Elems(0), Elems(delta.head_cols))?,
            lane: b.lane.window(Elems(0), Elems(o.lane_len))?,
            state: b
                .state
                .window_mut(Elems(0), Elems(LANES * g_row * delta.head_cols))?,
            ring: b.ring.window_mut(Elems(0), Elems(delta.ring_rows * ch))?,
            partials: b.partials.window_mut(Elems(0), Elems(o.partials_len))?,
            g: b.g.window_mut(Elems(0), Elems(o.g_len))?,
            prep: b.prep.window_mut(Elems(o.prep_at), Elems(o.prep_len))?,
            faults: h.device.fault_word(),
            n_v: delta.value_heads,
            head: delta.head_cols,
            taps: delta.conv_taps,
            n_k: o.n_k,
            map: HeadMap::Tiled,
            lanes: LANES,
            lane_at: o.lane_at,
            lane_to: o.lane_to,
            layer: LAYER,
            rows: o.rows,
            pos: o.pos,
            strides: DeltaRowStrides {
                x: o.x_stride,
                z: o.z_stride,
                b: delta.value_heads,
                a: delta.value_heads,
                g: g_row,
                partials: p_row,
            },
            eps: 1e-6,
            snaps: None,
        },
    )
}

fn delta_case<'a>(
    h: &'a Harness<'_>,
    whole: DeltaOver,
    what: &'static str,
    needle: &'static str,
    edit: fn(&mut DeltaOver, usize),
) -> BadArg<'a, DeltaRef> {
    BadArg::new(
        what,
        needle,
        Box::new(move |b, encoder| {
            let mut o = whole;
            edit(&mut o, b.ch);
            delta_ref_launch(h, b, &o, encoder)
        }),
    )
}

fn delta_refusals(h: &Harness<'_>) -> Result<usize, String> {
    let (device, stream) = (h.device, h.stream);
    let delta = &h.geom().delta;
    let (g_row, p_row) = (g_row(delta), p_row(delta));
    let ch = elems(delta.channels(N_K), "channels")?;
    let ones = |n: usize| device.upload(stream, &vec![1f32; n]).map_err(why);
    let mut bufs = DeltaRef {
        ch,
        // Not the 1.0 the ring holds, so a committed ring row is seen to move.
        x: device
            .upload(stream, &vec![2f32; REF_ROWS * ch])
            .map_err(why)?,
        conv_w: ones(delta.conv_taps * ch)?,
        b: ones(REF_ROWS * delta.value_heads)?,
        a: ones(REF_ROWS * delta.value_heads)?,
        dt: ones(delta.value_heads)?,
        ssm_a: ones(delta.value_heads)?,
        z: ones(REF_ROWS * g_row)?,
        norm_w: ones(delta.head_cols)?,
        lane: device.upload(stream, &[0u32, 1]).map_err(why)?,
        state: ones(LANES * g_row * delta.head_cols)?,
        ring: ones(delta.ring_rows * ch)?,
        partials: ones(REF_ROWS * p_row)?,
        g: ones(REF_ROWS * g_row)?,
        // One word more than the rows need, so a window can start a word in.
        prep: ones(REF_ROWS * prep_row(delta)? + 1)?,
    };
    let whole = DeltaOver::whole(delta, ch, prep_row(delta)?);
    let case = |what, needle, edit| delta_case(h, whole, what, needle, edit);
    let cases = vec![
        case("rows of 0", "rows of 0", |o, _| o.rows = 0),
        case("an n_k of 3", "n_k", |o, _| o.n_k = 3),
        case("positions past the 32-bit word", "position word", |o, _| {
            o.pos = u32::MAX - 1;
        }),
        case("an x stride shorter than a row", "x stride", |o, ch| {
            o.x_stride = ch - 1;
        }),
        case("a z stride shorter than a row", "z stride", |o, _| {
            o.z_stride -= 1;
        }),
        case("an x window short of the last row", "x window", |o, _| {
            o.x_len -= 1;
        }),
        case(
            "a numerator window short of the last row",
            "g window",
            |o, _| {
                o.g_len -= 1;
            },
        ),
        case(
            "a partials window short of the last row",
            "partials window",
            |o, _| {
                o.partials_len -= 1;
            },
        ),
        case(
            "a prepared rows window short of the last row",
            "prepared rows window",
            |o, _| {
                o.prep_len -= 1;
            },
        ),
        case(
            "a prepared rows window off the 16-byte word",
            "16-byte word",
            |o, _| {
                o.prep_at = 1;
            },
        ),
        case(
            "a read lane index past the lane window",
            "read lane index",
            |o, _| {
                o.lane_at = 2;
            },
        ),
        case(
            "a write lane index past the lane window",
            "write lane index",
            |o, _| {
                o.lane_to = 2;
            },
        ),
    ];
    let unchanged = |h: &Harness<'_>, b: &DeltaRef| {
        common::still(h, &b.g, |v| *v == 1.0)?;
        common::still(h, &b.state, |v| *v == 1.0)?;
        common::still(h, &b.partials, |v| *v == 1.0)?;
        common::still(h, &b.ring, |v| *v == 1.0)
    };
    let count = common::refuse_launches(h, "delta step", &mut bufs, cases, unchanged)?;
    let good: Edit<'_, DeltaRef> = Box::new(|b, encoder| delta_ref_launch(h, b, &whole, encoder));
    common::accepted_writes(h, &mut bufs, good, |h, b| {
        Ok(common::moved(h, &b.g, |v| *v != 1.0)? && common::moved(h, &b.ring, |v| *v != 1.0)?)
    })?;
    Ok(count)
}

/// The buffers of the attention rows refusal table.
struct AttnRef {
    q: MetalBuffer<f32>,
    k: MetalBuffer<f32>,
    v: MetalBuffer<f32>,
    q_gain: MetalBuffer<f32>,
    k_gain: MetalBuffer<f32>,
    rope: MetalBuffer<f32>,
    partials: MetalBuffer<f32>,
    out: MetalBuffer<f32>,
    ticket: MetalBuffer<u32>,
    cache: Uploaded,
    q8_codes: [MetalBuffer<u32>; 2],
    q8_scales: [MetalBuffer<u16>; 2],
}

const ATTN_SEGMENTS: usize = 2;

#[derive(Clone, Copy)]
struct AttnOver {
    kv_heads: usize,
    segments: usize,
    rows: usize,
    from: u32,
    q_stride: usize,
    q_len: usize,
    out_len: usize,
    partials_len: usize,
    ticket_len: usize,
    q8: bool,
    prefill: bool,
}

impl AttnOver {
    fn whole(attn: &AttnGeom) -> Result<Self, String> {
        let q_n = attn.qg_len().ok_or("the query length wraps")?;
        let out_n = attn.out_len().ok_or("the output length wraps")?;
        let partial_n = attn
            .partials_len(ATTN_SEGMENTS)
            .ok_or("the partial length wraps")?;
        Ok(Self {
            kv_heads: attn.kv_heads,
            segments: ATTN_SEGMENTS,
            rows: REF_ROWS,
            from: 10,
            q_stride: q_n,
            q_len: REF_ROWS * q_n,
            out_len: REF_ROWS * out_n,
            partials_len: REF_ROWS * partial_n,
            ticket_len: REF_ROWS * attn.kv_heads,
            q8: false,
            prefill: false,
        })
    }
}

fn attn_ref_launch(
    h: &Harness<'_>,
    b: &mut AttnRef,
    o: &AttnOver,
    encoder: &mut bloomery_metal::encode::ComputeEncoder<'_>,
) -> Result<(), bloomery_device::DeviceError> {
    let attn = &h.geom().attn;
    let kv_n = attn.kv_heads * attn.head;
    let out_n = attn.out_len().expect("the output length fits");
    let cache = if o.q8 {
        let codes = attn.q8_code_len(CTX).expect("the code length fits");
        let scales = attn.q8_scale_len(CTX).expect("the scale length fits");
        let [k_codes, v_codes] = &mut b.q8_codes;
        let [k_scales, v_scales] = &mut b.q8_scales;
        AttnCache::Q8 {
            k_codes: k_codes.window_mut(Elems(0), Elems(codes))?,
            k_scales: k_scales.window_mut(Elems(0), Elems(scales))?,
            v_codes: v_codes.window_mut(Elems(0), Elems(codes))?,
            v_scales: v_scales.window_mut(Elems(0), Elems(scales))?,
        }
    } else {
        cache_of_for(attn, &mut b.cache, CTX)?
    };
    if o.prefill {
        return h.kernels.attention_prefill(
            encoder,
            AttentionPrefillParams {
                head_dim: attn.head,
                query_heads: attn.query_heads,
                kv_heads: o.kv_heads,
                ctx: CTX,
                eps: 1e-6,
                layer: LAYER,
                from: o.from,
                rows: o.rows,
                q: b.q.window(Elems(0), Elems(o.q_len))?,
                q_stride: o.q_stride,
                k: b.k.window(Elems(0), Elems(REF_ROWS * kv_n))?,
                k_stride: kv_n,
                v: b.v.window(Elems(0), Elems(REF_ROWS * kv_n))?,
                v_stride: kv_n,
                q_gain: gain_of(attn, &b.q_gain, 0)?,
                k_gain: gain_of(attn, &b.k_gain, 0)?,
                rope: b.rope.window(Elems(0), Elems(CTX * attn.rot))?,
                cache,
                out: b.out.window_mut(Elems(0), Elems(o.out_len))?,
                out_stride: out_n,
                fault: h.device.fault_word(),
            },
        );
    }
    h.kernels.attention_rows(
        encoder,
        AttentionRowsParams {
            head_dim: attn.head,
            query_heads: attn.query_heads,
            kv_heads: o.kv_heads,
            segments: o.segments,
            ctx: CTX,
            eps: 1e-6,
            layer: LAYER,
            from: o.from,
            rows: o.rows,
            q: b.q.window(Elems(0), Elems(o.q_len))?,
            q_stride: o.q_stride,
            k: b.k.window(Elems(0), Elems(REF_ROWS * kv_n))?,
            k_stride: kv_n,
            v: b.v.window(Elems(0), Elems(REF_ROWS * kv_n))?,
            v_stride: kv_n,
            q_gain: gain_of(attn, &b.q_gain, 0)?,
            k_gain: gain_of(attn, &b.k_gain, 0)?,
            rope: b.rope.window(Elems(0), Elems(CTX * attn.rot))?,
            cache,
            partials: b.partials.window_mut(Elems(0), Elems(o.partials_len))?,
            out: b.out.window_mut(Elems(0), Elems(o.out_len))?,
            out_stride: out_n,
            fault: h.device.fault_word(),
            ticket: b.ticket.window_mut(Elems(0), Elems(o.ticket_len))?,
        },
    )
}

fn attn_case<'a>(
    h: &'a Harness<'_>,
    whole: AttnOver,
    what: &'static str,
    needle: &'static str,
    edit: fn(&mut AttnOver),
) -> BadArg<'a, AttnRef> {
    BadArg::new(
        what,
        needle,
        Box::new(move |b, encoder| {
            let mut o = whole;
            edit(&mut o);
            attn_ref_launch(h, b, &o, encoder)
        }),
    )
}

fn attn_refusals(h: &Harness<'_>, tables: &Tables) -> Result<usize, String> {
    let (device, stream) = (h.device, h.stream);
    let attn = &h.geom().attn;
    let q_n = attn.qg_len().ok_or("the query length wraps")?;
    let out_n = attn.out_len().ok_or("the output length wraps")?;
    let partial_n = attn
        .partials_len(ATTN_SEGMENTS)
        .ok_or("the partial length wraps")?;
    let kv_n = attn.kv_heads * attn.head;
    let scene = Scene {
        segments: ATTN_SEGMENTS,
        ctx: CTX,
        pos: 10,
        n_keys: 10,
        prefill: 10,
        q8: false,
    };
    let inp = attn_inputs(attn, REF_ROWS)?;
    let q8_codes = vec![0u32; attn.q8_code_len(CTX).ok_or("the code length wraps")?];
    let q8_scales = vec![0u16; attn.q8_scale_len(CTX).ok_or("the scale length wraps")?];
    let mut bufs = AttnRef {
        q: device.upload(stream, &inp.q).map_err(why)?,
        k: device.upload(stream, &inp.k).map_err(why)?,
        v: device.upload(stream, &inp.v).map_err(why)?,
        q_gain: device.upload(stream, &tables.q_gain).map_err(why)?,
        k_gain: device.upload(stream, &tables.k_gain).map_err(why)?,
        rope: device
            .upload(stream, &tables.rope[..CTX * attn.rot])
            .map_err(why)?,
        partials: device
            .upload(stream, &vec![1f32; REF_ROWS * partial_n])
            .map_err(why)?,
        out: device
            .upload(stream, &vec![1f32; REF_ROWS * out_n])
            .map_err(why)?,
        ticket: device
            .upload(stream, &vec![0u32; REF_ROWS * attn.kv_heads])
            .map_err(why)?,
        cache: upload_cache(device, stream, &plant_f16_for(attn, &scene)?)?,
        q8_codes: [0, 1]
            .map(|_| device.upload(stream, &q8_codes))
            .map(|b| b.expect("upload")),
        q8_scales: [0, 1]
            .map(|_| device.upload(stream, &q8_scales))
            .map(|b| b.expect("upload")),
    };
    let _ = (q_n, kv_n);
    let whole = AttnOver::whole(attn)?;
    let case = |what, needle, edit| attn_case(h, whole, what, needle, edit);
    let cases = vec![
        case("rows of 0", "rows of 0", |o| o.rows = 0),
        case("kv heads of 1", "kv heads", |o| o.kv_heads = 1),
        case("segments of 0", "segments of 0", |o| o.segments = 0),
        case("segments past the kernel's cap", "past the kernel", |o| {
            o.segments = 65;
        }),
        case("positions past the cache", "pass the cache", |o| {
            o.from = 77
        }),
        case("positions past the 32-bit word", "pass the cache", |o| {
            o.from = u32::MAX;
            o.rows = 2;
        }),
        case("a q stride shorter than a row", "q stride", |o| {
            o.q_stride -= 1;
        }),
        case("a q window short of the last row", "q window", |o| {
            o.q_len -= 1
        }),
        case("an out window short of the last row", "out window", |o| {
            o.out_len -= 1;
        }),
        case(
            "a partials window short of the last row",
            "partials window",
            |o| {
                o.partials_len -= 1;
            },
        ),
        case(
            "a ticket window short of one word a key head a row",
            "ticket window",
            |o| {
                o.ticket_len -= 1;
            },
        ),
        case("the q8_0 planes", "q8_0", |o| o.q8 = true),
    ];
    let unchanged = |h: &Harness<'_>, b: &AttnRef| {
        common::still(h, &b.out, |v| *v == 1.0)?;
        common::still(h, &b.partials, |v| *v == 1.0)?;
        common::still(h, &b.ticket, |v| *v == 0)
    };
    let count = common::refuse_launches(h, "attention rows", &mut bufs, cases, unchanged)?;
    let good: Edit<'_, AttnRef> = Box::new(|b, encoder| attn_ref_launch(h, b, &whole, encoder));
    common::accepted_writes(h, &mut bufs, good, |h, b| {
        common::moved(h, &b.out, |v| *v != 1.0)
    })?;
    // The prefill launch: the shape and window refusals it shares with the
    // rows launch, on the same buffers with the output as it was.
    bufs.out = device
        .upload(stream, &vec![1f32; REF_ROWS * out_n])
        .map_err(why)?;
    let whole = AttnOver {
        prefill: true,
        ..whole
    };
    let case = |what, needle, edit| attn_case(h, whole, what, needle, edit);
    let cases = vec![
        case("rows of 0", "rows of 0", |o| o.rows = 0),
        case("kv heads of 1", "kv heads", |o| o.kv_heads = 1),
        case("positions past the cache", "pass the cache", |o| {
            o.from = 77
        }),
        case("a q stride shorter than a row", "q stride", |o| {
            o.q_stride -= 1;
        }),
        case("a q window short of the last row", "q window", |o| {
            o.q_len -= 1
        }),
        case("an out window short of the last row", "out window", |o| {
            o.out_len -= 1;
        }),
        case("the q8_0 planes", "q8_0", |o| o.q8 = true),
    ];
    let unchanged = |h: &Harness<'_>, b: &AttnRef| common::still(h, &b.out, |v| *v == 1.0);
    let more = common::refuse_launches(h, "attention prefill", &mut bufs, cases, unchanged)?;
    let good: Edit<'_, AttnRef> = Box::new(|b, encoder| attn_ref_launch(h, b, &whole, encoder));
    common::accepted_writes(h, &mut bufs, good, |h, b| {
        common::moved(h, &b.out, |v| *v != 1.0)
    })?;
    Ok(count + more)
}

/// The buffers of the norm rows refusal table.
struct NormRef {
    x: MetalBuffer<f32>,
    gain: MetalBuffer<f32>,
    y: MetalBuffer<f32>,
}

#[derive(Clone, Copy)]
struct NormOver {
    rows: usize,
    cols: usize,
    x_stride: usize,
    y_stride: usize,
    x_at: usize,
    x_len: usize,
    y_at: usize,
    y_len: usize,
    gain_at: usize,
    gain_len: usize,
}

impl NormOver {
    fn whole(k: usize) -> Self {
        Self {
            rows: REF_ROWS,
            cols: k,
            x_stride: k,
            y_stride: k,
            x_at: 4,
            x_len: REF_ROWS * k,
            y_at: 4,
            y_len: REF_ROWS * k,
            gain_at: 4,
            gain_len: k,
        }
    }
}

fn norm_ref_launch(
    h: &Harness<'_>,
    b: &mut NormRef,
    o: &NormOver,
    encoder: &mut bloomery_metal::encode::ComputeEncoder<'_>,
) -> Result<(), bloomery_device::DeviceError> {
    h.kernels.norm_rows(
        encoder,
        NormRowsParams {
            rows: o.rows,
            cols: o.cols,
            x: b.x.window(Elems(o.x_at), Elems(o.x_len))?,
            x_stride: o.x_stride,
            gain: b.gain.window(Elems(o.gain_at), Elems(o.gain_len))?,
            y: b.y.window_mut(Elems(o.y_at), Elems(o.y_len))?,
            y_stride: o.y_stride,
            eps: 1e-6,
            fault: h.device.fault_word(),
            layer: LAYER,
        },
    )
}

fn norm_case<'a>(
    h: &'a Harness<'_>,
    whole: NormOver,
    what: &'static str,
    needle: &'static str,
    edit: fn(&mut NormOver),
) -> BadArg<'a, NormRef> {
    BadArg::new(
        what,
        needle,
        Box::new(move |b, encoder| {
            let mut o = whole;
            edit(&mut o);
            norm_ref_launch(h, b, &o, encoder)
        }),
    )
}

fn norm_refusals(h: &Harness<'_>) -> Result<usize, String> {
    let (device, stream) = (h.device, h.stream);
    let k = h.geom().hidden;
    // Windows start four floats into their buffers: 16-byte aligned.
    let mut bufs = NormRef {
        x: device
            .upload(stream, &vec![1f32; 4 + REF_ROWS * k + 4])
            .map_err(why)?,
        gain: device.upload(stream, &vec![1f32; 4 + k + 4]).map_err(why)?,
        y: device
            .upload(stream, &vec![7f32; 4 + REF_ROWS * k + 4])
            .map_err(why)?,
    };
    let whole = NormOver::whole(k);
    let case = |what, needle, edit| norm_case(h, whole, what, needle, edit);
    let cases = vec![
        case("rows of 0", "rows of 0", |o| o.rows = 0),
        case("a column count of 0", "cols 0", |o| o.cols = 0),
        case("columns the threads do not cover", "multiple of", |o| {
            o.cols += 4;
        }),
        case(
            "an x stride that is not a float4 multiple",
            "x stride",
            |o| {
                o.x_stride += 1;
            },
        ),
        case(
            "a y stride that is not a float4 multiple",
            "y stride",
            |o| {
                o.y_stride += 2;
            },
        ),
        case("an x stride shorter than a row", "x stride", |o| {
            o.x_stride -= 4;
        }),
        case("a y stride shorter than a row", "y stride", |o| {
            o.y_stride -= 4;
        }),
        case("an x window short of the last row", "x window", |o| {
            o.x_len -= 1
        }),
        case("a y window short of the last row", "y window", |o| {
            o.y_len -= 1
        }),
        case("a gain window short of the columns", "gain window", |o| {
            o.gain_len -= 1;
        }),
        case("an x window off the 16-byte word", "16-byte", |o| {
            o.x_at = 1
        }),
        case("a gain window off the 16-byte word", "16-byte", |o| {
            o.gain_at = 1
        }),
        case("a y window off the 16-byte word", "16-byte", |o| o.y_at = 1),
    ];
    let unchanged = |h: &Harness<'_>, b: &NormRef| common::still(h, &b.y, |v| *v == 7.0);
    let count = common::refuse_launches(h, "norm rows", &mut bufs, cases, unchanged)?;
    let good: Edit<'_, NormRef> = Box::new(|b, encoder| norm_ref_launch(h, b, &whole, encoder));
    common::accepted_writes(h, &mut bufs, good, |h, b| {
        common::moved(h, &b.y, |v| *v != 7.0)
    })?;
    Ok(count)
}

/// Each launcher refusal is a named range error and writes nothing: the delta
/// rows, the attention rows and the norm rows.
pub fn rows_refusals(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let tables = load_tables_for(&h.geom().attn, fixture)?;
    // A file with no delta layer launches no delta rows.
    let delta = if h.library.instance().key.delta.is_some() {
        delta_refusals(h).map_err(|e| format!("delta rows: {e}"))?
    } else {
        0
    };
    let attention = attn_refusals(h, &tables).map_err(|e| format!("attention rows: {e}"))?;
    let norm = norm_refusals(h).map_err(|e| format!("norm rows: {e}"))?;
    Ok(Verdict::Noted(format!(
        "{delta} delta, {attention} attention and {norm} norm refusals, each by name, nothing written"
    )))
}

/// The clauses of the row kernels, in gate order: the whole table, with the
/// delta net's.
pub fn table<'a>(ctx: &'a ClauseCtx<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    let mut clauses: Vec<(&'static str, ClauseRun<'a>)> = vec![
        ("delta_rows", Box::new(|| ctx.with_params(delta_rows))),
        (
            "delta_rows_nan",
            Box::new(|| ctx.with_params(delta_rows_nan)),
        ),
        (
            "delta_rows_snap",
            Box::new(|| ctx.with_params(delta_rows_snap)),
        ),
    ];
    clauses.extend(shaped(ctx));
    clauses
}

/// The delta net's clauses of the row kernels on drawn parameters: for an
/// instance whose delta layer is not the fixture's.
pub fn wide<'a>(ctx: &'a ClauseCtx<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    let h = ctx.harness;
    vec![
        ("delta_rows", Box::new(|| delta_rows(h, Params::Drawn))),
        (
            "delta_rows_nan",
            Box::new(|| delta_rows_nan(h, Params::Drawn)),
        ),
        (
            "delta_rows_snap",
            Box::new(|| delta_rows_snap(h, Params::Drawn)),
        ),
    ]
}

/// The clauses of the attention kernels of the row launches: the append, the
/// rows flash and the prefill flash, for an instance whose attention layers
/// are the only axes its row differs in.
pub fn attention<'a>(ctx: &'a ClauseCtx<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    vec![
        ("norm_rope_rows", Box::new(|| ctx.with(norm_rope_rows))),
        ("attn_rows_v0", Box::new(|| ctx.with(attn_rows_v0))),
        ("attn_prefill", Box::new(|| ctx.with(attn_prefill))),
        ("attn_prefill_nan", Box::new(|| ctx.with(attn_prefill_nan))),
    ]
}

/// The clauses of the row kernels an instance whose layers are all attention
/// runs: the delta net's are left out.
pub fn shaped<'a>(ctx: &'a ClauseCtx<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    let mut clauses = attention(ctx);
    let rest: Vec<(&'static str, ClauseRun<'a>)> = vec![
        ("norm_rows", Box::new(|| ctx.with(norm_rows))),
        ("norm_rows_nan", Box::new(|| ctx.with(norm_rows_nan))),
        ("rows_refusals", Box::new(|| ctx.with(rows_refusals))),
    ];
    clauses.extend(rest);
    clauses
}
