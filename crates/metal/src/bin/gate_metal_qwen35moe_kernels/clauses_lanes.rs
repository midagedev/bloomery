//! The lane ring's clauses: a rows launch that leaves each row's end state in
//! the lane of the position after it (`gdn_rows_ring`,
//! `Kernels::delta_rows_ring`), so keeping the first rows of a chunk moves a
//! position and nothing else.
//!
//! The reference is the decode step, stepped: row `j`'s end state is the state
//! of `j + 1` stepped launches from the same start, and a token after a commit
//! reads the lane the commit names and is the stepped launches' bits. The lane
//! bookkeeping under test is the engine's own (`arch::qwen3moe::lanes`): the
//! gate asks it for the words a launch takes and the lane a commit leaves.

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Device, Elems, Fault};
use bloomery_metal::MetalBuffer;
use bloomery_metal::arch::qwen3moe::lanes::Verify;
use bloomery_metal::geom::DeltaGeom;
use bloomery_metal::kernels::{
    DeltaParams, DeltaRowStrides, DeltaRowsParams, DeltaSnaps, HeadMap, PASS_ROWS,
};

use super::{
    Bufs, Chunk, Fixed, Harness, LAYER, N_K, Out, Params, chunk, draw, elems, fixed, g_row,
    head_of, oracle, p_row, prep_row, run_stepped, same_bits,
};

/// A ring case: the lanes of the state, the rows of the verify and its first
/// position. The first position's lane is the lane the launch reads, and the
/// rows end in the lanes after it, wrapping.
type RingCase = (usize, usize, u32);

/// The cases: the arena's two lanes at both read lanes, and at the first
/// position (taps before it); more lanes than rows, so lanes no row writes
/// must stay as they were; as many rows as lanes, whose last row ends in the
/// lane the launch read; the widest verify the conv ring rolls back; and the
/// last positions of the 32-bit word (a token still follows a commit of all rows).
const RING_CASES: [RingCase; 11] = [
    (2, 2, 4),
    (2, 2, 7),
    (2, 2, 0),
    (3, 2, 5),
    (3, 3, 4),
    (4, 4, 6),
    (4, 3, 9),
    (4, 2, 11),
    (4, 1, 3),
    (PASS_ROWS, PASS_ROWS, 5),
    (3, 3, u32::MAX - 4),
];

/// A verify of `rows` rows from `pos0` over `lanes` lanes, with one more row of
/// inputs than the verify has: the token after a commit. The state is `lanes`
/// lanes of draws, the lane words name lane `i` by word `i`, and the words the
/// launch takes are the engine's bookkeeping's.
fn ring_chunk(
    delta: &DeltaGeom,
    rng: &mut oracle::Rng,
    (lanes, rows, pos0): RingCase,
) -> Result<(Verify, Chunk), String> {
    let v = Verify::new(lanes, pos0, rows).map_err(|e| e.to_string())?;
    let mut c = chunk(delta, rng, rows + 1, pos0, (v.lane_at(), v.lane_to()))?;
    c.state = draw(rng, elems(delta.state_elems(lanes), "state")?, 0.04);
    c.lane_words = (0..lanes as u32).collect();
    Ok((v, c))
}

/// The lane of `state`, `lane_len` floats a lane.
fn lane_of_state(state: &[f32], lane_len: usize, lane: usize) -> &[f32] {
    &state[lane * lane_len..(lane + 1) * lane_len]
}

/// The chunk as one lane-ring launch over its first `rows` rows. The buffers
/// are returned with what the launch left, so a token can follow in them.
fn run_ring(h: &Harness<'_>, fx: &Fixed, c: &Chunk) -> Result<(Bufs, Out), String> {
    let delta = &h.geom().delta;
    let (g_row, p_row) = (g_row(delta), p_row(delta));
    let mut bufs = Bufs::new(h, fx, c)?;
    let strides = DeltaRowStrides::of(delta, N_K).ok_or("the strides do not fit")?;
    let prep_len = c.rows * prep_row(delta)?;
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.delta_rows_ring(
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
                lanes: c.lane_words.len(),
                lane_at: c.lane_at,
                lane_to: c.lane_to,
                layer: LAYER,
                rows: c.rows,
                pos: c.pos,
                strides,
                eps: fx.eps,
                snaps: None,
            },
        )
    })?;
    let out = bufs.read(h, c)?;
    Ok((bufs, out))
}

/// `c` with its write lane word `to`.
fn writing(mut c: Chunk, to: usize) -> Chunk {
    c.lane_to = to;
    c
}

/// The numerators, partials, conv ring and fault word of two launches are one
/// bit pattern; the state is held apart, lane by lane.
fn same_outputs(what: &str, got: &Out, want: &Out) -> Result<(), String> {
    same_bits(&format!("{what}: g"), &got.g, &want.g)?;
    same_bits(&format!("{what}: partials"), &got.partials, &want.partials)?;
    same_bits(&format!("{what}: ring"), &got.ring, &want.ring)?;
    if got.fault != want.fault {
        let show = |f: &Option<Fault>| f.map_or("none".to_string(), |f| f.to_string());
        return Err(format!(
            "{what}: the fault word is {}, the stepped launches' {}",
            show(&got.fault),
            show(&want.fault)
        ));
    }
    Ok(())
}

/// A lane-ring launch leaves, in lane `lane_of(pos0 + j + 1)` for every row
/// `j`, the state of `j + 1` stepped launches from the same start, bit for bit;
/// the lanes no row writes stay as they were; and its numerators, partials,
/// conv ring and fault word are the stepped launches'.
pub fn delta_rows_lanes(h: &Harness<'_>, params: Params<'_>) -> Clause {
    let delta = &h.geom().delta;
    let fx = fixed(delta, params)?;
    let lane_len = elems(delta.state_elems(1), "a lane")?;
    let mut rng = oracle::Rng::new(0xD0B1);
    let mut parts = Vec::new();
    for case in RING_CASES {
        let (lanes, rows, pos0) = case;
        let at = format!("{rows} rows from {pos0} over {lanes} lanes");
        let (v, full) = ring_chunk(delta, &mut rng, case)?;
        let c = head_of(delta, &full, rows)?;
        let (_, got) = run_ring(h, &fx, &c)?;
        if let Some(f) = got.fault {
            return Err(format!("{at}: raised {f}"));
        }
        let steps = run_stepped(h, &fx, &c)?;
        same_outputs(&at, &got, &steps)?;
        // The state the launch must leave: each row's end state in its lane,
        // every other lane as uploaded.
        let mut want = c.state.clone();
        for j in 0..rows {
            let lane = v
                .lane_after(j)
                .ok_or_else(|| format!("{at}: no lane after row {j}"))?;
            let end = run_stepped(h, &fx, &writing(head_of(delta, &c, j + 1)?, lane))?;
            let ended = lane_of_state(&end.state, lane_len, lane);
            if ended == lane_of_state(&c.state, lane_len, lane) {
                return Err(format!("{at}: row {j} left its lane as it was"));
            }
            want[lane * lane_len..(lane + 1) * lane_len].copy_from_slice(ended);
        }
        same_bits(&format!("{at}: state"), &got.state, &want)?;
        parts.push(format!("{rows}@{pos0}/{lanes}"));
    }
    Ok(Verdict::Noted(format!(
        "rows@first position/lanes {}: each row's end state is the stepped launches' bits in the lane \
         of the position after it, the other lanes as they were",
        parts.join(", ")
    )))
}

/// What the token after a commit left: its numerators and partials, the whole
/// state and the fault word.
struct After {
    g: Vec<f32>,
    partials: Vec<f32>,
    state: Vec<f32>,
    fault: Option<Fault>,
}

/// The token after a commit, in the buffers the verify left: a decode step at
/// position `pos`, reading the lane `lane` names and writing lane `to`, over
/// the inputs of row `rows` of `full`, the row after the verify's.
fn run_after_commit(
    h: &Harness<'_>,
    fx: &Fixed,
    full: &Chunk,
    bufs: &mut Bufs,
    rows: usize,
    (pos, lane, to): (u32, usize, usize),
) -> Result<After, String> {
    let delta = &h.geom().delta;
    let (device, stream) = (h.device, h.stream);
    let (g_row, p_row) = (g_row(delta), p_row(delta));
    let ch = elems(delta.channels(N_K), "channels")?;
    let heads = delta.value_heads;
    let lanes = full.lane_words.len();
    let x = device
        .upload(stream, &full.x[rows * ch..(rows + 1) * ch])
        .map_err(why)?;
    let b = device
        .upload(stream, &full.b[rows * heads..(rows + 1) * heads])
        .map_err(why)?;
    let a = device
        .upload(stream, &full.a[rows * heads..(rows + 1) * heads])
        .map_err(why)?;
    let z = device
        .upload(stream, &full.z[rows * g_row..(rows + 1) * g_row])
        .map_err(why)?;
    let mut partials = device.upload(stream, &vec![0f32; p_row]).map_err(why)?;
    let mut g = device.upload(stream, &vec![0f32; g_row]).map_err(why)?;
    let words = [pos];
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.delta(
            encoder,
            DeltaParams {
                x: x.window(Elems(0), Elems(ch))?,
                conv_w: bufs.conv_w.window(Elems(0), Elems(fx.conv_w.len()))?,
                b_raw: b.window(Elems(0), Elems(heads))?,
                a_raw: a.window(Elems(0), Elems(heads))?,
                dt_bias: bufs.dt.window(Elems(0), Elems(fx.dt.len()))?,
                ssm_a: bufs.ssm_a.window(Elems(0), Elems(fx.ssm_a.len()))?,
                z: z.window(Elems(0), Elems(g_row))?,
                norm_w: bufs.norm_w.window(Elems(0), Elems(fx.norm_w.len()))?,
                lane: bufs.lane.window(Elems(0), Elems(lanes))?,
                state: bufs.state.window_mut(Elems(0), Elems(full.state.len()))?,
                ring: bufs.ring.window_mut(Elems(0), Elems(full.ring.len()))?,
                partials: partials.window_mut(Elems(0), Elems(p_row))?,
                g: g.window_mut(Elems(0), Elems(g_row))?,
                faults: h.device.fault_word(),
                n_v: delta.value_heads,
                head: delta.head_cols,
                taps: delta.conv_taps,
                n_k: N_K,
                map: HeadMap::Tiled,
                lanes,
                lane_at: lane,
                lane_to: to,
                layer: LAYER,
                pos: &words,
                eps: fx.eps,
            },
        )
    })?;
    let fault = h.fault()?;
    let read = |buf: &MetalBuffer<f32>, n: usize| -> Result<Vec<f32>, String> {
        let mut out = vec![0f32; n];
        device
            .read(stream, buf.whole().map_err(why)?, &mut out)
            .map_err(why)?;
        Ok(out)
    };
    Ok(After {
        g: read(&g, g_row)?,
        partials: read(&partials, p_row)?,
        state: read(&bufs.state, full.state.len())?,
        fault,
    })
}

/// `c`'s first `kept` rows and then its row `kept + extra`, the token a commit
/// of `kept` rows is followed by, as one chunk.
fn kept_then(delta: &DeltaGeom, c: &Chunk, kept: usize, extra: usize) -> Result<Chunk, String> {
    let ch = elems(delta.channels(N_K), "channels")?;
    let (g_row, heads) = (g_row(delta), delta.value_heads);
    let mut x = c.x[..kept * ch].to_vec();
    let mut b = c.b[..kept * heads].to_vec();
    let mut a = c.a[..kept * heads].to_vec();
    let mut z = c.z[..kept * g_row].to_vec();
    x.extend_from_slice(&c.x[extra * ch..(extra + 1) * ch]);
    b.extend_from_slice(&c.b[extra * heads..(extra + 1) * heads]);
    a.extend_from_slice(&c.a[extra * heads..(extra + 1) * heads]);
    z.extend_from_slice(&c.z[extra * g_row..(extra + 1) * g_row]);
    Ok(Chunk {
        rows: kept + 1,
        pos: c.pos,
        lane_words: c.lane_words.clone(),
        lane_at: c.lane_at,
        lane_to: c.lane_to,
        x,
        b,
        a,
        z,
        state: c.state.clone(),
        ring: c.ring.clone(),
    })
}

/// After a commit of `a` of the verify's rows, for every `a` from 1 to its
/// rows, the next token's decode step at the position the commit stands at
/// reads the lane the commit names: its numerators, partials, and the lane it
/// writes are the stepped launches' bits, from a conv ring in which the dropped
/// rows' inputs sit in slots no kept position reads.
pub fn delta_rows_lanes_commit(h: &Harness<'_>, params: Params<'_>) -> Clause {
    let delta = &h.geom().delta;
    let fx = fixed(delta, params)?;
    let lane_len = elems(delta.state_elems(1), "a lane")?;
    let (g_len, p_len) = (g_row(delta), p_row(delta));
    let mut rng = oracle::Rng::new(0xD0B2);
    let mut parts = Vec::new();
    for case in RING_CASES {
        let (lanes, rows, pos0) = case;
        let (v, full) = ring_chunk(delta, &mut rng, case)?;
        let c = head_of(delta, &full, rows)?;
        for accepted in 1..=rows {
            let at = format!("{accepted} of {rows} rows kept, from {pos0} over {lanes} lanes");
            let committed = v.commit(accepted).map_err(|e| format!("{at}: {e}"))?;
            let next = Verify::new(lanes, committed.pos, 1)
                .map_err(|e| format!("{at}: {e}"))?
                .lane_to();
            let (mut bufs, verified) = run_ring(h, &fx, &c)?;
            if let Some(f) = verified.fault {
                return Err(format!("{at}: the verify raised {f}"));
            }
            let after = run_after_commit(
                h,
                &fx,
                &full,
                &mut bufs,
                rows,
                (committed.pos, committed.lane, next),
            )
            .map_err(|e| format!("{at}: {e}"))?;
            if let Some(f) = after.fault {
                return Err(format!("{at}: the token after the commit raised {f}"));
            }
            // The reference: the kept rows, then the next token, stepped, the
            // state ending where the token writes it.
            let steps = run_stepped(
                h,
                &fx,
                &writing(kept_then(delta, &full, accepted, rows)?, next),
            )?;
            same_bits(&format!("{at}: g"), &after.g, &steps.g[accepted * g_len..])?;
            same_bits(
                &format!("{at}: partials"),
                &after.partials,
                &steps.partials[accepted * p_len..],
            )?;
            // Every lane but the one the token wrote is what the verify left;
            // the one it wrote is the stepped launches' end state.
            let mut want = verified.state.clone();
            want[next * lane_len..(next + 1) * lane_len].copy_from_slice(lane_of_state(
                &steps.state,
                lane_len,
                next,
            ));
            same_bits(&format!("{at}: state"), &after.state, &want)?;
        }
        parts.push(format!("{rows}@{pos0}/{lanes}"));
    }
    Ok(Verdict::Noted(format!(
        "rows@first position/lanes {}: after a commit of every row count the next token is the \
         stepped launches' bits, reading the lane the commit names",
        parts.join(", ")
    )))
}

/// A lane-ring launch the lanes cannot serve is refused by name before it
/// encodes anything (more rows than lanes, copies, more rows than the conv ring
/// rolls back), and a last-row lane word that is not the lane of the position
/// after the last row raises the lane site on the device and writes no lane.
pub fn delta_rows_lanes_refused(h: &Harness<'_>, params: Params<'_>) -> Clause {
    let delta = &h.geom().delta;
    let fx = fixed(delta, params)?;
    let lane_len = elems(delta.state_elems(1), "a lane")?;
    let mut rng = oracle::Rng::new(0xD0B3);

    // The host's refusals: rows past the lanes, copies, rows past the roll-back.
    let wide = PASS_ROWS + 1;
    let refusals: [(&str, RingCase, bool, &str); 3] = [
        ("more rows than lanes", (2, 3, 4), false, "needs 3 lanes"),
        ("copies", (2, 2, 4), true, "no copies"),
        (
            "more rows than the conv ring rolls back",
            (wide, wide, 4),
            false,
            "rolls back at most",
        ),
    ];
    for (what, case, copies, needle) in refusals {
        let (lanes, rows, pos0) = case;
        // Built by hand, not by the verify bookkeeping, which refuses these.
        let mut c = chunk(delta, &mut rng, rows, pos0, (0, 0))?;
        c.state = draw(&mut rng, elems(delta.state_elems(lanes), "state")?, 0.04);
        c.lane_words = (0..lanes as u32).collect();
        c.lane_at = pos0 as usize % lanes;
        c.lane_to = (pos0 as usize + rows) % lanes;
        let after = [0u32];
        let mut bufs = Bufs::new(h, &fx, &c)?;
        let mut lane_copies = h
            .device
            .upload(h.stream, &vec![f32::NAN; lane_len])
            .map_err(why)?;
        let mut ring_copies = h
            .device
            .upload(h.stream, &vec![f32::NAN; c.ring.len()])
            .map_err(why)?;
        let strides = DeltaRowStrides::of(delta, N_K).ok_or("the strides do not fit")?;
        let prep_len = c.rows * prep_row(delta)?;
        let (g_row, p_row) = (g_row(delta), p_row(delta));
        let refused = h.encode(|encoder| {
            let snaps = if copies {
                Some(DeltaSnaps {
                    after: &after,
                    lane: lane_copies.window_mut(Elems(0), Elems(lane_len))?,
                    ring: ring_copies.window_mut(Elems(0), Elems(c.ring.len()))?,
                })
            } else {
                None
            };
            h.kernels.delta_rows_ring(
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
                    lanes: c.lane_words.len(),
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
        });
        match refused {
            Err(e) if e.starts_with("delta step") && e.contains(needle) => {}
            Err(e) => {
                return Err(format!(
                    "{what}: refused as {e:?}, not by name with {needle:?}"
                ));
            }
            Ok(()) => return Err(format!("{what}: the launch was accepted")),
        }
        let back = bufs.read(h, &c)?;
        same_bits(&format!("{what}: state"), &back.state, &c.state)?;
        if back.g.iter().any(|v| *v != 0.0) {
            return Err(format!("{what}: a refused launch wrote the numerator"));
        }
    }

    // The kernel's refusal: a last-row word that is not the lane of the
    // position after the last row. The launch is well formed, so the host
    // takes it.
    let case = (3, 2, 4);
    let (v, full) = ring_chunk(delta, &mut rng, case)?;
    let mut c = head_of(delta, &full, case.1)?;
    c.lane_to = (v.lane_to() + 1) % case.0;
    if c.lane_to == v.lane_to() {
        return Err("the case's wrong write word is the right one".to_string());
    }
    let (_, got) = run_ring(h, &fx, &c)?;
    match got.fault {
        Some(f) if f.layer == LAYER && f.sites & (1 << FaultSite::DeltaLane as u32) != 0 => {}
        Some(f) => {
            return Err(format!(
                "a write word that is not the last row's lane raised {f}, not {}",
                FaultSite::DeltaLane.name()
            ));
        }
        None => return Err("a write word that is not the last row's lane raised nothing".into()),
    }
    same_bits("a wrong write word: state", &got.state, &c.state)?;
    if got.g.iter().any(|v| !v.is_nan()) {
        return Err("a wrong write word left a number in the numerator".to_string());
    }
    Ok(Verdict::Noted(format!(
        "{} launches the lanes cannot serve are refused by name with nothing written, and a write word \
         that is not the last row's lane raises the lane site with every lane as it was",
        refusals.len()
    )))
}
