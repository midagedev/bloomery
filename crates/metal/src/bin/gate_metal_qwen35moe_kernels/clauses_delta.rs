//! The gated delta step's clauses.

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Device, Elems};
use bloomery_metal::geom::{DeltaGeom, Geometry};
use bloomery_metal::instance::REGISTERED;
use bloomery_metal::kernels::{BLOCK_VALUES, HeadMap, PARTIALS_PER_HEAD, PASS_ROWS, Quant, key_of};
use gguf::quant::GgmlType;

use super::common::{self, BadArg, Edit, LAYER};
use super::fixture::Fixture;
use super::oracle::{self, Kind};
use super::oracle_delta::{self, Fit, StepIn};
use super::synthetic;
use super::{ClauseRun, Harness, HeadNormLaunch};

/// The key heads of the first layer's conv: the fixture's, and the count the
/// clauses launch at.
pub(super) const KEY_HEADS: usize = 16;

/// A second key-head count, which divides the value heads too.
const KEY_HEADS_NARROW: usize = 8;

/// What the clauses' key-head counts are written for, held for every
/// registered row: the value heads are a multiple of both.
const _: () = {
    let mut i = 0;
    while i < REGISTERED.len() {
        let d = Geometry::of(&REGISTERED[i]).delta;
        assert!(d.value_heads.is_multiple_of(KEY_HEADS));
        assert!(d.value_heads.is_multiple_of(KEY_HEADS_NARROW));
        i += 1;
    }
};

pub(super) struct Fixed {
    pub(super) conv_w: Vec<f32>,
    pub(super) ssm_a: Vec<f32>,
    pub(super) dt: Vec<f32>,
    pub(super) norm_w: Vec<f32>,
    pub(super) eps: f32,
}

struct In<'a> {
    n_k: usize,
    map: HeadMap,
    lanes: usize,
    lane_at: usize,
    lane_to: usize,
    lane: &'a [u32],
    pos: &'a [u32],
    x: &'a [f32],
    conv_w: &'a [f32],
    b: &'a [f32],
    a: &'a [f32],
    dt: &'a [f32],
    ssm_a: &'a [f32],
    z: &'a [f32],
    norm_w: &'a [f32],
    state: &'a [f32],
    ring: &'a [f32],
    eps: f32,
}

struct Back {
    g: Vec<f32>,
    partials: Vec<f32>,
    state: Vec<f32>,
    ring: Vec<f32>,
}

/// Where a delta clause's fixed parameters and weights come from: the
/// fixture's first delta layer, or seeded draws for an instance whose delta
/// axes no fixture carries.
#[derive(Clone, Copy)]
pub(super) enum Params<'a> {
    Fixture(&'a Fixture),
    Drawn,
}

/// The fixed parameters of a delta layer at `delta`'s axes.
pub(super) fn fixed(delta: &DeltaGeom, params: Params<'_>) -> Result<Fixed, String> {
    match params {
        Params::Fixture(fixture) => from_fixture(delta, fixture),
        Params::Drawn => drawn(delta),
    }
}

/// Seeded draws over the ranges a layer's own tensors hold: the conv taps
/// and the decay slope `-exp(A_log)` (negative, so a decay stays below one),
/// the decay bias, a positive norm gain and the norm's epsilon.
fn drawn(delta: &DeltaGeom) -> Result<Fixed, String> {
    let ch = elems(delta.channels(KEY_HEADS), "channels")?;
    let mut rng = oracle::Rng::new(0xD4A9);
    let conv_w = draw(&mut rng, delta.conv_taps * ch, 0.5);
    let ssm_a = draw(&mut rng, delta.value_heads, 0.5)
        .iter()
        .map(|v| -0.55 - v)
        .collect();
    let dt = draw(&mut rng, delta.value_heads, 0.2);
    let norm_w = draw(&mut rng, delta.head_cols, 0.5)
        .iter()
        .map(|v| 1.0 + v)
        .collect();
    Ok(Fixed {
        conv_w,
        ssm_a,
        dt,
        norm_w,
        eps: 1e-6,
    })
}

fn from_fixture(delta: &DeltaGeom, fixture: &Fixture) -> Result<Fixed, String> {
    let ch = elems(delta.channels(KEY_HEADS), "channels")?;
    let file = gguf::Gguf::open(&fixture.path).map_err(|e| format!("conv weight: {e}"))?;
    let t = file
        .find("blk.0.ssm_conv1d.weight")
        .ok_or("the fixture has no blk.0.ssm_conv1d.weight")?;
    if t.ty != GgmlType::F32 || t.dims != [delta.conv_taps as u64, ch as u64] {
        return Err(format!(
            "blk.0.ssm_conv1d.weight is {:?} {:?}, the gate reads f32 [{}, {ch}]",
            t.ty, t.dims, delta.conv_taps
        ));
    }
    let bytes = file
        .data(t)
        .map_err(|e| format!("blk.0.ssm_conv1d.weight: {e}"))?;
    let words = delta.conv_taps * ch;
    if bytes.len() != words * 4 {
        return Err(format!("blk.0.ssm_conv1d.weight is {} bytes", bytes.len()));
    }
    let mut conv_w = vec![0f32; words];
    for (i, chunk) in bytes.as_chunks::<4>().0.iter().enumerate() {
        conv_w[i] = f32::from_le_bytes(*chunk);
    }
    Ok(Fixed {
        conv_w,
        ssm_a: fixture.f32s("blk.0.ssm_a", delta.value_heads)?,
        dt: fixture.f32s("blk.0.ssm_dt.bias", delta.value_heads)?,
        norm_w: fixture.f32s("blk.0.ssm_norm.weight", delta.head_cols)?,
        eps: fixture.rms_eps()?,
    })
}

pub(super) fn draw(rng: &mut oracle::Rng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| scale * rng.signed()).collect()
}

fn elems(n: Option<usize>, what: &str) -> Result<usize, String> {
    n.ok_or_else(|| format!("{what} does not fit"))
}

fn run(h: &Harness<'_>, inp: &In<'_>) -> Result<Back, String> {
    let (device, stream) = (h.device, h.stream);
    let delta = &h.geom().delta;
    let g_len = delta.value_heads * delta.head_cols;
    let p_len = delta.value_heads * PARTIALS_PER_HEAD;
    let x = device.upload(stream, inp.x).map_err(why)?;
    let conv_w = device.upload(stream, inp.conv_w).map_err(why)?;
    let b = device.upload(stream, inp.b).map_err(why)?;
    let a = device.upload(stream, inp.a).map_err(why)?;
    let dt = device.upload(stream, inp.dt).map_err(why)?;
    let ssm_a = device.upload(stream, inp.ssm_a).map_err(why)?;
    let z = device.upload(stream, inp.z).map_err(why)?;
    let norm_w = device.upload(stream, inp.norm_w).map_err(why)?;
    let lane = device.upload(stream, inp.lane).map_err(why)?;
    let mut state = device.upload(stream, inp.state).map_err(why)?;
    let mut ring = device.upload(stream, inp.ring).map_err(why)?;
    let mut partials = device.upload(stream, &vec![0f32; p_len]).map_err(why)?;
    let mut g = device.upload(stream, &vec![0f32; g_len]).map_err(why)?;
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.delta(
            encoder,
            bloomery_metal::kernels::DeltaParams {
                x: x.window(Elems(0), Elems(inp.x.len()))?,
                conv_w: conv_w.window(Elems(0), Elems(inp.conv_w.len()))?,
                b_raw: b.window(Elems(0), Elems(inp.b.len()))?,
                a_raw: a.window(Elems(0), Elems(inp.a.len()))?,
                dt_bias: dt.window(Elems(0), Elems(inp.dt.len()))?,
                ssm_a: ssm_a.window(Elems(0), Elems(inp.ssm_a.len()))?,
                z: z.window(Elems(0), Elems(inp.z.len()))?,
                norm_w: norm_w.window(Elems(0), Elems(inp.norm_w.len()))?,
                lane: lane.window(Elems(0), Elems(inp.lane.len()))?,
                state: state.window_mut(Elems(0), Elems(inp.state.len()))?,
                ring: ring.window_mut(Elems(0), Elems(inp.ring.len()))?,
                partials: partials.window_mut(Elems(0), Elems(p_len))?,
                g: g.window_mut(Elems(0), Elems(g_len))?,
                faults: device.fault_word(),
                n_v: delta.value_heads,
                head: delta.head_cols,
                taps: delta.conv_taps,
                n_k: inp.n_k,
                map: inp.map,
                lanes: inp.lanes,
                lane_at: inp.lane_at,
                lane_to: inp.lane_to,
                layer: LAYER,
                pos: inp.pos,
                eps: inp.eps,
            },
        )
    })?;
    let mut back = Back {
        g: vec![0f32; g_len],
        partials: vec![0f32; p_len],
        state: vec![0f32; inp.state.len()],
        ring: vec![0f32; inp.ring.len()],
    };
    device
        .read(stream, g.whole().map_err(why)?, &mut back.g)
        .map_err(why)?;
    device
        .read(stream, partials.whole().map_err(why)?, &mut back.partials)
        .map_err(why)?;
    device
        .read(stream, state.whole().map_err(why)?, &mut back.state)
        .map_err(why)?;
    device
        .read(stream, ring.whole().map_err(why)?, &mut back.ring)
        .map_err(why)?;
    Ok(back)
}

fn step_in<'a>(delta: &'a DeltaGeom, inp: &'a In<'a>) -> StepIn<'a> {
    StepIn {
        delta,
        n_k: inp.n_k,
        map: inp.map,
        lane: inp.lane[inp.lane_at],
        lane_out: inp.lane[inp.lane_to],
        pos: inp.pos[0],
        eps: inp.eps,
        lanes: inp.lanes,
        x: inp.x,
        conv_w: inp.conv_w,
        b: inp.b,
        a: inp.a,
        dt: inp.dt,
        ssm_a: inp.ssm_a,
        z: inp.z,
        norm_w: inp.norm_w,
        state: inp.state,
        ring: inp.ring,
    }
}

fn launch_fit(h: &Harness<'_>, inp: &In<'_>, tripwire: bool) -> Result<(Back, Fit), String> {
    let back = run(h, inp)?;
    common::expect_clean(h, "the step")?;
    let out = oracle_delta::step(&step_in(&h.geom().delta, inp))?;
    let fit = oracle_delta::fit(
        &back.g,
        &back.partials,
        &back.state,
        &back.ring,
        &out,
        tripwire,
    )?;
    Ok((back, fit))
}

fn inputs(
    delta: &DeltaGeom,
    rng: &mut oracle::Rng,
    n_k: usize,
    fixed: &Fixed,
    conv_w: &[f32],
) -> Result<Owned, String> {
    let ch = elems(delta.channels(n_k), "channels")?;
    Ok(Owned {
        x: draw(rng, ch, 0.3),
        b: draw(rng, delta.value_heads, 1.0),
        a: draw(rng, delta.value_heads, 2.0),
        z: draw(rng, delta.value_heads * delta.head_cols, 1.0),
        state: draw(rng, elems(delta.state_elems(1), "state")?, 0.04),
        ring: draw(rng, elems(delta.ring_elems(n_k), "ring")?, 0.05),
        conv_w: conv_w.to_vec(),
        ssm_a: fixed.ssm_a.clone(),
        dt: fixed.dt.clone(),
        norm_w: fixed.norm_w.clone(),
        eps: fixed.eps,
        n_k,
    })
}

struct Owned {
    x: Vec<f32>,
    b: Vec<f32>,
    a: Vec<f32>,
    z: Vec<f32>,
    state: Vec<f32>,
    ring: Vec<f32>,
    conv_w: Vec<f32>,
    ssm_a: Vec<f32>,
    dt: Vec<f32>,
    norm_w: Vec<f32>,
    eps: f32,
    n_k: usize,
}

struct View<'a> {
    map: HeadMap,
    lanes: usize,
    lane_at: usize,
    lane_to: usize,
    lane: &'a [u32],
    pos: &'a [u32],
    state: &'a [f32],
    ring: &'a [f32],
}

/// A view of `owned` whose step reads the lane the word at `lane_at` names
/// and writes the lane the word at `lane_to` names.
fn view_of<'a>(
    map: HeadMap,
    lanes: usize,
    (lane_at, lane_to): (usize, usize),
    lane: &'a [u32],
    pos: &'a [u32],
    owned: &'a Owned,
) -> View<'a> {
    View {
        map,
        lanes,
        lane_at,
        lane_to,
        lane,
        pos,
        state: &owned.state,
        ring: &owned.ring,
    }
}

impl Owned {
    fn as_in<'a>(&'a self, view: View<'a>) -> In<'a> {
        In {
            n_k: self.n_k,
            map: view.map,
            lanes: view.lanes,
            lane_at: view.lane_at,
            lane_to: view.lane_to,
            lane: view.lane,
            pos: view.pos,
            x: &self.x,
            conv_w: &self.conv_w,
            b: &self.b,
            a: &self.a,
            dt: &self.dt,
            ssm_a: &self.ssm_a,
            z: &self.z,
            norm_w: &self.norm_w,
            state: view.state,
            ring: view.ring,
            eps: self.eps,
        }
    }
}

fn keep_max(worst: &mut f64, widest: &mut f64, fit: &Fit) {
    *worst = worst.max(fit.ratio);
    *widest = widest.max(fit.widest_state);
}

/// One step from a host state, tiled and grouped, and a second key-head count.
pub fn delta_step(h: &Harness<'_>, params: Params<'_>) -> Clause {
    let delta = &h.geom().delta;
    let fixed = fixed(delta, params)?;
    let mut rng = oracle::Rng::new(0xD417);
    let owned = inputs(delta, &mut rng, KEY_HEADS, &fixed, &fixed.conv_w)?;
    let pos = [4u32];
    let lane = [0u32];
    let (tiled, tiled_fit) = launch_fit(
        h,
        &owned.as_in(view_of(HeadMap::Tiled, 1, (0, 0), &lane, &pos, &owned)),
        true,
    )?;
    let (grouped, grouped_fit) = launch_fit(
        h,
        &owned.as_in(view_of(HeadMap::Grouped, 1, (0, 0), &lane, &pos, &owned)),
        true,
    )?;
    if common::bits_eq(&tiled.g, &grouped.g) && common::bits_eq(&tiled.state, &grouped.state) {
        return Err("tiled and grouped wrote the same bits".to_string());
    }
    let mut rng = oracle::Rng::new(0xD418);
    let conv8 = draw(
        &mut rng,
        delta.conv_taps * elems(delta.channels(KEY_HEADS_NARROW), "channels")?,
        0.5,
    );
    let mut narrow = inputs(delta, &mut rng, KEY_HEADS_NARROW, &fixed, &conv8)?;
    let lanes = elems(delta.state_elems(2), "state")?;
    narrow.state = draw(&mut rng, lanes, 0.04);
    // Word 1 names lane 1, read; word 2 names lane 0, written. Word 0 names
    // no lane, so a step that reads a word by another index raises.
    let lane_words = [7u32, 1, 0];
    let (back, narrow_fit) = launch_fit(
        h,
        &narrow.as_in(view_of(
            HeadMap::Tiled,
            2,
            (1, 2),
            &lane_words,
            &pos,
            &narrow,
        )),
        true,
    )?;
    let lane1 = elems(delta.state_elems(1), "state")?;
    if !common::bits_eq(&back.state[lane1..], &narrow.state[lane1..]) {
        return Err("the lane the step read was written".to_string());
    }
    let mut worst = 0.0;
    let mut widest = 0.0;
    keep_max(&mut worst, &mut widest, &tiled_fit);
    keep_max(&mut worst, &mut widest, &grouped_fit);
    keep_max(&mut worst, &mut widest, &narrow_fit);
    Ok(Verdict::Noted(format!(
        "one-step band, roll factor 1, worst ratio {worst:.3}, widest state band {widest:.3e}"
    )))
}

/// Five steps, each oracled from the state and the ring the kernel wrote.
pub fn delta_roll(h: &Harness<'_>, params: Params<'_>) -> Clause {
    let delta = &h.geom().delta;
    let fixed = fixed(delta, params)?;
    let mut rng = oracle::Rng::new(0xD419);
    let owned = inputs(delta, &mut rng, KEY_HEADS, &fixed, &fixed.conv_w)?;
    let mut state = owned.state.clone();
    let mut ring = owned.ring.clone();
    let lane = [0u32];
    let mut worst = 0.0;
    let mut widest = 0.0;
    for step_i in 0u32..5 {
        let pos = [4 + step_i];
        let x = draw(&mut rng, owned.x.len(), 0.3);
        let b = draw(&mut rng, delta.value_heads, 1.0);
        let a = draw(&mut rng, delta.value_heads, 2.0);
        let z = draw(&mut rng, delta.value_heads * delta.head_cols, 1.0);
        let snap_state = state.clone();
        let snap_ring = ring.clone();
        let inp = In {
            n_k: KEY_HEADS,
            map: HeadMap::Tiled,
            lanes: 1,
            lane_at: 0,
            lane_to: 0,
            lane: &lane,
            pos: &pos,
            x: &x,
            conv_w: &owned.conv_w,
            b: &b,
            a: &a,
            dt: &owned.dt,
            ssm_a: &owned.ssm_a,
            z: &z,
            norm_w: &owned.norm_w,
            state: &snap_state,
            ring: &snap_ring,
            eps: owned.eps,
        };
        let (back, fit) = launch_fit(h, &inp, true)?;
        keep_max(&mut worst, &mut widest, &fit);
        state = back.state;
        ring = back.ring;
    }
    Ok(Verdict::Noted(format!(
        "roll factor 1, so the five steps are held to the one-step radius, worst ratio {worst:.3}, widest state band {widest:.3e}"
    )))
}

fn written_once(
    ring_rows: usize,
    out: &[f32],
    before: &[f32],
    x: &[f32],
    pos: u32,
) -> Result<(), String> {
    let ch = x.len();
    let slot = (pos as usize) % ring_rows;
    for c in 0..ch {
        if out[slot * ch + c].to_bits() != x[c].to_bits() {
            return Err(format!("slot {slot} channel {c} was not the conv input"));
        }
    }
    for s in 0..ring_rows {
        if s == slot {
            continue;
        }
        let at = s * ch;
        if !common::bits_eq(&out[at..at + ch], &before[at..at + ch]) {
            return Err(format!("slot {s} was written at position {pos}"));
        }
    }
    Ok(())
}

fn ring_launch(
    h: &Harness<'_>,
    owned: &Owned,
    ring: &[f32],
    pos: u32,
    tripwire: bool,
) -> Result<Back, String> {
    let lane = [0u32];
    let pos = [pos];
    let (back, _) = launch_fit(
        h,
        &owned.as_in(View {
            map: HeadMap::Tiled,
            lanes: 1,
            lane_at: 0,
            lane_to: 0,
            lane: &lane,
            pos: &pos,
            state: &owned.state,
            ring,
        }),
        tripwire,
    )?;
    written_once(h.geom().delta.ring_rows, &back.ring, ring, &owned.x, pos[0])?;
    Ok(back)
}

/// The conv reads a predecessor only while `pos >= d`, and writes
/// `pos mod ring_rows` once.
pub fn delta_ring(h: &Harness<'_>, params: Params<'_>) -> Clause {
    let delta = &h.geom().delta;
    let fixed = fixed(delta, params)?;
    let mut rng = oracle::Rng::new(0xD41A);
    let owned = inputs(delta, &mut rng, KEY_HEADS, &fixed, &fixed.conv_w)?;
    let ch = owned.x.len();
    let zeros = vec![0f32; owned.ring.len()];
    let mut sentinel = vec![1e6f32; owned.ring.len()];
    let at_zero_s = ring_launch(h, &owned, &sentinel, 0, false)?;
    let at_zero = ring_launch(h, &owned, &zeros, 0, true)?;
    if !common::bits_eq(&at_zero_s.g, &at_zero.g)
        || !common::bits_eq(&at_zero_s.state, &at_zero.state)
    {
        return Err("position 0 read the ring".to_string());
    }
    sentinel.fill(0.0);
    sentinel[..ch].fill(1e6);
    let live = ring_launch(h, &owned, &sentinel, 1, false)?;
    let quiet = ring_launch(h, &owned, &zeros, 1, true)?;
    if common::bits_eq(&live.g, &quiet.g) {
        return Err("position 1 ignored the ring slot it reads".to_string());
    }
    let mut unread_a = zeros.clone();
    let unread_b = zeros.clone();
    let unread = (1usize + delta.ring_rows - 2) % delta.ring_rows;
    unread_a[unread * ch..(unread + 1) * ch].fill(1e6);
    let a = ring_launch(h, &owned, &unread_a, 1, false)?;
    let b = ring_launch(h, &owned, &unread_b, 1, true)?;
    if !common::bits_eq(&a.g, &b.g) || !common::bits_eq(&a.state, &b.state) {
        return Err(format!("position 1 read unread slot {unread}"));
    }
    let mut pred = zeros.clone();
    pred[..ch].fill(1e6);
    let at_two = ring_launch(h, &owned, &pred, 2, false)?;
    let at_two_zero = ring_launch(h, &owned, &zeros, 2, true)?;
    if common::bits_eq(&at_two.g, &at_two_zero.g) {
        return Err("position 2 ignored the distance-2 ring slot".to_string());
    }
    let ring_rows = u32::try_from(delta.ring_rows).expect("the ring length fits the position word");
    let last_safe = u32::MAX - ring_rows;
    for pos in [last_safe, last_safe + 1, u32::MAX] {
        let equivalent = pos % ring_rows + ring_rows;
        let high = ring_launch(h, &owned, &owned.ring, pos, true)?;
        let low = ring_launch(h, &owned, &owned.ring, equivalent, true)?;
        if !common::bits_eq(&high.g, &low.g)
            || !common::bits_eq(&high.partials, &low.partials)
            || !common::bits_eq(&high.state, &low.state)
            || !common::bits_eq(&high.ring, &low.ring)
        {
            return Err(format!(
                "position {pos} differs from ring-equivalent position {equivalent}"
            ));
        }
    }
    Ok(Verdict::Noted(
        "position 0 reads no ring slot, positions 1 and 2 read the live slot, one slot is written, and positions through u32::MAX match their ring-equivalent positions"
            .to_string(),
    ))
}

/// The partials and the numerator match the oracle, and the head-norm launch accepts them.
pub fn delta_partial(h: &Harness<'_>, params: Params<'_>) -> Clause {
    let delta = &h.geom().delta;
    let fixed = fixed(delta, params)?;
    let mut rng = oracle::Rng::new(0xD41B);
    let owned = inputs(delta, &mut rng, KEY_HEADS, &fixed, &fixed.conv_w)?;
    let pos = [4u32];
    let lane = [0u32];
    let (back, fit) = launch_fit(
        h,
        &owned.as_in(view_of(HeadMap::Tiled, 1, (0, 0), &lane, &pos, &owned)),
        true,
    )?;
    let rows = 32;
    let cols = delta.head_cols * delta.value_heads;
    let drawn_weights;
    let weights: &[u8] = match params {
        Params::Fixture(fixture) => {
            fixture.rows("blk.0.ssm_out.weight", GgmlType::Q4_K, cols, rows)?
        }
        Params::Drawn => {
            drawn_weights = synthetic::matrix(Quant::Q4K, rows, cols, 0x41B);
            &drawn_weights
        }
    };
    let y0 = draw(&mut rng, rows, 0.1);
    h.clear_fault()?;
    let got = h.headnorm(&HeadNormLaunch {
        quant: Quant::Q4K,
        rows,
        cols,
        weights,
        g: &back.g,
        partials: &back.partials,
        y0: &y0,
        eps: owned.eps,
        layer: LAYER,
    })?;
    common::expect_clean(h, "the head norm")?;
    let x = common::head_scaled(
        &back.g,
        &back.partials,
        delta.value_heads,
        delta.head_cols,
        PARTIALS_PER_HEAD,
        owned.eps,
    );
    let reference = oracle::reference(Quant::Q4K, weights, rows, cols, &x)?;
    let kind = Kind::HeadNorm(oracle::HeadGeometry {
        heads: delta.value_heads,
        cols_per_head: delta.head_cols,
        partials_per_head: PARTIALS_PER_HEAD,
    });
    let band = oracle::band(Quant::Q4K, cols / BLOCK_VALUES, kind);
    let head_ratio = oracle::check(&got, &reference, band, kind, Some(&y0))?;
    Ok(Verdict::Noted(format!(
        "partials[h * 8 + b] and g[h * 128 + c], worst ratio {:.3}, widest state band {:.3e}, head norm worst ratio {head_ratio:.3}",
        fit.ratio, fit.widest_state
    )))
}

fn poison_column(h: &Harness<'_>, owned: &Owned) -> Result<(), String> {
    if key_of(3, 0) != 12 {
        return Err(format!(
            "key_of(3, 0) is {}, the lane's first key is 12",
            key_of(3, 0)
        ));
    }
    let delta = &h.geom().delta;
    let mut state = owned.state.clone();
    let at = delta.state_index(0, 0, 0, 12);
    state[at] = f32::NAN;
    let pos = [4u32];
    let lane = [0u32];
    let back = run(
        h,
        &owned.as_in(View {
            map: HeadMap::Tiled,
            lanes: 1,
            lane_at: 0,
            lane_to: 0,
            lane: &lane,
            pos: &pos,
            state: &state,
            ring: &owned.ring,
        }),
    )?;
    common::expect_fault(h, "a non-finite state lane", FaultSite::DeltaLane)?;
    for j in 0..8 {
        for i in 0..16 {
            let key = key_of(j, i);
            let idx = delta.state_index(0, 0, 0, key);
            if j == 3 {
                if !back.state[idx].is_nan() {
                    return Err(format!(
                        "key {key} of the non-finite lane is {:?}",
                        back.state[idx]
                    ));
                }
            } else if back.state[idx].to_bits() != owned.state[idx].to_bits() {
                return Err(format!("lane {j} of the bad column was updated"));
            }
        }
    }
    if !back.g[0].is_nan() {
        return Err(format!("the bad column's numerator is {:?}", back.g[0]));
    }
    if !back.partials[0].is_nan() {
        return Err(format!(
            "the bad column's partial is {:?}",
            back.partials[0]
        ));
    }
    let clean = oracle_delta::step(&StepIn {
        delta,
        n_k: owned.n_k,
        map: HeadMap::Tiled,
        lane: 0,
        lane_out: 0,
        pos: 4,
        eps: owned.eps,
        lanes: 1,
        x: &owned.x,
        conv_w: &owned.conv_w,
        b: &owned.b,
        a: &owned.a,
        dt: &owned.dt,
        ssm_a: &owned.ssm_a,
        z: &owned.z,
        norm_w: &owned.norm_w,
        state: &owned.state,
        ring: &owned.ring,
    })?;
    let mut g = back.g.clone();
    let mut partials = back.partials.clone();
    let mut got_state = back.state.clone();
    g[0] = clean.g[0] as f32;
    partials[0] = clean.partials[0] as f32;
    for j in 0..8 {
        for i in 0..16 {
            let idx = delta.state_index(0, 0, 0, key_of(j, i));
            got_state[idx] = clean.state[idx] as f32;
        }
    }
    oracle_delta::fit(&g, &partials, &got_state, &back.ring, &clean, true)?;
    Ok(())
}

/// A read or a write lane word past the lanes, and one non-finite key, each
/// raise the lane site.
pub fn delta_nan(h: &Harness<'_>, params: Params<'_>) -> Clause {
    let delta = &h.geom().delta;
    let fixed = fixed(delta, params)?;
    let mut rng = oracle::Rng::new(0xD41C);
    let owned = inputs(delta, &mut rng, KEY_HEADS, &fixed, &fixed.conv_w)?;
    let pos = [4u32];
    // Word 0 names lane 0, word 1 a lane past the one there is.
    let lane = [0u32, 4];
    for (what, at) in [("a read", (1, 0)), ("a write", (0, 1))] {
        let back = run(
            h,
            &owned.as_in(view_of(HeadMap::Tiled, 1, at, &lane, &pos, &owned)),
        )?;
        common::expect_fault(
            h,
            &format!("{what} lane word past the lanes"),
            FaultSite::DeltaLane,
        )?;
        if !common::bits_eq(&back.state, &owned.state) {
            return Err(format!("{what} lane word past the lanes wrote the state"));
        }
        if back.g.iter().any(|v| !v.is_nan()) || back.partials.iter().any(|v| !v.is_nan()) {
            return Err(format!(
                "{what} lane word past the lanes left a finite numerator or partial"
            ));
        }
        written_once(delta.ring_rows, &back.ring, &owned.ring, &owned.x, 4)?;
    }
    poison_column(h, &owned)?;
    Ok(Verdict::Noted(
        "a read or a write lane word past the lanes left the state untouched, and one non-finite key wrote NaN on that lane only"
            .to_string(),
    ))
}

/// The buffers of the refusal table: one upload per window, and the shapes
/// every well-formed case of the table shares.
struct Refusals {
    ch: usize,
    state_len: usize,
    ring_len: usize,
    g_len: usize,
    p_len: usize,
    x: bloomery_metal::MetalBuffer<f32>,
    conv_w: bloomery_metal::MetalBuffer<f32>,
    b: bloomery_metal::MetalBuffer<f32>,
    a: bloomery_metal::MetalBuffer<f32>,
    dt: bloomery_metal::MetalBuffer<f32>,
    ssm_a: bloomery_metal::MetalBuffer<f32>,
    z: bloomery_metal::MetalBuffer<f32>,
    norm_w: bloomery_metal::MetalBuffer<f32>,
    lane: bloomery_metal::MetalBuffer<u32>,
    state: bloomery_metal::MetalBuffer<f32>,
    ring: bloomery_metal::MetalBuffer<f32>,
    partials: bloomery_metal::MetalBuffer<f32>,
    g: bloomery_metal::MetalBuffer<f32>,
}

/// The launch of one case: a shape, a position window or the numerator
/// window, one of them bad.
#[derive(Clone, Copy)]
struct Args {
    n_v: usize,
    head: usize,
    taps: usize,
    n_k: usize,
    /// The position window's length; `refuse` picks the slice it stands for.
    pos_len: usize,
    g: usize,
    /// The written lane's word index, in a lane window of one word.
    lane_to: usize,
}

impl Args {
    const fn whole(delta: &DeltaGeom, n_k: usize, g_len: usize) -> Self {
        Self {
            n_v: delta.value_heads,
            head: delta.head_cols,
            taps: delta.conv_taps,
            n_k,
            pos_len: 1,
            g: g_len,
            lane_to: 0,
        }
    }
}

/// A text a refusal names, for a needle that carries a count of the row.
fn named(text: String) -> &'static str {
    text.leak()
}

/// The position windows a case can ask for by length.
const POS_ONE: [u32; 1] = [4];
const POS_TWO: [u32; 2] = [0, 1];
const POS_LONG: [u32; PASS_ROWS + 1] = [0; PASS_ROWS + 1];

fn refuse(
    h: &Harness<'_>,
    b: &mut Refusals,
    a: &Args,
    encoder: &mut bloomery_metal::encode::ComputeEncoder<'_>,
) -> Result<(), bloomery_device::DeviceError> {
    let delta = &h.geom().delta;
    let pos: &[u32] = if a.pos_len == PASS_ROWS + 1 {
        &POS_LONG
    } else if a.pos_len == 2 {
        &POS_TWO
    } else {
        &POS_ONE
    };
    h.kernels.delta(
        encoder,
        bloomery_metal::kernels::DeltaParams {
            x: b.x.window(Elems(0), Elems(b.ch))?,
            conv_w: b.conv_w.window(Elems(0), Elems(delta.conv_taps * b.ch))?,
            b_raw: b.b.window(Elems(0), Elems(delta.value_heads))?,
            a_raw: b.a.window(Elems(0), Elems(delta.value_heads))?,
            dt_bias: b.dt.window(Elems(0), Elems(delta.value_heads))?,
            ssm_a: b.ssm_a.window(Elems(0), Elems(delta.value_heads))?,
            z: b.z.window(Elems(0), Elems(b.g_len))?,
            norm_w: b.norm_w.window(Elems(0), Elems(delta.head_cols))?,
            lane: b.lane.window(Elems(0), Elems(1))?,
            state: b.state.window_mut(Elems(0), Elems(b.state_len))?,
            ring: b.ring.window_mut(Elems(0), Elems(b.ring_len))?,
            partials: b.partials.window_mut(Elems(0), Elems(b.p_len))?,
            g: b.g.window_mut(Elems(0), Elems(a.g))?,
            faults: h.device.fault_word(),
            n_v: a.n_v,
            head: a.head,
            taps: a.taps,
            n_k: a.n_k,
            map: HeadMap::Tiled,
            lanes: 1,
            lane_at: 0,
            lane_to: a.lane_to,
            layer: LAYER,
            pos,
            eps: 1e-6,
        },
    )
}

fn bad<'a>(
    what: &'static str,
    needle: &'static str,
    edit: impl FnOnce(&mut Args) + 'a,
    h: &'a Harness<'_>,
    whole: Args,
) -> BadArg<'a, Refusals> {
    BadArg::new(
        what,
        needle,
        Box::new(move |b, encoder| {
            let mut a = whole;
            edit(&mut a);
            refuse(h, b, &a, encoder)
        }),
    )
}

/// A shape, a position window and a numerator window the launch refuses by name.
pub fn delta_bounds(h: &Harness<'_>) -> Clause {
    let device = h.device;
    let stream = h.stream;
    let delta = &h.geom().delta;
    let n_k = KEY_HEADS;
    let ch = elems(delta.channels(n_k), "channels")?;
    let state_len = elems(delta.state_elems(1), "state")?;
    let ring_len = elems(delta.ring_elems(n_k), "ring")?;
    let g_len = delta.value_heads * delta.head_cols;
    let p_len = delta.value_heads * PARTIALS_PER_HEAD;
    let heads = vec![1f32; delta.value_heads];
    let mut bufs = Refusals {
        ch,
        state_len,
        ring_len,
        g_len,
        p_len,
        x: device.upload(stream, &vec![1f32; ch]).map_err(why)?,
        conv_w: device
            .upload(stream, &vec![1f32; delta.conv_taps * ch])
            .map_err(why)?,
        b: device.upload(stream, &heads).map_err(why)?,
        a: device.upload(stream, &heads).map_err(why)?,
        dt: device.upload(stream, &heads).map_err(why)?,
        ssm_a: device.upload(stream, &heads).map_err(why)?,
        z: device.upload(stream, &vec![1f32; g_len]).map_err(why)?,
        norm_w: device
            .upload(stream, &vec![1f32; delta.head_cols])
            .map_err(why)?,
        lane: device.upload(stream, &[0u32]).map_err(why)?,
        state: device.upload(stream, &vec![1f32; state_len]).map_err(why)?,
        ring: device.upload(stream, &vec![1f32; ring_len]).map_err(why)?,
        partials: device.upload(stream, &vec![1f32; p_len]).map_err(why)?,
        g: device.upload(stream, &vec![1f32; g_len + 1]).map_err(why)?,
    };
    let whole = Args::whole(delta, n_k, g_len);
    let (value_heads, head_cols, conv_taps) = (delta.value_heads, delta.head_cols, delta.conv_taps);
    // The smallest count of key heads that does not divide the value heads.
    let non_divisor = (2..)
        .find(|n| !value_heads.is_multiple_of(*n))
        .ok_or("every count divides the value heads")?;
    let cases = vec![
        bad(
            "half the value heads",
            named(format!("n_v {value_heads}")),
            move |a| a.n_v = value_heads / 2,
            h,
            whole,
        ),
        bad(
            "half the head",
            named(format!("head {head_cols}")),
            move |a| a.head = head_cols / 2,
            h,
            whole,
        ),
        bad(
            "one tap fewer",
            "taps",
            move |a| a.taps = conv_taps - 1,
            h,
            whole,
        ),
        bad(
            "a count of key heads that does not divide the value heads",
            "n_k",
            move |a| a.n_k = non_divisor,
            h,
            whole,
        ),
        bad(
            "a position window of 33 words",
            "position window",
            |a| a.pos_len = PASS_ROWS + 1,
            h,
            whole,
        ),
        bad(
            "a position window of 2 words",
            "position window",
            |a| a.pos_len = 2,
            h,
            whole,
        ),
        bad(
            "a short numerator window",
            "g window",
            |a| a.g = g_len - 1,
            h,
            whole,
        ),
        bad(
            "a long numerator window",
            "g window",
            |a| a.g = g_len + 1,
            h,
            whole,
        ),
        bad(
            "a write lane index past the lane window",
            "write lane index",
            |a| a.lane_to = 1,
            h,
            whole,
        ),
    ];
    let unchanged = |h: &Harness<'_>, b: &Refusals| {
        common::still(h, &b.g, |v| *v == 1.0)?;
        common::still(h, &b.state, |v| *v == 1.0)?;
        common::still(h, &b.partials, |v| *v == 1.0)?;
        common::still(h, &b.ring, |v| *v == 1.0)
    };
    let count = common::refuse_launches(h, "delta step", &mut bufs, cases, unchanged)?;
    let good: Edit<'_, Refusals> = Box::new(|b, encoder| {
        let a = whole;
        refuse(h, b, &a, encoder)
    });
    common::accepted_writes(h, &mut bufs, good, |h, b| {
        common::moved(h, &b.g, |v| *v != 1.0)
    })?;
    Ok(Verdict::Noted(format!(
        "{count} refusals, each by name, nothing written"
    )))
}

/// The gated-delta clauses, in gate order.
pub fn table<'a>(ctx: &'a super::ClauseCtx<'a>) -> Vec<(&'static str, super::ClauseRun<'a>)> {
    vec![
        ("delta_step", Box::new(|| ctx.with_params(delta_step))),
        ("delta_roll", Box::new(|| ctx.with_params(delta_roll))),
        ("delta_ring", Box::new(|| ctx.with_params(delta_ring))),
        ("delta_partial", Box::new(|| ctx.with_params(delta_partial))),
        ("delta_nan", Box::new(|| ctx.with_params(delta_nan))),
        ("delta_bounds", Box::new(|| delta_bounds(ctx.harness))),
    ]
}

/// The head-normed mat-vec's clause on synthetic weights at the layer's
/// value heads: a clean fault word before and after, as the synthetic
/// tables run theirs.
fn headnorm_on<'a>(h: &'a super::Harness<'a>, quant: Quant) -> ClauseRun<'a> {
    Box::new(move || {
        h.clear_fault()?;
        let verdict = super::clauses_synth::headnorm(h, quant);
        h.clear_fault()?;
        verdict
    })
}

/// The clauses of the kernels the delta axes shape, on drawn parameters: for
/// an instance whose delta layer is not the fixture's. The step's (the
/// fixture-free `delta_bounds` too), the head-normed mat-vecs over Q4_K and
/// the MLX 4-bit groups, and the head norm over a chunk of rows.
pub fn wide<'a>(ctx: &'a super::ClauseCtx<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    let h = ctx.harness;
    vec![
        ("delta_step", Box::new(|| delta_step(h, Params::Drawn))),
        ("delta_roll", Box::new(|| delta_roll(h, Params::Drawn))),
        ("delta_ring", Box::new(|| delta_ring(h, Params::Drawn))),
        (
            "delta_partial",
            Box::new(|| delta_partial(h, Params::Drawn)),
        ),
        ("delta_nan", Box::new(|| delta_nan(h, Params::Drawn))),
        ("delta_bounds", Box::new(|| delta_bounds(h))),
        ("headnorm_q4k", headnorm_on(h, Quant::Q4K)),
        ("headnorm_a4", headnorm_on(h, Quant::A4)),
        (
            "headnorm_rows",
            Box::new(|| super::clauses_route::headnorm_rows(h)),
        ),
    ]
}
