//! The gated delta step's clauses.

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Device, Elems};
use bloomery_metal::kernels::{
    BLOCK_VALUES, CONV_TAPS, HEAD_COLS, HeadMap, PARTIALS_PER_HEAD, PASS_ROWS, Quant, RING_ROWS,
    VALUE_HEADS, channels, key_of, ring_elems, state_elems, state_index,
};
use gguf::quant::GgmlType;

use super::fixture::Fixture;
use super::oracle::{self, Kind};
use super::oracle_delta::{self, Fit, StepIn};
use super::{Harness, HeadNormLaunch};

const LAYER: u32 = 5;

struct Fixed {
    conv_w: Vec<f32>,
    ssm_a: Vec<f32>,
    dt: Vec<f32>,
    norm_w: Vec<f32>,
    eps: f32,
}

struct In<'a> {
    n_k: usize,
    map: HeadMap,
    lanes: usize,
    lane_at: usize,
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

fn fixed(fixture: &Fixture) -> Result<Fixed, String> {
    let file = gguf::Gguf::open(&fixture.path).map_err(|e| format!("conv weight: {e}"))?;
    let t = file
        .find("blk.0.ssm_conv1d.weight")
        .ok_or("the fixture has no blk.0.ssm_conv1d.weight")?;
    if t.ty != GgmlType::F32 || t.dims != [4, 8192] {
        return Err(format!(
            "blk.0.ssm_conv1d.weight is {:?} {:?}, the gate reads f32 [4, 8192]",
            t.ty, t.dims
        ));
    }
    let bytes = file
        .data(t)
        .map_err(|e| format!("blk.0.ssm_conv1d.weight: {e}"))?;
    if bytes.len() != 4 * 8192 * 4 {
        return Err(format!("blk.0.ssm_conv1d.weight is {} bytes", bytes.len()));
    }
    let mut conv_w = vec![0f32; 4 * 8192];
    for (i, chunk) in bytes.as_chunks::<4>().0.iter().enumerate() {
        conv_w[i] = f32::from_le_bytes(*chunk);
    }
    Ok(Fixed {
        conv_w,
        ssm_a: fixture.f32s("blk.0.ssm_a", VALUE_HEADS)?,
        dt: fixture.f32s("blk.0.ssm_dt.bias", VALUE_HEADS)?,
        norm_w: fixture.f32s("blk.0.ssm_norm.weight", HEAD_COLS)?,
        eps: fixture.rms_eps()?,
    })
}

fn draw(rng: &mut oracle::Rng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| scale * rng.signed()).collect()
}

fn elems(n: Option<usize>, what: &str) -> Result<usize, String> {
    n.ok_or_else(|| format!("{what} does not fit"))
}

fn run(h: &Harness<'_>, inp: &In<'_>) -> Result<Back, String> {
    let (device, stream) = (h.device, h.stream);
    let g_len = VALUE_HEADS * HEAD_COLS;
    let p_len = VALUE_HEADS * PARTIALS_PER_HEAD;
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
                n_v: VALUE_HEADS,
                head: HEAD_COLS,
                taps: CONV_TAPS,
                n_k: inp.n_k,
                map: inp.map,
                lanes: inp.lanes,
                lane_at: inp.lane_at,
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

fn step_in<'a>(inp: &'a In<'a>) -> StepIn<'a> {
    StepIn {
        n_k: inp.n_k,
        map: inp.map,
        lane: inp.lane[inp.lane_at],
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
    expect_clean(h, "the step")?;
    let out = oracle_delta::step(&step_in(inp))?;
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

fn expect_clean(h: &Harness<'_>, what: &str) -> Result<(), String> {
    if let Some(fault) = h.fault()? {
        return Err(format!("{what} raised {fault}"));
    }
    Ok(())
}

fn expect_fault(h: &Harness<'_>, what: &str, site: FaultSite) -> Result<(), String> {
    match h.fault()? {
        Some(f)
            if f.layer == LAYER && f.code == site as u32 && f.sites == 1u32 << (site as u32) =>
        {
            Ok(())
        }
        Some(f) => Err(format!(
            "{what} raised {f}, not {} in layer {LAYER}",
            site.name()
        )),
        None => Err(format!("{what} raised nothing, not {}", site.name())),
    }
}

fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits())
}

fn inputs(
    rng: &mut oracle::Rng,
    n_k: usize,
    fixed: &Fixed,
    conv_w: &[f32],
) -> Result<Owned, String> {
    let ch = elems(channels(n_k), "channels")?;
    Ok(Owned {
        x: draw(rng, ch, 0.3),
        b: draw(rng, VALUE_HEADS, 1.0),
        a: draw(rng, VALUE_HEADS, 2.0),
        z: draw(rng, VALUE_HEADS * HEAD_COLS, 1.0),
        state: draw(rng, elems(state_elems(1), "state")?, 0.04),
        ring: draw(rng, elems(ring_elems(n_k), "ring")?, 0.05),
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
    lane: &'a [u32],
    pos: &'a [u32],
    state: &'a [f32],
    ring: &'a [f32],
}

fn view_of<'a>(
    map: HeadMap,
    lanes: usize,
    lane_at: usize,
    lane: &'a [u32],
    pos: &'a [u32],
    owned: &'a Owned,
) -> View<'a> {
    View {
        map,
        lanes,
        lane_at,
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
pub fn delta_step(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let fixed = fixed(fixture)?;
    let mut rng = oracle::Rng::new(0xD417);
    let owned = inputs(&mut rng, 16, &fixed, &fixed.conv_w)?;
    let pos = [4u32];
    let lane = [0u32];
    let (tiled, tiled_fit) = launch_fit(
        h,
        &owned.as_in(view_of(HeadMap::Tiled, 1, 0, &lane, &pos, &owned)),
        true,
    )?;
    let (grouped, grouped_fit) = launch_fit(
        h,
        &owned.as_in(view_of(HeadMap::Grouped, 1, 0, &lane, &pos, &owned)),
        true,
    )?;
    if bits_eq(&tiled.g, &grouped.g) && bits_eq(&tiled.state, &grouped.state) {
        return Err("tiled and grouped wrote the same bits".to_string());
    }
    let mut rng = oracle::Rng::new(0xD418);
    let conv8 = draw(&mut rng, CONV_TAPS * elems(channels(8), "channels")?, 0.5);
    let mut narrow = inputs(&mut rng, 8, &fixed, &conv8)?;
    let lanes = elems(state_elems(2), "state")?;
    narrow.state = draw(&mut rng, lanes, 0.04);
    let lane_words = [7u32, 1];
    let (back, narrow_fit) = launch_fit(
        h,
        &narrow.as_in(view_of(HeadMap::Tiled, 2, 1, &lane_words, &pos, &narrow)),
        true,
    )?;
    let lane0 = elems(state_elems(1), "state")?;
    if !bits_eq(&back.state[..lane0], &narrow.state[..lane0]) {
        return Err("the unselected lane of the state was written".to_string());
    }
    let mut worst = 0.0;
    let mut widest = 0.0;
    keep_max(&mut worst, &mut widest, &tiled_fit);
    keep_max(&mut worst, &mut widest, &grouped_fit);
    keep_max(&mut worst, &mut widest, &narrow_fit);
    Ok(Verdict::Noted(format!(
        "one-step band, roll factor {}, worst ratio {worst:.3}, widest state band {widest:.3e}",
        oracle_delta::ROLL_FACTOR
    )))
}

/// Five steps, each oracled from the state and the ring the kernel wrote.
pub fn delta_roll(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let fixed = fixed(fixture)?;
    let mut rng = oracle::Rng::new(0xD419);
    let owned = inputs(&mut rng, 16, &fixed, &fixed.conv_w)?;
    let mut state = owned.state.clone();
    let mut ring = owned.ring.clone();
    let lane = [0u32];
    let mut worst = 0.0;
    let mut widest = 0.0;
    for step_i in 0u32..5 {
        let pos = [4 + step_i];
        let x = draw(&mut rng, owned.x.len(), 0.3);
        let b = draw(&mut rng, VALUE_HEADS, 1.0);
        let a = draw(&mut rng, VALUE_HEADS, 2.0);
        let z = draw(&mut rng, VALUE_HEADS * HEAD_COLS, 1.0);
        let snap_state = state.clone();
        let snap_ring = ring.clone();
        let inp = In {
            n_k: 16,
            map: HeadMap::Tiled,
            lanes: 1,
            lane_at: 0,
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
        "roll factor {}, so the five steps are held to the one-step radius, worst ratio {worst:.3}, widest state band {widest:.3e}",
        oracle_delta::ROLL_FACTOR
    )))
}

fn written_once(out: &[f32], before: &[f32], x: &[f32], pos: u32) -> Result<(), String> {
    let ch = x.len();
    let slot = (pos as usize) % RING_ROWS;
    for c in 0..ch {
        if out[slot * ch + c].to_bits() != x[c].to_bits() {
            return Err(format!("slot {slot} channel {c} was not the conv input"));
        }
    }
    for s in 0..RING_ROWS {
        if s == slot {
            continue;
        }
        let at = s * ch;
        if !bits_eq(&out[at..at + ch], &before[at..at + ch]) {
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
            lane: &lane,
            pos: &pos,
            state: &owned.state,
            ring,
        }),
        tripwire,
    )?;
    written_once(&back.ring, ring, &owned.x, pos[0])?;
    Ok(back)
}

/// The conv reads a predecessor only while `pos >= d`, and writes `pos mod 11` once.
pub fn delta_ring(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let fixed = fixed(fixture)?;
    let mut rng = oracle::Rng::new(0xD41A);
    let owned = inputs(&mut rng, 16, &fixed, &fixed.conv_w)?;
    let ch = owned.x.len();
    let zeros = vec![0f32; owned.ring.len()];
    let mut sentinel = vec![1e6f32; owned.ring.len()];
    let at_zero_s = ring_launch(h, &owned, &sentinel, 0, false)?;
    let at_zero = ring_launch(h, &owned, &zeros, 0, true)?;
    if !bits_eq(&at_zero_s.g, &at_zero.g) || !bits_eq(&at_zero_s.state, &at_zero.state) {
        return Err("position 0 read the ring".to_string());
    }
    sentinel.fill(0.0);
    sentinel[..ch].fill(1e6);
    let live = ring_launch(h, &owned, &sentinel, 1, false)?;
    let quiet = ring_launch(h, &owned, &zeros, 1, true)?;
    if bits_eq(&live.g, &quiet.g) {
        return Err("position 1 ignored the ring slot it reads".to_string());
    }
    let mut unread_a = zeros.clone();
    let unread_b = zeros.clone();
    let unread = (1usize + RING_ROWS - 2) % RING_ROWS;
    unread_a[unread * ch..(unread + 1) * ch].fill(1e6);
    let a = ring_launch(h, &owned, &unread_a, 1, false)?;
    let b = ring_launch(h, &owned, &unread_b, 1, true)?;
    if !bits_eq(&a.g, &b.g) || !bits_eq(&a.state, &b.state) {
        return Err(format!("position 1 read unread slot {unread}"));
    }
    let mut pred = zeros.clone();
    pred[..ch].fill(1e6);
    let at_two = ring_launch(h, &owned, &pred, 2, false)?;
    let at_two_zero = ring_launch(h, &owned, &zeros, 2, true)?;
    if bits_eq(&at_two.g, &at_two_zero.g) {
        return Err("position 2 ignored the distance-2 ring slot".to_string());
    }
    Ok(Verdict::Noted(
        "position 0 reads no ring slot, positions 1 and 2 read the live slot, and one slot is written"
            .to_string(),
    ))
}

fn head_input(g: &[f32], partials: &[f32], eps: f32) -> Vec<f64> {
    let scales: Vec<f64> = (0..VALUE_HEADS)
        .map(|head| {
            let row: Vec<f64> = partials[head * PARTIALS_PER_HEAD..(head + 1) * PARTIALS_PER_HEAD]
                .iter()
                .map(|&v| f64::from(v))
                .collect();
            oracle::head_scale(&row, HEAD_COLS, f64::from(eps))
        })
        .collect();
    g.iter()
        .enumerate()
        .map(|(c, &v)| f64::from(v) * scales[c / HEAD_COLS])
        .collect()
}

/// The partials and the numerator match the oracle, and the head-norm launch accepts them.
pub fn delta_partial(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let fixed = fixed(fixture)?;
    let mut rng = oracle::Rng::new(0xD41B);
    let owned = inputs(&mut rng, 16, &fixed, &fixed.conv_w)?;
    let pos = [4u32];
    let lane = [0u32];
    let (back, fit) = launch_fit(
        h,
        &owned.as_in(view_of(HeadMap::Tiled, 1, 0, &lane, &pos, &owned)),
        true,
    )?;
    let rows = 32;
    let cols = HEAD_COLS * VALUE_HEADS;
    let weights = fixture.rows("blk.0.ssm_out.weight", GgmlType::Q4_K, cols, rows)?;
    let y0 = draw(&mut rng, rows, 0.1);
    h.clear_fault()?;
    let got = h.headnorm(&HeadNormLaunch {
        rows,
        cols,
        weights,
        g: &back.g,
        partials: &back.partials,
        y0: &y0,
        eps: owned.eps,
        layer: LAYER,
    })?;
    expect_clean(h, "the head norm")?;
    let x = head_input(&back.g, &back.partials, owned.eps);
    let reference = oracle::reference(Quant::Q4K, weights, rows, cols, &x)?;
    let kind = Kind::HeadNorm(oracle::HeadGeometry {
        heads: VALUE_HEADS,
        cols_per_head: HEAD_COLS,
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
    let mut state = owned.state.clone();
    let at = state_index(0, 0, 0, 12);
    state[at] = f32::NAN;
    let pos = [4u32];
    let lane = [0u32];
    let back = run(
        h,
        &owned.as_in(View {
            map: HeadMap::Tiled,
            lanes: 1,
            lane_at: 0,
            lane: &lane,
            pos: &pos,
            state: &state,
            ring: &owned.ring,
        }),
    )?;
    expect_fault(h, "a non-finite state lane", FaultSite::DeltaLane)?;
    for j in 0..8 {
        for i in 0..16 {
            let key = key_of(j, i);
            let idx = state_index(0, 0, 0, key);
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
        n_k: owned.n_k,
        map: HeadMap::Tiled,
        lane: 0,
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
            let idx = state_index(0, 0, 0, key_of(j, i));
            got_state[idx] = clean.state[idx] as f32;
        }
    }
    oracle_delta::fit(&g, &partials, &got_state, &back.ring, &clean, true)?;
    Ok(())
}

/// A lane word past the lanes, and one non-finite key, each raise the lane site.
pub fn delta_nan(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let fixed = fixed(fixture)?;
    let mut rng = oracle::Rng::new(0xD41C);
    let owned = inputs(&mut rng, 16, &fixed, &fixed.conv_w)?;
    let pos = [4u32];
    let lane = [4u32];
    let back = run(
        h,
        &owned.as_in(view_of(HeadMap::Tiled, 1, 0, &lane, &pos, &owned)),
    )?;
    expect_fault(h, "a lane word past the lanes", FaultSite::DeltaLane)?;
    if !bits_eq(&back.state, &owned.state) {
        return Err("a lane word past the lanes wrote the state".to_string());
    }
    if back.g.iter().any(|v| !v.is_nan()) || back.partials.iter().any(|v| !v.is_nan()) {
        return Err("a lane word past the lanes left a finite numerator or partial".to_string());
    }
    written_once(&back.ring, &owned.ring, &owned.x, 4)?;
    poison_column(h, &owned)?;
    Ok(Verdict::Noted(
        "a lane word past the lanes left the state untouched, and one non-finite key wrote NaN on that lane only"
            .to_string(),
    ))
}

struct Refusal {
    n_v: usize,
    head: usize,
    taps: usize,
    n_k: usize,
    pos: usize,
    g: usize,
    needle: &'static str,
}

/// A shape, a position window and a numerator window the launch refuses by name.
pub fn delta_bounds(h: &Harness<'_>) -> Clause {
    let device = h.device;
    let stream = h.stream;
    let n_k = 16usize;
    let ch = elems(channels(n_k), "channels")?;
    let state_len = elems(state_elems(1), "state")?;
    let ring_len = elems(ring_elems(n_k), "ring")?;
    let g_len = VALUE_HEADS * HEAD_COLS;
    let p_len = VALUE_HEADS * PARTIALS_PER_HEAD;
    let x = device.upload(stream, &vec![1f32; ch]).map_err(why)?;
    let conv_w = device
        .upload(stream, &vec![1f32; CONV_TAPS * ch])
        .map_err(why)?;
    let b = device.upload(stream, &[1f32; VALUE_HEADS]).map_err(why)?;
    let a = device.upload(stream, &[1f32; VALUE_HEADS]).map_err(why)?;
    let dt = device.upload(stream, &[1f32; VALUE_HEADS]).map_err(why)?;
    let ssm_a = device.upload(stream, &[1f32; VALUE_HEADS]).map_err(why)?;
    let z = device.upload(stream, &vec![1f32; g_len]).map_err(why)?;
    let norm_w = device.upload(stream, &vec![1f32; HEAD_COLS]).map_err(why)?;
    let lane = device.upload(stream, &[0u32]).map_err(why)?;
    let mut state = device.upload(stream, &vec![1f32; state_len]).map_err(why)?;
    let mut ring = device.upload(stream, &vec![1f32; ring_len]).map_err(why)?;
    let mut partials = device.upload(stream, &vec![1f32; p_len]).map_err(why)?;
    let mut g = device.upload(stream, &vec![1f32; g_len + 1]).map_err(why)?;
    let whole = Refusal {
        n_v: VALUE_HEADS,
        head: HEAD_COLS,
        taps: CONV_TAPS,
        n_k,
        pos: 1,
        g: g_len,
        needle: "",
    };
    let cases = [
        Refusal {
            n_v: 16,
            needle: "n_v 32",
            ..whole
        },
        Refusal {
            head: 64,
            needle: "head 128",
            ..whole
        },
        Refusal {
            taps: 3,
            needle: "taps",
            ..whole
        },
        Refusal {
            n_k: 3,
            needle: "n_k",
            ..whole
        },
        Refusal {
            pos: PASS_ROWS + 1,
            needle: "past the ring",
            ..whole
        },
        Refusal {
            pos: 2,
            needle: "position window",
            ..whole
        },
        Refusal {
            g: g_len - 1,
            needle: "g window",
            ..whole
        },
        Refusal {
            g: g_len + 1,
            needle: "g window",
            ..whole
        },
    ];
    let count = cases.len();
    let pos_long = [0u32; PASS_ROWS + 1];
    let pos_two = [0u32, 1];
    let pos_one = [4u32];
    for case in cases {
        let pos: &[u32] = if case.pos == PASS_ROWS + 1 {
            &pos_long
        } else if case.pos == 2 {
            &pos_two
        } else {
            &pos_one
        };
        let refused = h.encode(|encoder| {
            h.kernels.delta(
                encoder,
                bloomery_metal::kernels::DeltaParams {
                    x: x.window(Elems(0), Elems(ch))?,
                    conv_w: conv_w.window(Elems(0), Elems(CONV_TAPS * ch))?,
                    b_raw: b.window(Elems(0), Elems(VALUE_HEADS))?,
                    a_raw: a.window(Elems(0), Elems(VALUE_HEADS))?,
                    dt_bias: dt.window(Elems(0), Elems(VALUE_HEADS))?,
                    ssm_a: ssm_a.window(Elems(0), Elems(VALUE_HEADS))?,
                    z: z.window(Elems(0), Elems(g_len))?,
                    norm_w: norm_w.window(Elems(0), Elems(HEAD_COLS))?,
                    lane: lane.window(Elems(0), Elems(1))?,
                    state: state.window_mut(Elems(0), Elems(state_len))?,
                    ring: ring.window_mut(Elems(0), Elems(ring_len))?,
                    partials: partials.window_mut(Elems(0), Elems(p_len))?,
                    g: g.window_mut(Elems(0), Elems(case.g))?,
                    faults: device.fault_word(),
                    n_v: case.n_v,
                    head: case.head,
                    taps: case.taps,
                    n_k: case.n_k,
                    map: HeadMap::Tiled,
                    lanes: 1,
                    lane_at: 0,
                    layer: LAYER,
                    pos,
                    eps: 1e-6,
                },
            )
        });
        match refused {
            Ok(()) => return Err(format!("{} was accepted", case.needle)),
            Err(text) if !text.contains(case.needle) => {
                return Err(format!("a refusal did not name {}: {text}", case.needle));
            }
            Err(_) => {}
        }
    }
    let mut g_back = vec![0f32; g_len + 1];
    let mut state_back = vec![0f32; state_len];
    let mut partials_back = vec![0f32; p_len];
    let mut ring_back = vec![0f32; ring_len];
    device
        .read(stream, g.whole().map_err(why)?, &mut g_back)
        .map_err(why)?;
    device
        .read(stream, state.whole().map_err(why)?, &mut state_back)
        .map_err(why)?;
    device
        .read(stream, partials.whole().map_err(why)?, &mut partials_back)
        .map_err(why)?;
    device
        .read(stream, ring.whole().map_err(why)?, &mut ring_back)
        .map_err(why)?;
    if g_back.iter().any(|&v| v != 1.0)
        || state_back.iter().any(|&v| v != 1.0)
        || partials_back.iter().any(|&v| v != 1.0)
        || ring_back.iter().any(|&v| v != 1.0)
    {
        return Err("a refused launch wrote".to_string());
    }
    Ok(Verdict::Noted(format!(
        "{count} refusals, each by name, nothing written"
    )))
}
