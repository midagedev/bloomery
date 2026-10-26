//! The Mamba-2 decode kernels' clauses (`kernels/mamba2.metal`): the step and
//! the grouped norm at the real files' geometries (Nano 96 heads, 8 groups,
//! head width 80; Lightning 64, 8, 64) and a tail instance (head width 72,
//! four chunks of 16 and one of 8), against the f64 oracle of
//! `oracle_mamba2.rs`.
//!
//! A chain clause feeds 41 consecutive positions, from the zero-filled ring
//! (the buffer starts with values in every slot) through the ring's wrap at
//! 11 to position 40, with the state in two lanes that alternate: the step at
//! position `p` reads lane `p mod 2` and writes the other. Every token checks
//! the ring bit for bit, the lane read bit for bit, the written state, `y`, the
//! partial sums and the norm inside their bands, and the carried state against
//! the free-running recurrence. The other clauses hold the `dt` span, the lane
//! contract, the faults, the norm's faults and the launch refusals.
//!
//! Every input and output window sits between words of a NaN sentinel, so a
//! read past a window poisons its result and a write past one is seen.

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Device, DeviceError, Elems, Window, WindowMut};
use bloomery_metal::MetalBuffer;
use bloomery_metal::device::MetalDevice;
use bloomery_metal::encode::ComputeEncoder;
use bloomery_metal::kernels::{
    MAMBA2_CH_PER_BLOCK, MAMBA2_STATE, MAMBA2_TAPS, Mamba2NormParams, Mamba2Params, Mamba2Shape,
    RING_ROWS,
};

use model::mamba2 as cpu;

use super::mutants::Mutant;
use super::oracle::Rng;
use super::oracle_mamba2::{self as oracle, Carry, StepIn, Weights};
use super::{ClauseRun, Harness};

/// The layer the clauses launch in.
const LAYER: u32 = 5;

/// Words of sentinel either side of a window.
const GUARD: usize = 7;

/// A quiet NaN: a word that was never written is not a number either.
const SENTINEL: f32 = f32::from_bits(0x7FC0_DEAD);

/// A lane word no state has.
const NO_LANE: u32 = u32::MAX;

/// A state band wider than this fraction of the largest term of the step's
/// updates would hide a wrong lane or a wrong decay.
const TRIPWIRE: f64 = 1e-3;

/// The real file's `A` span, from the smallest to the largest magnitude.
const A_MIN: f64 = 8.4e-6;
const A_MAX: f64 = 2714.0;

/// The `dt` values of the span clause: both softplus branches, the branch
/// edge, and a value whose `exp` overflows.
const DT_SPAN: [f32; 6] = [-30.0, 0.0, 19.9, 20.0, 25.0, 100.0];

/// A geometry the chain clauses run.
struct Spec {
    n_head: usize,
    n_group: usize,
    head_dim: usize,
    tokens: usize,
    seed: u64,
    dt: Dt,
}

#[derive(Clone, Copy)]
enum Dt {
    /// Drawn each token.
    Random,
    /// The span, a value a head, every token.
    Span,
}

const NANO: Spec = Spec {
    n_head: 96,
    n_group: 8,
    head_dim: 80,
    tokens: 41,
    seed: 0x4E41,
    dt: Dt::Random,
};

const LIGHTNING: Spec = Spec {
    n_head: 64,
    n_group: 8,
    head_dim: 64,
    tokens: 41,
    seed: 0x4C54,
    dt: Dt::Random,
};

const TAIL: Spec = Spec {
    n_head: 24,
    n_group: 4,
    head_dim: 72,
    tokens: 41,
    seed: 0x5441,
    dt: Dt::Random,
};

const SPAN: Spec = Spec {
    n_head: 12,
    n_group: 4,
    head_dim: 40,
    tokens: 6,
    seed: 0x4454,
    dt: Dt::Span,
};

/// The small geometry of the lane, fault and refusal clauses: 12 heads in 4
/// groups, head width 40 (two chunks of 16 and one of 8).
const SMALL: Spec = Spec {
    n_head: 12,
    n_group: 4,
    head_dim: 40,
    tokens: 1,
    seed: 0x534D,
    dt: Dt::Random,
};

/// The clauses on the Mamba-2 kernels.
pub fn table<'a>(h: &'a Harness<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    vec![
        ("mamba2_nano", Box::new(move || chain(h, &NANO))),
        ("mamba2_ltn", Box::new(move || chain(h, &LIGHTNING))),
        ("mamba2_tail", Box::new(move || chain(h, &TAIL))),
        ("mamba2_dt", Box::new(move || chain(h, &SPAN))),
        ("mamba2_lanes", Box::new(move || lanes(h))),
        ("mamba2_faults", Box::new(move || faults(h))),
        ("mamba2_norm_faults", Box::new(move || norm_faults(h))),
        ("mamba2_refusals", Box::new(move || refusals(h))),
        ("mamba2_cpu_tie", Box::new(cpu_tie)),
    ]
}

/// The kernel defects the clauses must catch: `--mutant NAME` runs the clause
/// named here on kernels compiled from the edited source.
pub const MUTANTS: &[Mutant] = &[
    Mutant {
        name: "mamba2_dt_branch",
        clause: "mamba2_dt",
        why: "the dt > 20 branch is removed: log1p(exp(100)) overflows to NaN",
        edits: &[(
            "dtv > 20.0f ? dtv : mamba2_log1p(exp(dtv))",
            "mamba2_log1p(exp(dtv))",
            1,
        )],
    },
    Mutant {
        name: "mamba2_ring_slot",
        clause: "mamba2_nano",
        why: "the step writes ring slot (p + 1) mod 11, not p mod 11",
        edits: &[(
            "const uint ring_slot = pos % BLOOMERY_DELTA_RING_ROWS;",
            "const uint ring_slot = (pos + 1u) % BLOOMERY_DELTA_RING_ROWS;",
            1,
        )],
    },
    Mutant {
        name: "mamba2_zero_fill",
        clause: "mamba2_nano",
        why: "a predecessor before the first position is read from the ring, not 0",
        edits: &[("if (pos < back) {", "if (false) {", 1)],
    },
    Mutant {
        name: "mamba2_group_map",
        clause: "mamba2_nano",
        why: "B and C come from group head % n_group, not head / (n_head / n_group)",
        edits: &[(
            "const uint g = h / per;",
            "const uint g = h % args.n_group;",
            1,
        )],
    },
    Mutant {
        name: "mamba2_norm_width",
        clause: "mamba2_nano",
        why: "the norm's mean runs over d_inner, not d_inner / n_group",
        edits: &[
            (
                "const uint stat_heads = per;",
                "const uint stat_heads = args.n_head;",
                1,
            ),
            (
                "const uint stat_first = g * stat_heads * args.chunks;",
                "const uint stat_first = 0u;",
                1,
            ),
        ],
    },
    Mutant {
        name: "mamba2_lane_swap",
        clause: "mamba2_nano",
        why: "the state is read from the lane the step writes",
        edits: &[(
            "const uint read_word = lane_buf[args.lane_at];",
            "const uint read_word = lane_buf[args.lane_to];",
            1,
        )],
    },
    Mutant {
        name: "mamba2_skip_after_gate",
        clause: "mamba2_nano",
        why: "the D skip is added after the z gate",
        edits: &[(
            "mamba2_silu(z[at]) * (acc + x * d_h)",
            "mamba2_silu(z[at]) * acc + x * d_h",
            1,
        )],
    },
    Mutant {
        name: "mamba2_decay_first_order",
        clause: "mamba2_nano",
        why: "the decay is 1 + sigma A, not exp(sigma A)",
        edits: &[(
            "const float decay = exp(sigma * a_h);",
            "const float decay = 1.0f + sigma * a_h;",
            1,
        )],
    },
    Mutant {
        name: "mamba2_tail_live",
        clause: "mamba2_tail",
        why: "every channel of the last chunk is live, past the head's width",
        edits: &[(
            "return chunk * MAMBA2_CH_PER_BLOCK + k < hd;",
            "return true;",
            1,
        )],
    },
    Mutant {
        name: "mamba2_write_lane",
        clause: "mamba2_lanes",
        why: "the state is written to the lane it was read from",
        edits: &[(
            "const uint base = (write_word * inner + at) * MAMBA2_STATE + lane;",
            "const uint base = (read_word * inner + at) * MAMBA2_STATE + lane;",
            1,
        )],
    },
    Mutant {
        name: "mamba2_lane_range",
        clause: "mamba2_lanes",
        why: "a lane word at or past the lanes is used",
        edits: &[(
            "if (read_word >= args.lanes || write_word >= args.lanes) {",
            "if (false) {",
            1,
        )],
    },
    Mutant {
        name: "mamba2_a_finite",
        clause: "mamba2_faults",
        why: "a non-finite A passes the scalar check: -inf decays to a finite y",
        edits: &[("isfinite(a_h) && ", "", 1)],
    },
    Mutant {
        name: "mamba2_norm_finite",
        clause: "mamba2_norm_faults",
        why: "a normalized value that is not finite is written without a raise",
        edits: &[("bad = bad || !isfinite(o);", "", 1)],
    },
    Mutant {
        name: "mamba2_norm_usable",
        clause: "mamba2_norm_faults",
        why: "a group whose mean square is not finite is normalized by a zero scale",
        edits: &[(
            "usable_tg[0] = rms_usable(total, scale) ? 1u : 0u;",
            "usable_tg[0] = 1u;",
            1,
        )],
    },
];

/// A window of f32s with words of sentinel either side.
struct Buf {
    buf: MetalBuffer<f32>,
    before: usize,
    len: usize,
}

impl Buf {
    fn new(h: &Harness<'_>, body: &[f32], before: usize, after: usize) -> Result<Self, String> {
        let mut words = vec![SENTINEL; before];
        words.extend_from_slice(body);
        words.extend(std::iter::repeat_n(SENTINEL, after));
        let buf = h.device.upload(h.stream, &words).map_err(why)?;
        Ok(Self {
            buf,
            before,
            len: body.len(),
        })
    }

    /// A buffer of sentinel words only, `len` of them in the window.
    fn blank(h: &Harness<'_>, len: usize, after: usize) -> Result<Self, String> {
        Self::new(h, &vec![SENTINEL; len], GUARD, after)
    }

    /// An input: guards either side.
    fn input(h: &Harness<'_>, body: &[f32]) -> Result<Self, String> {
        Self::new(h, body, GUARD, GUARD)
    }

    fn win_of(&self, len: usize) -> Result<Window<'_, f32, MetalDevice>, String> {
        self.buf.window(Elems(self.before), Elems(len)).map_err(why)
    }

    fn win(&self) -> Result<Window<'_, f32, MetalDevice>, String> {
        self.win_of(self.len)
    }

    fn win_mut_of(&mut self, len: usize) -> Result<WindowMut<'_, f32, MetalDevice>, String> {
        self.buf
            .window_mut(Elems(self.before), Elems(len))
            .map_err(why)
    }

    fn win_mut(&mut self) -> Result<WindowMut<'_, f32, MetalDevice>, String> {
        let len = self.len;
        self.win_mut_of(len)
    }

    /// The window's words, after checking that the sentinel words around it
    /// are as they were.
    fn read(&self, h: &Harness<'_>) -> Result<Vec<f32>, String> {
        let total = self.buf.len().get();
        let mut back = vec![0f32; total];
        h.device
            .read(h.stream, self.buf.whole().map_err(why)?, &mut back)
            .map_err(why)?;
        let after = self.before + self.len;
        for (i, v) in back.iter().enumerate() {
            let outside = i < self.before || i >= after;
            if outside && v.to_bits() != SENTINEL.to_bits() {
                return Err(format!(
                    "a word {} the window was written: {v:?}",
                    if i < self.before { "before" } else { "past" }
                ));
            }
        }
        Ok(back[self.before..after].to_vec())
    }
}

/// Values `scale · [-1, 1)`.
fn draw(rng: &mut Rng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| scale * rng.signed()).collect()
}

/// The layer's weights on the host.
struct Host {
    conv_w: Vec<f32>,
    conv_b: Vec<f32>,
    dt_bias: Vec<f32>,
    a: Vec<f32>,
    d: Vec<f32>,
    gain: Vec<f32>,
    eps: f32,
}

impl Host {
    /// `A` runs over the real file's span, from -8.4e-6 to -2714 in log steps,
    /// the heads in a fixed order that puts neighbours far apart.
    fn new(rng: &mut Rng, shape: Mamba2Shape, dt: Dt) -> Self {
        let span = (A_MAX / A_MIN).ln();
        let n = shape.n_head;
        let a = (0..n)
            .map(|h| {
                let t = ((h * 7) % n) as f64 / (n.max(2) - 1) as f64;
                -((A_MIN.ln() + t * span).exp() as f32)
            })
            .collect();
        let dt_bias = match dt {
            Dt::Random => draw(rng, n, 0.5),
            Dt::Span => vec![0.0; n],
        };
        Self {
            conv_w: draw(rng, MAMBA2_TAPS * shape.conv_dim, 0.5),
            conv_b: draw(rng, shape.conv_dim, 0.1),
            dt_bias,
            a,
            d: draw(rng, n, 0.5).into_iter().map(|v| 1.0 + v).collect(),
            gain: draw(rng, shape.inner, 0.3)
                .into_iter()
                .map(|v| 1.0 + v)
                .collect(),
            eps: 1e-5,
        }
    }

    fn weights(&self) -> Weights<'_> {
        Weights {
            conv_w: &self.conv_w,
            conv_b: &self.conv_b,
            dt_bias: &self.dt_bias,
            a: &self.a,
            d: &self.d,
            gain: &self.gain,
            eps: self.eps,
        }
    }
}

/// One token's inputs.
struct Token {
    pos: u32,
    xbc: Vec<f32>,
    dt: Vec<f32>,
    z: Vec<f32>,
}

impl Token {
    fn new(rng: &mut Rng, shape: Mamba2Shape, pos: u32, dt: Dt) -> Self {
        let dt = match dt {
            Dt::Random => draw(rng, shape.n_head, 2.0),
            Dt::Span => (0..shape.n_head).map(|h| DT_SPAN[h % 6]).collect(),
        };
        Self {
            pos,
            xbc: draw(rng, shape.conv_dim, 1.0),
            dt,
            z: draw(rng, shape.inner, 2.0),
        }
    }
}

/// The layer on the device: weights, and the state, ring and outputs that
/// persist across tokens.
struct Rig<'a> {
    h: &'a Harness<'a>,
    shape: Mamba2Shape,
    lanes: usize,
    conv_w: Buf,
    conv_b: Buf,
    dt_bias: Buf,
    a: Buf,
    d: Buf,
    gain: Buf,
    state: Buf,
    ring: Buf,
    y: Buf,
    part: Buf,
    out: Buf,
    /// What the state buffer holds, every lane.
    state_host: Vec<f32>,
    /// What the ring holds.
    ring_host: Vec<f32>,
}

/// What a step left in the buffers.
struct Back {
    state: Vec<f32>,
    ring: Vec<f32>,
    y: Vec<f32>,
    part: Vec<f32>,
    out: Vec<f32>,
}

impl<'a> Rig<'a> {
    /// The rig with `lanes` lanes of state and a ring, each holding values
    /// (so that a read of a lane or a slot that the rule says is not read
    /// shows).
    fn new(
        h: &'a Harness<'a>,
        shape: Mamba2Shape,
        lanes: usize,
        host: &Host,
        rng: &mut Rng,
    ) -> Result<Self, String> {
        let state_host = draw(rng, lanes * shape.inner * MAMBA2_STATE, 0.3);
        let ring_host = draw(rng, shape.ring_elems(), 0.5);
        // A state read past the lanes, and the dead channels of the last
        // chunk, land in the words after them.
        let tail = shape.inner * MAMBA2_STATE + MAMBA2_CH_PER_BLOCK * MAMBA2_STATE;
        Ok(Self {
            h,
            shape,
            lanes,
            conv_w: Buf::input(h, &host.conv_w)?,
            conv_b: Buf::input(h, &host.conv_b)?,
            dt_bias: Buf::input(h, &host.dt_bias)?,
            a: Buf::input(h, &host.a)?,
            d: Buf::input(h, &host.d)?,
            gain: Buf::input(h, &host.gain)?,
            state: Buf::new(h, &state_host, GUARD, tail)?,
            ring: Buf::new(h, &ring_host, GUARD, GUARD)?,
            y: Buf::blank(h, shape.inner, GUARD)?,
            part: Buf::blank(h, shape.partials(), GUARD)?,
            out: Buf::blank(h, shape.inner, GUARD)?,
            state_host,
            ring_host,
        })
    }

    /// The step of `tok`, one launch, with the fault word cleared first. The
    /// lane words are `words`, and the step reads the words at `lane_at` and
    /// `lane_to`.
    fn step(
        &mut self,
        tok: &Token,
        words: &[u32],
        (lane_at, lane_to): (usize, usize),
    ) -> Result<(), String> {
        let h = self.h;
        let xbc = Buf::input(h, &tok.xbc)?;
        let dt = Buf::input(h, &tok.dt)?;
        let z = Buf::input(h, &tok.z)?;
        let lane = h.device.upload(h.stream, words).map_err(why)?;
        let pos = [tok.pos];
        let shape = self.shape;
        h.clear_fault()?;
        let params = Mamba2Params {
            xbc: xbc.win()?,
            dt: dt.win()?,
            dt_bias: self.dt_bias.win()?,
            z: z.win()?,
            a: self.a.win()?,
            d: self.d.win()?,
            conv_w: self.conv_w.win()?,
            conv_b: self.conv_b.win()?,
            lane: lane.window(Elems(0), Elems(words.len())).map_err(why)?,
            state: self
                .state
                .win_mut_of(self.lanes * shape.inner * MAMBA2_STATE)?,
            ring: self.ring.win_mut()?,
            y: self.y.win_mut()?,
            part: self.part.win_mut()?,
            faults: h.device.fault_word(),
            n_head: shape.n_head,
            n_group: shape.n_group,
            head_dim: shape.head_dim,
            d_state: MAMBA2_STATE,
            taps: MAMBA2_TAPS,
            lanes: self.lanes,
            lane_at,
            lane_to,
            layer: LAYER,
            pos: &pos,
        };
        h.encode(|encoder| h.kernels.mamba2_step(encoder, params))
    }

    /// The norm over what the step wrote.
    fn norm(&mut self, eps: f32) -> Result<(), String> {
        let h = self.h;
        let shape = self.shape;
        let params = Mamba2NormParams {
            y: self.y.win()?,
            part: self.part.win()?,
            gain: self.gain.win()?,
            out: self.out.win_mut()?,
            faults: h.device.fault_word(),
            n_head: shape.n_head,
            n_group: shape.n_group,
            head_dim: shape.head_dim,
            layer: LAYER,
            eps,
        };
        h.encode(|encoder| h.kernels.mamba2_norm(encoder, params))
    }

    /// Every buffer the step writes, the guards checked.
    fn read(&self) -> Result<Back, String> {
        let h = self.h;
        let state = self.state.read(h)?;
        let ring = self.ring.read(h)?;
        Ok(Back {
            state: state[..self.lanes * self.shape.inner * MAMBA2_STATE].to_vec(),
            ring,
            y: self.y.read(h)?,
            part: self.part.read(h)?,
            out: self.out.read(h)?,
        })
    }

    /// The words of lane `lane` of a state buffer's body.
    fn lane_of<'s>(&self, body: &'s [f32], lane: usize) -> &'s [f32] {
        let n = self.shape.inner * MAMBA2_STATE;
        &body[lane * n..(lane + 1) * n]
    }
}

/// The fault word names `site` in [`LAYER`], and only that site.
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

/// The fault word is clean.
fn expect_clean(h: &Harness<'_>, what: &str) -> Result<(), String> {
    match h.fault()? {
        Some(fault) => Err(format!("{what} raised {fault}")),
        None => Ok(()),
    }
}

/// The first word where `got` and `want` differ in bits.
fn first_diff(got: &[f32], want: &[f32]) -> Option<(usize, f32, f32)> {
    got.iter()
        .zip(want)
        .enumerate()
        .find_map(|(i, (&g, &w))| (g.to_bits() != w.to_bits()).then_some((i, g, w)))
}

/// `got` holds the bits of `want`.
fn same_bits(what: &str, got: &[f32], want: &[f32]) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!(
            "{what}: {} words, expected {}",
            got.len(),
            want.len()
        ));
    }
    match first_diff(got, want) {
        None => Ok(()),
        Some((i, g, w)) => Err(format!("{what}[{i}] is {g:e}, expected {w:e} bit for bit")),
    }
}

/// The worst error over its band, per output.
#[derive(Default)]
struct Fit {
    state: f64,
    y: f64,
    part: f64,
    norm: f64,
    carried: f64,
    tightest: f64,
}

impl Fit {
    fn keep(&mut self, other: &Fit) {
        self.state = self.state.max(other.state);
        self.y = self.y.max(other.y);
        self.part = self.part.max(other.part);
        self.norm = self.norm.max(other.norm);
        self.carried = self.carried.max(other.carried);
        self.tightest = self.tightest.max(other.tightest);
    }

    fn note(&self) -> String {
        format!(
            "worst error over its band: state {:.3}, y {:.3}, partials {:.3}, norm {:.3}, carried {:.3}; widest state band over the largest term {:.1e}",
            self.state, self.y, self.part, self.norm, self.carried, self.tightest
        )
    }
}

/// One step on `rig`, from the lane `read` to the lane `write` (the lane words
/// `[NO_LANE, read, write]`, read at index 1 and written at index 2), then the
/// norm, with everything the step and the norm wrote held to the oracle.
fn token(
    rig: &mut Rig<'_>,
    host: &Host,
    tok: &Token,
    (read, write): (usize, usize),
    carry: &mut Carry,
    history: &[Vec<f32>],
) -> Result<Fit, String> {
    let h = rig.h;
    let shape = rig.shape;
    let words = [NO_LANE, read as u32, write as u32];
    rig.step(tok, &words, (1, 2))?;
    expect_clean(h, "the step")?;
    rig.norm(host.eps)?;
    expect_clean(h, "the norm")?;
    let back = rig.read()?;

    // Every lane but the written one is as it was; the lane read is one of
    // them unless the step is in place.
    for lane in 0..rig.lanes {
        if lane != write {
            same_bits(
                &format!("lane {lane}"),
                rig.lane_of(&back.state, lane),
                rig.lane_of(&rig.state_host, lane),
            )?;
        }
    }
    let w = host.weights();
    let inp = StepIn {
        shape,
        w: &w,
        pos: tok.pos,
        xbc: &tok.xbc,
        dt: &tok.dt,
        z: &tok.z,
        state: rig.lane_of(&rig.state_host, read),
        ring: &rig.ring_host,
    };
    let want = oracle::step(&inp)?;
    let got_state = rig.lane_of(&back.state, write);
    let mut fit = Fit {
        tightest: want.tightest,
        ..Fit::default()
    };
    if want.tightest > TRIPWIRE {
        return Err(format!(
            "the widest state band is {:e} of the largest update term, over {TRIPWIRE:e}, which would hide a wrong decay or lane",
            want.tightest
        ));
    }
    fit.state = oracle::scan(
        got_state,
        &want.state,
        &want.state_err,
        oracle::STEP_FLOOR,
        "state",
    )?;
    same_bits("ring", &back.ring, &want.ring)?;
    let (yv, ye) = oracle::y_gated(&inp, &want, got_state)?;
    fit.y = oracle::scan(&back.y, &yv, &ye, oracle::STEP_FLOOR, "y")?;
    let (pv, pe) = oracle::partials(shape, &back.y);
    fit.part = oracle::scan(&back.part, &pv, &pe, oracle::STEP_FLOOR, "partials")?;
    let (nv, ne) = oracle::norm(shape, &w, &back.y, &back.part)?;
    fit.norm = oracle::scan(&back.out, &nv, &ne, oracle::STEP_FLOOR, "norm")?;
    carry.step(&inp, history, &want.state_err);
    fit.carried = carry.fit(got_state)?;
    rig.state_host = back.state;
    rig.ring_host = back.ring;
    Ok(fit)
}

/// `spec.tokens` consecutive positions from 0, the state in two lanes that
/// alternate.
fn chain(h: &Harness<'_>, spec: &Spec) -> Clause {
    let shape =
        Mamba2Shape::of("mamba2 step", spec.n_head, spec.n_group, spec.head_dim).map_err(why)?;
    let mut rng = Rng::new(spec.seed);
    let host = Host::new(&mut rng, shape, spec.dt);
    let mut rig = Rig::new(h, shape, 2, &host, &mut rng)?;
    let mut carry = Carry::new(rig.lane_of(&rig.state_host, 0));
    let mut history: Vec<Vec<f32>> = Vec::new();
    let mut worst = Fit::default();
    for t in 0..spec.tokens {
        let tok = Token::new(&mut rng, shape, t as u32, spec.dt);
        history.push(tok.xbc.clone());
        let fit = token(
            &mut rig,
            &host,
            &tok,
            (t % 2, (t + 1) % 2),
            &mut carry,
            &history,
        )
        .map_err(|e| format!("position {t}: {e}"))?;
        worst.keep(&fit);
    }
    Ok(Verdict::Noted(format!(
        "{} heads, {} groups, head width {}, {} positions; {}",
        spec.n_head,
        spec.n_group,
        spec.head_dim,
        spec.tokens,
        worst.note()
    )))
}

/// The small rig, one token drawn.
fn small<'a>(h: &'a Harness<'a>, lanes: usize) -> Result<(Rig<'a>, Host, Token, Rng), String> {
    let shape =
        Mamba2Shape::of("mamba2 step", SMALL.n_head, SMALL.n_group, SMALL.head_dim).map_err(why)?;
    let mut rng = Rng::new(SMALL.seed);
    let host = Host::new(&mut rng, shape, SMALL.dt);
    let rig = Rig::new(h, shape, lanes, &host, &mut rng)?;
    // Past the ring's zero-fill, so the conv reads the ring.
    let tok = Token::new(&mut rng, shape, 6, SMALL.dt);
    Ok((rig, host, tok, rng))
}

/// The lane contract: the step reads the lane the word at `lane_at` names and
/// writes the one at `lane_to` names, leaving the others, whichever word
/// holds it; a word at or past the lanes raises the lane site and writes
/// nothing.
fn lanes(h: &Harness<'_>) -> Clause {
    let (mut rig, host, tok, _) = small(h, 3)?;
    let w = host.weights();
    let shape = rig.shape;
    let lane_state = |rig: &Rig<'_>, lane: usize| rig.lane_of(&rig.state_host, lane).to_vec();

    // Two lanes apart: read lane 2, write lane 0, lane 1 untouched; word 0
    // names no lane, so a read of another index raises.
    let (read, write) = (2usize, 0usize);
    let words = [NO_LANE, read as u32, write as u32, 1];
    rig.step(&tok, &words, (1, 2))?;
    expect_clean(h, "the step across lanes")?;
    let back = rig.read()?;
    for lane in [1usize, 2] {
        same_bits(
            &format!("lane {lane}"),
            rig.lane_of(&back.state, lane),
            rig.lane_of(&rig.state_host, lane),
        )?;
    }
    let inp = StepIn {
        shape,
        w: &w,
        pos: tok.pos,
        xbc: &tok.xbc,
        dt: &tok.dt,
        z: &tok.z,
        state: rig.lane_of(&rig.state_host, read),
        ring: &rig.ring_host,
    };
    let want = oracle::step(&inp)?;
    oracle::scan(
        rig.lane_of(&back.state, write),
        &want.state,
        &want.state_err,
        oracle::STEP_FLOOR,
        "the written lane",
    )?;
    rig.state_host = back.state;
    rig.ring_host = back.ring;

    // In place: both words name lane 1.
    let before = lane_state(&rig, 1);
    let tok2 = Token {
        pos: 7,
        xbc: tok.xbc.iter().map(|v| -v).collect(),
        dt: tok.dt.clone(),
        z: tok.z.clone(),
    };
    rig.step(&tok2, &[NO_LANE, 1, 1], (1, 2))?;
    expect_clean(h, "the in-place step")?;
    let back = rig.read()?;
    let inp = StepIn {
        shape,
        w: &w,
        pos: tok2.pos,
        xbc: &tok2.xbc,
        dt: &tok2.dt,
        z: &tok2.z,
        state: &before,
        ring: &rig.ring_host,
    };
    let want = oracle::step(&inp)?;
    oracle::scan(
        rig.lane_of(&back.state, 1),
        &want.state,
        &want.state_err,
        oracle::STEP_FLOOR,
        "the in-place lane",
    )?;
    for lane in [0usize, 2] {
        same_bits(
            &format!("lane {lane} of the in-place step"),
            rig.lane_of(&back.state, lane),
            rig.lane_of(&rig.state_host, lane),
        )?;
    }
    rig.state_host = back.state;
    rig.ring_host = back.ring;

    // A word at the lane count, read and written: the lane site, nothing
    // written (the words after the last lane hold the sentinel).
    for (name, words) in [
        ("read", [NO_LANE, 3, 0]),
        ("write", [NO_LANE, 0, 3]),
        ("read and write", [NO_LANE, 3, 3]),
    ] {
        let before_state = rig.state_host.clone();
        let before_ring = rig.ring_host.clone();
        let before_y = rig.read()?.y;
        rig.step(&tok, &words, (1, 2))?;
        expect_fault(
            h,
            &format!("a {name} lane word at the lane count"),
            FaultSite::DeltaLane,
        )?;
        let back = rig.read()?;
        same_bits(
            &format!("the state after a bad {name} word"),
            &back.state,
            &before_state,
        )?;
        same_bits(
            &format!("the ring after a bad {name} word"),
            &back.ring,
            &before_ring,
        )?;
        same_bits(&format!("y after a bad {name} word"), &back.y, &before_y)?;
    }
    Ok(Verdict::Pass)
}

/// An edit of a launch's scalars: the weights or the token.
type ScalarEdit = fn(&mut Host, &mut Token);

/// An edit of a token's activations, giving the channels it reaches.
type ActEdit = fn(&Mamba2Shape, &mut Token) -> Vec<usize>;

/// The first head's channels of a state lane.
fn head_range(shape: Mamba2Shape, head: usize) -> std::ops::Range<usize> {
    let n = shape.head_dim * MAMBA2_STATE;
    head * n..(head + 1) * n
}

/// A launch whose input is not finite raises the scan site. A scalar of a
/// head that is not finite writes nothing of that head; a channel whose `y`
/// is not finite leaves its state.
fn faults(h: &Harness<'_>) -> Clause {
    // The control: the same rig and token run clean.
    {
        let (mut rig, host, tok, _) = small(h, 2)?;
        let mut carry = Carry::new(rig.lane_of(&rig.state_host, 0));
        token(
            &mut rig,
            &host,
            &tok_at0(&tok),
            (0, 1),
            &mut carry,
            std::slice::from_ref(&tok.xbc),
        )
        .map_err(|e| format!("the control: {e}"))?;
    }
    let head = 5usize;
    let scalars: [(&str, ScalarEdit); 10] = [
        ("dt NaN", |_, t| t.dt[5] = f32::NAN),
        ("dt -inf", |_, t| t.dt[5] = f32::NEG_INFINITY),
        ("dt +inf", |_, t| t.dt[5] = f32::INFINITY),
        ("bias -inf", |w, _| w.dt_bias[5] = f32::NEG_INFINITY),
        ("A NaN", |w, _| w.a[5] = f32::NAN),
        ("A -inf", |w, _| w.a[5] = f32::NEG_INFINITY),
        ("A +inf", |w, _| w.a[5] = f32::INFINITY),
        ("D NaN", |w, _| w.d[5] = f32::NAN),
        ("D +inf", |w, _| w.d[5] = f32::INFINITY),
        ("dt + bias overflow", |w, t| {
            t.dt[5] = 3.0e38;
            w.dt_bias[5] = 3.0e38;
        }),
    ];
    for (name, edit) in scalars {
        let (mut rig, mut host, mut tok, _) = small(h, 2)?;
        // The head's dt keeps its sigma above 0 so that a -inf A is not
        // 0 · -inf.
        tok.dt[head] = 0.5;
        edit(&mut host, &mut tok);
        rig = rig_with(h, rig, &host)?;
        let shape = rig.shape;
        let (before_state, before_ring) = (rig.state_host.clone(), rig.ring_host.clone());
        rig.step(&tok, &[NO_LANE, 0, 1], (1, 2))?;
        expect_fault(h, name, FaultSite::Mamba2Scan)?;
        let back = rig.read()?;
        let range = head_range(shape, head);
        let write = rig.lane_of(&back.state, 1);
        let was = rig.lane_of(&before_state, 1);
        same_bits(
            &format!("{name}: the head's state"),
            &write[range.clone()],
            &was[range],
        )?;
        let slot = tok.pos as usize % RING_ROWS;
        let channels = head * shape.head_dim..(head + 1) * shape.head_dim;
        same_bits(
            &format!("{name}: the head's ring"),
            &back.ring[slot * shape.conv_dim..][channels.clone()],
            &before_ring[slot * shape.conv_dim..][channels.clone()],
        )?;
        if back.y[channels]
            .iter()
            .any(|v| v.to_bits() != SENTINEL.to_bits())
        {
            return Err(format!("{name}: the head's y was written"));
        }
        // The other heads ran.
        let other = if head == 0 { 1 } else { 0 };
        if back.y[other * shape.head_dim].is_nan() {
            return Err(format!("{name}: head {other} did not run"));
        }
    }

    // An activation that is not finite: the channels it reaches raise and
    // keep their state; the others run.
    let acts: [(&str, ActEdit); 3] = [
        ("x NaN", |_, t| {
            t.xbc[3] = f32::NAN;
            vec![3]
        }),
        ("z NaN", |_, t| {
            t.z[7] = f32::NAN;
            vec![7]
        }),
        ("B NaN", |s, t| {
            // B of group 0, which the heads 0 to 2 read.
            t.xbc[s.inner + 9] = f32::NAN;
            (0..3 * s.head_dim).collect()
        }),
    ];
    for (name, edit) in acts {
        let (mut rig, _, mut tok, _) = small(h, 2)?;
        let shape = rig.shape;
        let hit = edit(&shape, &mut tok);
        let before_state = rig.state_host.clone();
        rig.step(&tok, &[NO_LANE, 0, 1], (1, 2))?;
        expect_fault(h, name, FaultSite::Mamba2Scan)?;
        let back = rig.read()?;
        let write = rig.lane_of(&back.state, 1);
        let was = rig.lane_of(&before_state, 1);
        for &c in &hit {
            let at = c * MAMBA2_STATE..(c + 1) * MAMBA2_STATE;
            same_bits(
                &format!("{name}: channel {c}"),
                &write[at.clone()],
                &was[at],
            )?;
            if !back.y[c].is_nan() {
                return Err(format!("{name}: y[{c}] is {:e}, not NaN", back.y[c]));
            }
        }
        // The last head's channels are none of the hit ones and moved.
        let c = shape.inner - 1;
        let at = c * MAMBA2_STATE..(c + 1) * MAMBA2_STATE;
        if first_diff(&write[at.clone()], &was[at]).is_none() {
            return Err(format!("{name}: channel {c} did not run"));
        }
    }
    Ok(Verdict::Pass)
}

/// `tok` as position 0.
fn tok_at0(tok: &Token) -> Token {
    Token {
        pos: 0,
        xbc: tok.xbc.clone(),
        dt: tok.dt.clone(),
        z: tok.z.clone(),
    }
}

/// `rig` with the weights of `host`.
fn rig_with<'a>(h: &'a Harness<'a>, old: Rig<'a>, host: &Host) -> Result<Rig<'a>, String> {
    Ok(Rig {
        dt_bias: Buf::input(h, &host.dt_bias)?,
        a: Buf::input(h, &host.a)?,
        d: Buf::input(h, &host.d)?,
        ..old
    })
}

/// One norm launch over `y` and `part`, the out words read back.
fn norm_once(
    h: &Harness<'_>,
    shape: Mamba2Shape,
    y: &[f32],
    part: &[f32],
    gain: &[f32],
    eps: f32,
) -> Result<Vec<f32>, String> {
    let y = Buf::input(h, y)?;
    let part = Buf::input(h, part)?;
    let gain = Buf::input(h, gain)?;
    let mut out = Buf::blank(h, shape.inner, GUARD)?;
    h.clear_fault()?;
    let params = Mamba2NormParams {
        y: y.win()?,
        part: part.win()?,
        gain: gain.win()?,
        out: out.win_mut()?,
        faults: h.device.fault_word(),
        n_head: shape.n_head,
        n_group: shape.n_group,
        head_dim: shape.head_dim,
        layer: LAYER,
        eps,
    };
    h.encode(|encoder| h.kernels.mamba2_norm(encoder, params))?;
    out.read(h)
}

/// The norm's faults: a group whose partials are not finite or whose mean is
/// unusable is NaN and raises the scan site, a gain that is not finite
/// raises, and the other groups are normalized.
fn norm_faults(h: &Harness<'_>) -> Clause {
    let shape =
        Mamba2Shape::of("mamba2 norm", SMALL.n_head, SMALL.n_group, SMALL.head_dim).map_err(why)?;
    let mut rng = Rng::new(SMALL.seed + 1);
    let host = Host::new(&mut rng, shape, SMALL.dt);
    let y = draw(&mut rng, shape.inner, 2.0);
    let w = host.weights();
    // The step's own partials over this y.
    let (pv, _) = oracle::partials(shape, &y);
    let part: Vec<f32> = pv.iter().map(|&v| v as f32).collect();
    let width = shape.per * shape.head_dim;
    let stat = shape.per * shape.chunks;

    let ok = norm_once(h, shape, &y, &part, &host.gain, host.eps)?;
    expect_clean(h, "the norm of finite partials")?;
    let (nv, ne) = oracle::norm(shape, &w, &y, &part)?;
    oracle::scan(&ok, &nv, &ne, oracle::STEP_FLOOR, "the control")?;

    // A partial of group 1 that is +inf: its channels NaN, the raise, and the
    // other groups as they were.
    let mut bad = part.clone();
    bad[stat + 1] = f32::INFINITY;
    let out = norm_once(h, shape, &y, &bad, &host.gain, host.eps)?;
    expect_fault(h, "an infinite partial", FaultSite::Mamba2Scan)?;
    for (i, v) in out.iter().enumerate() {
        let in_group = (width..2 * width).contains(&i);
        if in_group && !v.is_nan() {
            return Err(format!("out[{i}] of the faulted group is {v:e}, not NaN"));
        }
        if !in_group && first_diff(&[*v], &[ok[i]]).is_some() {
            return Err(format!("out[{i}] of another group moved to {v:e}"));
        }
    }

    // Zero partials with a zero epsilon: the mean is 0 and the scale is not
    // finite.
    let zero = vec![0.0f32; part.len()];
    let out = norm_once(h, shape, &y, &zero, &host.gain, 0.0)?;
    expect_fault(
        h,
        "a zero mean square without epsilon",
        FaultSite::Mamba2Scan,
    )?;
    if out.iter().any(|v| !v.is_nan()) {
        return Err("a zero mean square without epsilon wrote a value".to_string());
    }

    // A gain that is NaN: its channel raises and is NaN, the rest normalized.
    let mut gain = host.gain.clone();
    gain[width + 3] = f32::NAN;
    let out = norm_once(h, shape, &y, &part, &gain, host.eps)?;
    expect_fault(h, "a NaN gain", FaultSite::Mamba2Scan)?;
    for (i, v) in out.iter().enumerate() {
        if i == width + 3 {
            if !v.is_nan() {
                return Err(format!("out[{i}] of a NaN gain is {v:e}, not NaN"));
            }
        } else if first_diff(&[*v], &[ok[i]]).is_some() {
            return Err(format!("out[{i}] beside a NaN gain moved to {v:e}"));
        }
    }
    Ok(Verdict::Pass)
}

/// Overrides of one launch: every field that a refusal names.
#[derive(Clone, Copy)]
struct Case {
    what: &'static str,
    needle: &'static str,
    d_state: usize,
    taps: usize,
    n_head: usize,
    n_group: usize,
    head_dim: usize,
    lanes: usize,
    lane_at: usize,
    lane_to: usize,
    pos_len: usize,
    /// The window one word short.
    short: &'static str,
    /// The window one word long.
    long: &'static str,
}

impl Case {
    fn base() -> Self {
        Self {
            what: "",
            needle: "",
            d_state: MAMBA2_STATE,
            taps: MAMBA2_TAPS,
            n_head: SMALL.n_head,
            n_group: SMALL.n_group,
            head_dim: SMALL.head_dim,
            lanes: 2,
            lane_at: 0,
            lane_to: 1,
            pos_len: 1,
            short: "",
            long: "",
        }
    }
}

/// The raw encode, so that a refusal keeps its `DeviceError`.
fn raw(
    h: &Harness<'_>,
    launch: impl FnOnce(&mut ComputeEncoder<'_>) -> Result<(), DeviceError>,
) -> Result<(), DeviceError> {
    let mut recording = h.stream.begin()?;
    {
        let mut encoder = recording.compute_encoder()?;
        launch(&mut encoder)?;
        encoder.end();
    }
    h.stream.submit(recording)
}

/// The refusal of `kernel` whose detail holds `needle`.
fn expect_refusal(
    got: Result<(), DeviceError>,
    kernel: &str,
    what: &str,
    needle: &str,
) -> Result<(), String> {
    match got {
        Ok(()) => Err(format!("{what} was accepted")),
        Err(DeviceError::Range {
            what: named,
            detail,
        }) => {
            if named != kernel {
                return Err(format!(
                    "{what} was refused as {named}, not {kernel}: {detail}"
                ));
            }
            if !detail.contains(needle) {
                return Err(format!(
                    "{what} was refused without naming {needle}: {detail}"
                ));
            }
            Ok(())
        }
        Err(other) => Err(format!("{what} was refused, not as a named range: {other}")),
    }
}

/// The launches a kernel refuses by name before it encodes anything, and the
/// well-formed launch that is not refused.
fn refusals(h: &Harness<'_>) -> Clause {
    let (mut rig, host, tok, _) = small(h, 2)?;
    let shape = rig.shape;
    let n_state = shape.inner * MAMBA2_STATE;
    let cases = [
        Case {
            what: "state 96",
            needle: "state 96",
            d_state: 96,
            ..Case::base()
        },
        Case {
            what: "state 256",
            needle: "state 256",
            d_state: 256,
            ..Case::base()
        },
        Case {
            what: "3 conv taps",
            needle: "conv taps 3",
            taps: 3,
            ..Case::base()
        },
        Case {
            what: "no heads",
            needle: "n_head of 0",
            n_head: 0,
            ..Case::base()
        },
        Case {
            what: "no groups",
            needle: "n_group of 0",
            n_group: 0,
            ..Case::base()
        },
        Case {
            what: "heads that do not divide into groups",
            needle: "not a multiple of n_group",
            n_head: 10,
            n_group: 4,
            ..Case::base()
        },
        Case {
            what: "no channels",
            needle: "head_dim of 0",
            head_dim: 0,
            ..Case::base()
        },
        Case {
            what: "no lanes",
            needle: "lanes of 0",
            lanes: 0,
            ..Case::base()
        },
        Case {
            what: "two positions",
            needle: "position window",
            pos_len: 2,
            ..Case::base()
        },
        Case {
            what: "a read lane index past the lane window",
            needle: "read lane index",
            lane_at: 2,
            ..Case::base()
        },
        Case {
            what: "a write lane index past the lane window",
            needle: "write lane index",
            lane_to: 2,
            ..Case::base()
        },
    ];
    let mut count = cases.len();
    let windows = [
        "xbc",
        "dt",
        "dt bias",
        "z",
        "a",
        "d",
        "conv weight",
        "conv bias",
        "state",
        "ring",
        "y",
        "partials",
    ];
    let mut all: Vec<Case> = cases.to_vec();
    for name in windows {
        for (short, long) in [(name, ""), ("", name)] {
            all.push(Case {
                what: name,
                needle: name,
                short,
                long,
                ..Case::base()
            });
            count += 1;
        }
    }
    // The refusals write nothing.
    let (before_state, before_ring) = (rig.state_host.clone(), rig.ring_host.clone());
    for case in &all {
        let got = try_step(h, &mut rig, &tok, case);
        let named = if case.short.is_empty() && case.long.is_empty() {
            case.needle.to_string()
        } else {
            format!("the {} window", case.needle)
        };
        expect_refusal(got, "mamba2 step", case.what, &named)?;
        let back = rig.read()?;
        same_bits(case.what, &back.state, &before_state)?;
        same_bits(case.what, &back.ring, &before_ring)?;
    }
    // The norm's refusals.
    let norm_cases: [(&str, &str, f32, usize); 5] = [
        ("a NaN epsilon", "epsilon", f32::NAN, 0),
        ("an infinite epsilon", "epsilon", f32::INFINITY, 0),
        ("y one word short", "the y window", 1e-5, 1),
        ("partials one word short", "the partials window", 1e-5, 2),
        ("out one word short", "the out window", 1e-5, 3),
    ];
    for (what, needle, eps, short) in norm_cases {
        let got = try_norm(h, &mut rig, eps, short);
        expect_refusal(got, "mamba2 norm", what, needle)?;
        count += 1;
    }
    // The well-formed launches are not refused and write.
    let words = [0u32, 1];
    rig.step(&tok, &words, (0, 1))?;
    expect_clean(h, "the well-formed step")?;
    rig.norm(host.eps)?;
    let back = rig.read()?;
    if back.state[n_state..2 * n_state] == before_state[n_state..2 * n_state] {
        return Err("the well-formed step wrote nothing".to_string());
    }
    Ok(Verdict::Noted(format!("{count} refusals")))
}

/// One step launch of `case`, as a `DeviceError`.
fn try_step(
    h: &Harness<'_>,
    rig: &mut Rig<'_>,
    tok: &Token,
    case: &Case,
) -> Result<(), DeviceError> {
    let shape = rig.shape;
    let len = |name: &str, normal: usize| {
        if case.short == name {
            normal - 1
        } else if case.long == name {
            normal + 1
        } else {
            normal
        }
    };
    let up = |body: &[f32]| Buf::input(h, body).map_err(|e| DeviceError::backend("gate", e));
    let xbc = up(&tok.xbc)?;
    let dt = up(&tok.dt)?;
    let z = up(&tok.z)?;
    let lane = h.device.upload(h.stream, &[0u32, 1])?;
    let pos = [tok.pos, tok.pos];
    let n_state = rig.lanes * shape.inner * MAMBA2_STATE;
    raw(h, |encoder| {
        let to_dev = |e: String| DeviceError::backend("gate", e);
        let params = Mamba2Params {
            xbc: xbc.win_of(len("xbc", shape.conv_dim)).map_err(to_dev)?,
            dt: dt.win_of(len("dt", shape.n_head)).map_err(to_dev)?,
            dt_bias: rig
                .dt_bias
                .win_of(len("dt bias", shape.n_head))
                .map_err(to_dev)?,
            z: z.win_of(len("z", shape.inner)).map_err(to_dev)?,
            a: rig.a.win_of(len("a", shape.n_head)).map_err(to_dev)?,
            d: rig.d.win_of(len("d", shape.n_head)).map_err(to_dev)?,
            conv_w: rig
                .conv_w
                .win_of(len("conv weight", MAMBA2_TAPS * shape.conv_dim))
                .map_err(to_dev)?,
            conv_b: rig
                .conv_b
                .win_of(len("conv bias", shape.conv_dim))
                .map_err(to_dev)?,
            lane: lane.window(Elems(0), Elems(2))?,
            state: rig
                .state
                .win_mut_of(len("state", n_state))
                .map_err(to_dev)?,
            ring: rig
                .ring
                .win_mut_of(len("ring", shape.ring_elems()))
                .map_err(to_dev)?,
            y: rig.y.win_mut_of(len("y", shape.inner)).map_err(to_dev)?,
            part: rig
                .part
                .win_mut_of(len("partials", shape.partials()))
                .map_err(to_dev)?,
            faults: h.device.fault_word(),
            n_head: case.n_head,
            n_group: case.n_group,
            head_dim: case.head_dim,
            d_state: case.d_state,
            taps: case.taps,
            lanes: case.lanes,
            lane_at: case.lane_at,
            lane_to: case.lane_to,
            layer: LAYER,
            pos: &pos[..case.pos_len],
        };
        h.kernels.mamba2_step(encoder, params)
    })
}

/// One norm launch with a bad epsilon, or with the window `short` (1 `y`, 2
/// `part`, 3 `out`) one word short.
fn try_norm(h: &Harness<'_>, rig: &mut Rig<'_>, eps: f32, short: usize) -> Result<(), DeviceError> {
    let shape = rig.shape;
    let less = |which: usize, n: usize| if short == which { n - 1 } else { n };
    raw(h, |encoder| {
        let to_dev = |e: String| DeviceError::backend("gate", e);
        let params = Mamba2NormParams {
            y: rig.y.win_of(less(1, shape.inner)).map_err(to_dev)?,
            part: rig.part.win_of(less(2, shape.partials())).map_err(to_dev)?,
            gain: rig.gain.win().map_err(to_dev)?,
            out: rig.out.win_mut_of(less(3, shape.inner)).map_err(to_dev)?,
            faults: h.device.fault_word(),
            n_head: shape.n_head,
            n_group: shape.n_group,
            head_dim: shape.head_dim,
            layer: LAYER,
            eps,
        };
        h.kernels.mamba2_norm(encoder, params)
    })
}

/// The host engine's Mamba-2 step (`model::mamba2`) against the oracle, on
/// the same inputs: the oracle is a transcription of that rule, and this holds
/// the transcription to it. Each position runs the host's conv, scan, `D`
/// skip and `z` gate (no norm) from the state it holds, and its state and `y`
/// must sit inside the oracle's bands; the host's grouped norm of that `y`
/// must sit inside the oracle's norm band. The host's conv state is the ring's
/// predecessors, so the zero fill and the oracle's ring rule meet it.
fn cpu_tie() -> Clause {
    let shape =
        Mamba2Shape::of("mamba2 step", SMALL.n_head, SMALL.n_group, SMALL.head_dim).map_err(why)?;
    let mut rng = Rng::new(SMALL.seed + 2);
    let host = Host::new(&mut rng, shape, SMALL.dt);
    let w = host.weights();
    let geom = cpu::check(cpu::Geom {
        d_inner: shape.inner,
        n_heads: shape.n_head,
        n_groups: shape.n_group,
        d_state: MAMBA2_STATE,
        d_conv: MAMBA2_TAPS,
    })
    .map_err(|e| e.to_string())?;
    let mut scan_state = draw(&mut rng, shape.inner * MAMBA2_STATE, 0.3);
    let mut conv_state = vec![0f32; (MAMBA2_TAPS - 1) * shape.conv_dim];
    let mut ring = vec![0f32; shape.ring_elems()];
    let mut worst = Fit::default();
    for pos in 0..13u32 {
        let tok = Token::new(&mut rng, shape, pos, SMALL.dt);
        let before = scan_state.clone();
        let mut y = vec![0f32; shape.inner];
        cpu::step_token(
            &geom,
            &mut conv_state,
            &mut scan_state,
            &cpu::Token {
                xbc: &tok.xbc,
                dt: &tok.dt,
                dt_bias: &host.dt_bias,
                z: &tok.z,
                a: &host.a,
                d: &host.d,
                conv_w: &host.conv_w,
                conv_bias: &host.conv_b,
                norm: None,
                eps: host.eps,
            },
            &mut y,
        )
        .map_err(|e| format!("position {pos}: {e}"))?;
        let inp = StepIn {
            shape,
            w: &w,
            pos,
            xbc: &tok.xbc,
            dt: &tok.dt,
            z: &tok.z,
            state: &before,
            ring: &ring,
        };
        let want = oracle::step(&inp)?;
        let at = |what: &str, e: String| format!("position {pos}: the host's {what}: {e}");
        worst.state = worst.state.max(
            oracle::scan(
                &scan_state,
                &want.state,
                &want.state_err,
                oracle::STEP_FLOOR,
                "state",
            )
            .map_err(|e| at("state", e))?,
        );
        let (yv, ye) = oracle::y_gated(&inp, &want, &scan_state)?;
        worst.y = worst
            .y
            .max(oracle::scan(&y, &yv, &ye, oracle::STEP_FLOOR, "y").map_err(|e| at("y", e))?);
        let mut normed = y.clone();
        cpu::grouped_rms(&mut normed, &host.gain, shape.n_group, host.eps)
            .map_err(|e| e.to_string())?;
        let (pv, _) = oracle::partials(shape, &y);
        let part: Vec<f32> = pv.iter().map(|&v| v as f32).collect();
        let (nv, ne) = oracle::norm(shape, &w, &y, &part)?;
        worst.norm = worst.norm.max(
            oracle::scan(&normed, &nv, &ne, oracle::STEP_FLOOR, "norm")
                .map_err(|e| at("norm", e))?,
        );
        let slot = pos as usize % RING_ROWS;
        ring[slot * shape.conv_dim..(slot + 1) * shape.conv_dim].copy_from_slice(&tok.xbc);
    }
    Ok(Verdict::Noted(format!(
        "13 positions of the host's step; worst error over its band: state {:.3}, y {:.3}, norm {:.3}",
        worst.state, worst.y, worst.norm
    )))
}
