//! Qwen3.8's gated-residual hyper-connections as host rules: the one table of
//! the shapes the card's kernels (`bloomery_gpu::hc_gated`) are built for, the
//! refusal of every other shape by name, the buffer sizes a launch reads and
//! writes, and the f64 reference the kernels' gate holds them to.
//!
//! The residual is `streams` f32 streams of `hidden` values. Around each
//! sub-layer (llama.cpp `src/models/qwen4exp.cpp` `build_hc_mix`,
//! `build_hc_combine`):
//! - **mix**: `xn_s = rms_norm(x_s) · γ_s` per stream (the file carries `γ`
//!   with the `+1` folded in); `lo = silu(down · xn / streams)` over the
//!   streams concatenated (`rank` values); `g = up · lo` (`streams · hidden`
//!   values); `mixed = mean_s(xn_s ⊙ σ(g_s))`, the sub-layer's input; and
//!   `inject = W_inj · xn` (`streams` scalars) where the site carries one;
//! - **combine**: `x_s += wgt_s · y` with `wgt_s = 2·σ(inject_s / streams)` and
//!   `y` the sub-layer's output — no mix between streams;
//! - the **head** is a mix with no inject after the last layer, the output norm.
//!
//! Layouts, per column (token) `c` of `m`: the streams `[m][streams][hidden]`,
//! `γ` `[streams][hidden]`, `down` `rank` rows of `streams · hidden`, `up`
//! `streams · hidden` rows of `rank`, `W_inj` `streams` rows of
//! `streams · hidden` — each a flat index `s·hidden + d` over the streams.

use std::fmt;

/// A shape the card's kernels are built for: `streams` streams through a
/// `rank`-value bottleneck. Every other shape is refused by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Instance {
    pub streams: u32,
    pub rank: u32,
}

/// The instance table: Qwen3.8-Flash-Next's `hyper_connection.count` 4 and
/// `hyper_connection.low_rank` 320.
pub const INSTANCES: &[Instance] = &[Instance {
    streams: 4,
    rank: 320,
}];

/// The most columns (tokens) one launch serves: a decode step or a verify.
pub const MAX_COLS: usize = 8;

/// `hidden` must be a multiple of this: the down gemv's inject rows split a
/// stream across eight warps of whole 32-value lane chunks.
pub const HIDDEN_ALIGN: u32 = 256;

/// A shape or a launch the host rules refuse, by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HcRefused {
    /// No kernel instance serves `streams` streams at rank `rank`.
    NoInstance { streams: u32, rank: u32 },
    /// `hidden` is zero or not a multiple of [`HIDDEN_ALIGN`].
    Hidden { hidden: u32 },
    /// A launch of `m` columns, outside `1..=MAX_COLS`.
    Cols { m: usize },
}

impl fmt::Display for HcRefused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            HcRefused::NoInstance { streams, rank } => write!(
                f,
                "gated-residual hyper-connections: no kernel instance for {streams} streams at \
                 rank {rank} (instances: {})",
                INSTANCES
                    .iter()
                    .map(|i| format!("{}x{}", i.streams, i.rank))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            HcRefused::Hidden { hidden } => write!(
                f,
                "gated-residual hyper-connections: hidden {hidden} is not a positive multiple of \
                 {HIDDEN_ALIGN}"
            ),
            HcRefused::Cols { m } => write!(
                f,
                "gated-residual hyper-connections: a launch of {m} columns, not 1..={MAX_COLS}"
            ),
        }
    }
}

impl std::error::Error for HcRefused {}

/// The instance for `streams` and `rank`, or its refusal by name. No shape is
/// mapped to the nearest row.
///
/// # Errors
/// [`HcRefused::NoInstance`] when the table has no such row.
pub fn select(streams: u32, rank: u32) -> Result<Instance, HcRefused> {
    INSTANCES
        .iter()
        .copied()
        .find(|i| i.streams == streams && i.rank == rank)
        .ok_or(HcRefused::NoInstance { streams, rank })
}

/// A model's hyper-connection geometry: an instance and the hidden width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub instance: Instance,
    pub hidden: u32,
}

impl Geometry {
    /// The geometry of `streams` streams of `hidden` at rank `rank`.
    ///
    /// # Errors
    /// [`HcRefused::NoInstance`] for a shape no instance serves,
    /// [`HcRefused::Hidden`] for a hidden width the kernels do not split.
    pub fn new(streams: u32, rank: u32, hidden: u32) -> Result<Geometry, HcRefused> {
        let instance = select(streams, rank)?;
        if hidden == 0 || !hidden.is_multiple_of(HIDDEN_ALIGN) {
            return Err(HcRefused::Hidden { hidden });
        }
        Ok(Geometry { instance, hidden })
    }

    /// Streams.
    #[must_use]
    pub fn s(self) -> usize {
        self.instance.streams as usize
    }

    /// The bottleneck's width.
    #[must_use]
    pub fn r(self) -> usize {
        self.instance.rank as usize
    }

    /// Values of one stream.
    #[must_use]
    pub fn d(self) -> usize {
        self.hidden as usize
    }

    /// Values of all streams of one column: the down's and the inject's `k`,
    /// the up's rows.
    #[must_use]
    pub fn wide(self) -> usize {
        self.s() * self.d()
    }

    /// `m` itself, or its refusal.
    ///
    /// # Errors
    /// [`HcRefused::Cols`] for `m` outside `1..=MAX_COLS`.
    pub fn cols(self, m: usize) -> Result<usize, HcRefused> {
        if m == 0 || m > MAX_COLS {
            return Err(HcRefused::Cols { m });
        }
        Ok(m)
    }

    /// The buffer sizes of an `m`-column launch, in f32.
    #[must_use]
    pub fn sizes(self, m: usize) -> Sizes {
        Sizes {
            res: m * self.wide(),
            y: m * self.d(),
            xn: m * self.wide(),
            down_part: m * self.r() * self.s(),
            inject_part: m * self.s() * self.s(),
            lo: m * self.r(),
            wgt: m * self.s(),
            mixed: m * self.d(),
        }
    }
}

/// The f32 lengths of the buffers an `m`-column mix and combine touch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sizes {
    /// The streams, `[m][streams][hidden]`, updated in place by a combine.
    pub res: usize,
    /// A sub-layer's output, `[m][hidden]`.
    pub y: usize,
    /// The normed streams, `[m][streams][hidden]`.
    pub xn: usize,
    /// The down's per-stream partial dots, `[m][rank][streams]`.
    pub down_part: usize,
    /// The inject's per-stream partial dots, `[m][streams (row)][streams]`.
    pub inject_part: usize,
    /// The bottleneck after its silu, `[m][rank]`.
    pub lo: usize,
    /// The combine's weights `2·σ(inject / streams)`, `[m][streams]`.
    pub wgt: usize,
    /// The sub-layer's input, `[m][hidden]`.
    pub mixed: usize,
}

/// `x / (1 + e^−x)` in f64.
#[must_use]
pub fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// `1 / (1 + e^−x)` in f64.
#[must_use]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// One column's mix in f64, every intermediate kept.
#[derive(Clone, Debug, PartialEq)]
pub struct MixRef {
    /// `[streams][hidden]`.
    pub xn: Vec<f64>,
    /// `[rank]`, after the silu.
    pub lo: Vec<f64>,
    /// `[streams][hidden]`, before the sigmoid.
    pub g: Vec<f64>,
    /// `[hidden]`.
    pub mixed: Vec<f64>,
    /// `[streams]`, `2·σ(inject / streams)`; `None` for the head.
    pub wgt: Option<Vec<f64>>,
}

/// One site's weights, dequantized to f32 in the layouts of the module doc.
#[derive(Clone, Copy, Debug)]
pub struct MixWeights<'a> {
    pub gamma: &'a [f32],
    pub down: &'a [f32],
    pub up: &'a [f32],
    /// `None` for the head.
    pub inject: Option<&'a [f32]>,
}

/// The mix of one column `x` (`[streams][hidden]`) in f64 over the f32
/// inputs: the rule the module doc states, every sum in f64.
///
/// # Panics
/// On a weight or input whose length is not the geometry's, by name.
#[must_use]
pub fn mix_ref(geo: Geometry, w: MixWeights<'_>, x: &[f32], eps: f32) -> MixRef {
    let (s, r, d, wide) = (geo.s(), geo.r(), geo.d(), geo.wide());
    let lens = [
        ("x", x.len(), wide),
        ("gamma", w.gamma.len(), wide),
        ("down", w.down.len(), r * wide),
        ("up", w.up.len(), wide * r),
    ];
    for (name, got, want) in lens {
        assert_eq!(
            got, want,
            "hc_gated::mix_ref: {name} holds {got} values, want {want}"
        );
    }
    if let Some(inj) = w.inject {
        assert_eq!(
            inj.len(),
            s * wide,
            "hc_gated::mix_ref: inject holds {} values, want {}",
            inj.len(),
            s * wide
        );
    }
    let hc = s as f64;
    let mut xn = vec![0.0f64; wide];
    for st in 0..s {
        let xs = &x[st * d..(st + 1) * d];
        let ms = xs.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / d as f64;
        let scale = 1.0 / (ms + f64::from(eps)).sqrt();
        for (i, &v) in xs.iter().enumerate() {
            xn[st * d + i] = f64::from(v) * scale * f64::from(w.gamma[st * d + i]);
        }
    }
    let dot = |row: &[f32], v: &[f64]| -> f64 {
        row.iter().zip(v).map(|(&a, &b)| f64::from(a) * b).sum()
    };
    let lo: Vec<f64> = (0..r)
        .map(|j| silu(dot(&w.down[j * wide..(j + 1) * wide], &xn) / hc))
        .collect();
    let g: Vec<f64> = (0..wide)
        .map(|i| dot(&w.up[i * r..(i + 1) * r], &lo))
        .collect();
    let mixed: Vec<f64> = (0..d)
        .map(|i| {
            (0..s)
                .map(|st| xn[st * d + i] * sigmoid(g[st * d + i]))
                .sum::<f64>()
                / hc
        })
        .collect();
    let wgt = w.inject.map(|inj| {
        (0..s)
            .map(|o| 2.0 * sigmoid(dot(&inj[o * wide..(o + 1) * wide], &xn) / hc))
            .collect()
    });
    MixRef {
        xn,
        lo,
        g,
        mixed,
        wgt,
    }
}

/// The combine as the card rounds it, over `m` columns: `res[c][s][d] =
/// fma(wgt[c][s], y[c][d], res[c][s][d])`, one rounding per value.
///
/// # Panics
/// On a buffer whose length is not the geometry's for `m`, by name.
pub fn combine(geo: Geometry, m: usize, res: &mut [f32], y: &[f32], wgt: &[f32]) {
    let z = geo.sizes(m);
    for (name, got, want) in [
        ("res", res.len(), z.res),
        ("y", y.len(), z.y),
        ("wgt", wgt.len(), z.wgt),
    ] {
        assert_eq!(
            got, want,
            "hc_gated::combine: {name} holds {got} values, want {want}"
        );
    }
    let (s, d) = (geo.s(), geo.d());
    for c in 0..m {
        for st in 0..s {
            let w = wgt[c * s + st];
            for i in 0..d {
                let at = (c * s + st) * d + i;
                res[at] = w.mul_add(y[c * d + i], res[at]);
            }
        }
    }
}

/// Bands of the card's mix against [`mix_ref`], as `max|card − ref| /
/// max|ref|` over one intermediate of one column, at the file's shape (4 ×
/// 2560, rank 320). Worst-case bounds, u = 2⁻²⁴; each band is its bound with
/// a margin of 4–6.
///
/// PIN(2026-09-27): `xn` — a thread's sum of squares is at most 10 fused
/// multiply-adds, then 5 butterfly steps and 7 warp sums, all terms positive:
/// ≤ 22u; the mean, root and reciprocal add ≤ 2u, the root halves the sum's
/// share; `(x·r)·γ` two roundings: ≤ 15u ≈ 9e-7.
pub const XN_BAND: f32 = 4e-6;
/// PIN(2026-09-27): `lo` and the combine weights — a stream's scale error
/// (≤ 13u, one factor for the whole stream) moves the dot by ≤ 13u·Σ_s|dot_s|
/// ≤ 26u of max|dot|; the lane walk (80 fused multiply-adds, 5 butterfly
/// steps, 3 stream sums) adds a random-walk ≈ u·√80·√128·σ_t against max|dot|
/// ≈ 3·√10240·σ_t, under 5u; silu and 2σ carry it at slope ≤ 1.1 and add ≤ 2u:
/// ≤ 35u ≈ 2e-6.
pub const LO_BAND: f32 = 1e-5;
/// PIN(2026-09-27): `mixed` — `lo`'s error reaches `g` through the rank-320
/// dot at a gain of Σ_j|up·lo| / max|g| ≈ 320·0.8 / (3·√320) ≈ 14 on random
/// inputs: ≤ 3e-5 of max|g|; σ's slope (≤ 1/4) and the mean (1/4 against
/// max|mixed| ≈ max|xn|/2) scale it by ≤ 1/8: ≤ 4e-6; `xn`'s ≤ 9e-7 adds
/// directly: ≤ 5e-6.
pub const MIXED_BAND: f32 = 3e-5;

/// The card's order, one stage at a time, in f32: what each kernel of
/// `bloomery_gpu::hc_gated` computes from its inputs, operation for
/// operation, so a stage fed the card's readback of its inputs reproduces the
/// card's output bit for bit. `silu` and `sigmoid` are the caller's: the card
/// runs `bloomery_gpu::linear`'s, a native test the system `exp`'s. Every
/// function takes one column.
pub mod card {
    use super::Geometry;

    /// Threads of a norm block: one block a stream and column.
    pub const NORM_THREADS: usize = 256;
    /// Warps of a norm block.
    pub const NORM_WARPS: usize = NORM_THREADS / 32;
    /// Warps of a down or inject block: a down block takes this many rows of
    /// one stream, an inject block one row of one stream in this many chunks.
    pub const DOWN_WARPS: usize = 8;

    /// The activation functions the stages call.
    #[derive(Clone, Copy, Debug)]
    pub struct Act {
        pub silu: fn(f32) -> f32,
        pub sigmoid: fn(f32) -> f32,
    }

    /// `warp::reduce_sum_f32`: `v += shfl_xor(v, o)` for `o` = 16, 8, 4, 2, 1,
    /// lane 0's value (every lane's is the same).
    #[must_use]
    pub fn butterfly(lanes: &[f32; 32]) -> f32 {
        let mut v = *lanes;
        for off in [16usize, 8, 4, 2, 1] {
            let w = v;
            for (l, x) in v.iter_mut().enumerate() {
                *x = w[l] + w[l ^ off];
            }
        }
        v[0]
    }

    /// A Q8_0 row's dot in the q8f32 lane order (`q8_0_lane_partial_1col`):
    /// lane `L` takes words `L, L + 32, …` of the row's `k/4`, each word's
    /// four values in order, one fused multiply-add each, then the
    /// butterfly. `w` is the row dequantized (`q·d`, exact in f32).
    #[must_use]
    pub fn q8_dot(w: &[f32], x: &[f32]) -> f32 {
        let words = w.len() / 4;
        let mut lanes = [0.0f32; 32];
        for (l, acc) in lanes.iter_mut().enumerate() {
            let mut wd = l;
            while wd < words {
                for j in 0..4 {
                    *acc = w[4 * wd + j].mul_add(x[4 * wd + j], *acc);
                }
                wd += 32;
            }
        }
        butterfly(&lanes)
    }

    /// An F32 row's dot in `f32_lane_partial_1col`'s order: lane `L` takes
    /// values `L, L + 32, …`, one fused multiply-add each, then the butterfly.
    #[must_use]
    pub fn f32_dot(w: &[f32], x: &[f32]) -> f32 {
        let mut lanes = [0.0f32; 32];
        for (l, acc) in lanes.iter_mut().enumerate() {
            let mut i = l;
            while i < w.len() {
                *acc = w[i].mul_add(x[i], *acc);
                i += 32;
            }
        }
        butterfly(&lanes)
    }

    /// `hc_gated_norm`: per stream, thread `t` of [`NORM_THREADS`] sums the
    /// squares of values `t, t + 256, …` by fused multiply-adds, each warp by
    /// the butterfly, the warps in order from warp 0; `r = 1 / √(sum / hidden
    /// + eps)`, `xn = (x·r)·γ`. `x` is the column's streams after its combine.
    #[must_use]
    pub fn norm(geo: Geometry, x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        let (s, d) = (geo.s(), geo.d());
        let mut xn = vec![0.0f32; s * d];
        for st in 0..s {
            let xs = &x[st * d..(st + 1) * d];
            let mut th = [0.0f32; NORM_THREADS];
            for (t, acc) in th.iter_mut().enumerate() {
                let mut i = t;
                while i < d {
                    *acc = xs[i].mul_add(xs[i], *acc);
                    i += NORM_THREADS;
                }
            }
            let mut sum = 0.0f32;
            for wp in 0..NORM_WARPS {
                let mut lanes = [0.0f32; 32];
                lanes.copy_from_slice(&th[wp * 32..(wp + 1) * 32]);
                let v = butterfly(&lanes);
                sum = if wp == 0 { v } else { sum + v };
            }
            let r = 1.0 / (sum / d as f32 + eps).sqrt();
            for i in 0..d {
                xn[st * d + i] = (xs[i] * r) * gamma[st * d + i];
            }
        }
        xn
    }

    /// `hc_gated_down`'s partials: `[rank][streams]`, row `j` stream `s` the
    /// [`q8_dot`] of the row's stream-`s` values with `xn`'s stream `s` — the
    /// q8f32 gemv of the `[rank·streams × hidden]` view, row `j·streams + s`.
    #[must_use]
    pub fn down_parts(geo: Geometry, down: &[f32], xn: &[f32]) -> Vec<f32> {
        let (s, d, wide) = (geo.s(), geo.d(), geo.wide());
        (0..geo.r() * s)
            .map(|e| {
                let (j, st) = (e / s, e % s);
                q8_dot(
                    &down[j * wide + st * d..j * wide + (st + 1) * d],
                    &xn[st * d..(st + 1) * d],
                )
            })
            .collect()
    }

    /// The streams' partials of one value added in stream order, from stream 0.
    fn in_order(p: &[f32]) -> f32 {
        let mut v = p[0];
        for &x in &p[1..] {
            v += x;
        }
        v
    }

    /// `lo[j] = silu(v · (1/streams))`, `v` row `j`'s partials in stream order.
    #[must_use]
    pub fn lo(geo: Geometry, part: &[f32], act: Act) -> Vec<f32> {
        let s = geo.s();
        let inv = 1.0 / s as f32;
        (0..geo.r())
            .map(|j| (act.silu)(in_order(&part[j * s..(j + 1) * s]) * inv))
            .collect()
    }

    /// `hc_gated_down`'s inject partials: `[streams (row)][streams]`, row `o`
    /// stream `s` the sum, warp by warp from warp 0, of [`DOWN_WARPS`] chunks
    /// of `hidden / DOWN_WARPS` values, each an [`f32_dot`].
    #[must_use]
    pub fn inject_parts(geo: Geometry, inject: &[f32], xn: &[f32]) -> Vec<f32> {
        let (s, d, wide) = (geo.s(), geo.d(), geo.wide());
        let chunk = d / DOWN_WARPS;
        (0..s * s)
            .map(|e| {
                let (o, st) = (e / s, e % s);
                let mut v = 0.0f32;
                for wp in 0..DOWN_WARPS {
                    let at = st * d + wp * chunk;
                    let p = f32_dot(
                        &inject[o * wide + at..o * wide + at + chunk],
                        &xn[at..at + chunk],
                    );
                    v = if wp == 0 { p } else { v + p };
                }
                v
            })
            .collect()
    }

    /// `wgt[o] = 2 · σ(v · (1/streams))`, `v` row `o`'s partials in stream order.
    #[must_use]
    pub fn wgt(geo: Geometry, part: &[f32], act: Act) -> Vec<f32> {
        let s = geo.s();
        let inv = 1.0 / s as f32;
        (0..s)
            .map(|o| 2.0 * (act.sigmoid)(in_order(&part[o * s..(o + 1) * s]) * inv))
            .collect()
    }

    /// `hc_gated_up_mix`: `g` row `s·hidden + i` the [`q8_dot`] of the up row
    /// with `lo`; `mixed[i] = (xn_0·σ(g_0), then fma(xn_s, σ(g_s), ·) for
    /// s = 1, …) · (1/streams)`.
    #[must_use]
    pub fn up_mix(geo: Geometry, up: &[f32], lo: &[f32], xn: &[f32], act: Act) -> Vec<f32> {
        let (s, r, d) = (geo.s(), geo.r(), geo.d());
        let inv = 1.0 / s as f32;
        (0..d)
            .map(|i| {
                let mut acc = 0.0f32;
                for st in 0..s {
                    let row = st * d + i;
                    let sg = (act.sigmoid)(q8_dot(&up[row * r..(row + 1) * r], lo));
                    acc = if st == 0 {
                        xn[row] * sg
                    } else {
                        xn[row].mul_add(sg, acc)
                    };
                }
                acc * inv
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: usize = 4;
    const R: usize = 320;
    const D: usize = 2560;
    const EPS: f32 = 1e-6;

    fn q38() -> Geometry {
        Geometry::new(4, 320, 2560).expect("Qwen3.8's shape")
    }

    #[test]
    fn only_the_table_rows_are_served_and_the_rest_refused_by_name() {
        assert_eq!(
            select(4, 320),
            Ok(Instance {
                streams: 4,
                rank: 320
            })
        );
        for (s, r) in [(3, 320), (8, 320), (4, 256), (4, 321), (0, 0)] {
            let e = select(s, r).expect_err("no such row");
            assert_eq!(
                e,
                HcRefused::NoInstance {
                    streams: s,
                    rank: r
                }
            );
            let msg = e.to_string();
            assert!(msg.contains(&format!("{s} streams at rank {r}")), "{msg}");
            assert!(msg.contains("4x320"), "{msg}");
        }
        assert_eq!(
            Geometry::new(4, 320, 2048 + 128),
            Err(HcRefused::Hidden { hidden: 2176 })
        );
        assert_eq!(
            Geometry::new(4, 320, 0),
            Err(HcRefused::Hidden { hidden: 0 })
        );
        assert_eq!(
            Geometry::new(4, 256, 2560),
            Err(HcRefused::NoInstance {
                streams: 4,
                rank: 256
            })
        );
        let g = q38();
        assert_eq!((g.s(), g.r(), g.d(), g.wide()), (S, R, D, S * D));
        assert_eq!(g.cols(0), Err(HcRefused::Cols { m: 0 }));
        assert_eq!(g.cols(9), Err(HcRefused::Cols { m: 9 }));
        assert_eq!(g.cols(8), Ok(8));
    }

    #[test]
    fn sizes_are_the_layouts_of_the_module_doc() {
        let z = q38().sizes(8);
        assert_eq!(
            z,
            Sizes {
                res: 8 * 10240,
                y: 8 * 2560,
                xn: 8 * 10240,
                down_part: 8 * 320 * 4,
                inject_part: 8 * 16,
                lo: 8 * 320,
                wgt: 8 * 4,
                mixed: 8 * 2560,
            }
        );
    }

    /// A fixed LCG in `[-1, 1)`, every 61st value ×8.
    fn lcg(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(12_345);
        (0..n)
            .map(|i| {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let u = ((s >> 8) & 0xffff) as f32 / 32_768.0 - 1.0;
                if i % 61 == 0 { u * 8.0 } else { u }
            })
            .collect()
    }

    /// `n` Q8_0-shaped weights (codes in −127..=127 times a per-32 scale
    /// `d = base · (1 + k/8)`, exact in f16 and f32), dequantized.
    fn q8_weights(n: usize, base: f32, seed: u32) -> Vec<f32> {
        let u = lcg(n + n / 32, seed);
        (0..n)
            .map(|i| {
                let k = ((u[n + i / 32] + 1.0) * 4.0) as i32 as f32;
                let d = base * (1.0 + k / 8.0);
                let q = (u[i] * 127.0).round().clamp(-127.0, 127.0);
                q * d
            })
            .collect()
    }

    struct Site {
        gamma: Vec<f32>,
        down: Vec<f32>,
        up: Vec<f32>,
        inject: Vec<f32>,
    }

    fn site(seed: u32) -> Site {
        Site {
            gamma: lcg(S * D, seed).iter().map(|v| 1.0 + v / 64.0).collect(),
            down: q8_weights(R * S * D, 1.0 / 4096.0, seed + 1),
            up: q8_weights(S * D * R, 1.0 / 256.0, seed + 2),
            inject: lcg(S * S * D, seed + 3).iter().map(|v| v / 64.0).collect(),
        }
    }

    fn weights(st: &Site, inject: bool) -> MixWeights<'_> {
        MixWeights {
            gamma: &st.gamma,
            down: &st.down,
            up: &st.up,
            inject: inject.then_some(st.inject.as_slice()),
        }
    }

    #[test]
    fn identities_hold_in_the_reference() {
        let g = q38();
        // A zero inject row is a plain residual add: 2·σ(0) = 1 exactly.
        let st = site(11);
        let zero = vec![0.0f32; S * S * D];
        let x = lcg(S * D, 5);
        let w = MixWeights {
            inject: Some(&zero),
            ..weights(&st, true)
        };
        let r = mix_ref(g, w, &x, EPS);
        assert_eq!(r.wgt, Some(vec![1.0; S]));
        // A zero up gives σ(0) = 1/2 everywhere: mixed is half the streams' mean.
        let up0 = vec![0.0f32; S * D * R];
        let w = MixWeights {
            up: &up0,
            ..weights(&st, false)
        };
        let r = mix_ref(g, w, &x, EPS);
        assert!(r.wgt.is_none());
        for i in 0..D {
            let want = (0..S).map(|s| r.xn[s * D + i]).sum::<f64>() / S as f64 * 0.5;
            assert!((r.mixed[i] - want).abs() <= 1e-15 * want.abs().max(1.0));
        }
        // Each stream's normed values have unit mean square before γ.
        let w = MixWeights {
            gamma: &vec![1.0f32; S * D],
            ..weights(&st, false)
        };
        let r = mix_ref(g, w, &x, 0.0);
        for s in 0..S {
            let ms = r.xn[s * D..(s + 1) * D].iter().map(|v| v * v).sum::<f64>() / D as f64;
            assert!((ms - 1.0).abs() < 1e-12, "stream {s}: {ms}");
        }
    }

    #[test]
    fn the_card_combine_is_one_fused_multiply_add_per_value() {
        let g = Geometry::new(4, 320, 256).expect("a narrow geometry");
        let m = 2;
        let z = g.sizes(m);
        let mut res = lcg(z.res, 3);
        let before = res.clone();
        let y = lcg(z.y, 4);
        let wgt = vec![0.5f32, 1.0, 1.5, 2.0, 0.25, 0.75, 1.25, 1.75];
        combine(g, m, &mut res, &y, &wgt);
        for c in 0..m {
            for s in 0..4 {
                for i in 0..256 {
                    let at = (c * 4 + s) * 256 + i;
                    let want = wgt[c * 4 + s].mul_add(y[c * 256 + i], before[at]);
                    assert_eq!(res[at].to_bits(), want.to_bits());
                }
            }
        }
    }

    fn exp_silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    fn exp_sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// One column's mix through the card-order stages, each fed the one
    /// before: the transcription's values, not the kernel's.
    fn mix_card(
        w: MixWeights<'_>,
        x: &[f32],
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let g = q38();
        let act = card::Act {
            silu: exp_silu,
            sigmoid: exp_sigmoid,
        };
        let xn = card::norm(g, x, w.gamma, eps);
        let lo = card::lo(g, &card::down_parts(g, w.down, &xn), act);
        let mixed = card::up_mix(g, w.up, &lo, &xn, act);
        let wgt = card::wgt(
            g,
            &card::inject_parts(g, w.inject.unwrap_or_default(), &xn),
            act,
        );
        (xn, lo, mixed, wgt)
    }

    /// On small integers every f32 sum is exact, so the stages' index
    /// arithmetic is checked against the plain dots: each partial reads its
    /// own row and stream, and together they read every value once.
    #[test]
    fn the_stages_read_every_value_once() {
        let g = q38();
        let int = |n: usize, seed: u32, span: f32| -> Vec<f32> {
            lcg(n, seed).iter().map(|v| (v * span).round()).collect()
        };
        let (down, up, inj) = (
            int(R * S * D, 1, 2.0),
            int(S * D * R, 2, 2.0),
            int(S * S * D, 3, 2.0),
        );
        let xn = int(S * D, 4, 3.0);
        let wide = S * D;
        let exact = |row: &[f32], v: &[f32]| -> f32 {
            row.iter()
                .zip(v)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum::<f64>() as f32
        };
        let dp = card::down_parts(g, &down, &xn);
        for j in 0..R {
            let v = ((dp[j * S] + dp[j * S + 1]) + dp[j * S + 2]) + dp[j * S + 3];
            assert_eq!(
                v,
                exact(&down[j * wide..(j + 1) * wide], &xn),
                "down row {j}"
            );
        }
        let ip = card::inject_parts(g, &inj, &xn);
        for o in 0..S {
            let v = ((ip[o * S] + ip[o * S + 1]) + ip[o * S + 2]) + ip[o * S + 3];
            assert_eq!(
                v,
                exact(&inj[o * wide..(o + 1) * wide], &xn),
                "inject row {o}"
            );
        }
        let lo = int(R, 5, 3.0);
        for row in [0, 1, D - 1, D, 3 * D + 17, wide - 1] {
            assert_eq!(
                card::q8_dot(&up[row * R..(row + 1) * R], &lo),
                exact(&up[row * R..(row + 1) * R], &lo),
                "up row {row}"
            );
        }
    }

    fn rel(got: &[f32], want: &[f64]) -> f32 {
        let den = want.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        let num = got
            .iter()
            .zip(want)
            .fold(0.0f64, |a, (&g, &w)| a.max((f64::from(g) - w).abs()));
        (num / den) as f32
    }

    /// The f32 transcription of the card's order stays inside the bands the
    /// gate pins, at the file's shape, on two columns of the gate's inputs.
    #[test]
    fn the_card_order_in_f32_is_inside_the_bands() {
        let g = q38();
        let st = site(0x0c38);
        for col in 0..2u32 {
            let x = lcg(S * D, 900 + col);
            let w = weights(&st, true);
            let r = mix_ref(g, w, &x, EPS);
            let (xn, lo, mixed, wgt) = mix_card(w, &x, EPS);
            let e = [
                ("xn", rel(&xn, &r.xn), XN_BAND),
                ("lo", rel(&lo, &r.lo), LO_BAND),
                ("mixed", rel(&mixed, &r.mixed), MIXED_BAND),
                (
                    "wgt",
                    rel(&wgt, r.wgt.as_deref().unwrap_or_default()),
                    LO_BAND,
                ),
            ];
            for (name, err, band) in e {
                println!("transcription col {col} {name} max_rel_err {err:.3e} band {band:e}");
                assert!(err <= band, "{name}: {err:e} > {band:e}");
            }
            // The inputs move every stage: a gate that could not see a
            // dropped `/streams` would read equal values here.
            let lo_max = r.lo.iter().fold(0.0f64, |a, v| a.max(v.abs()));
            let g_max = r.g.iter().fold(0.0f64, |a, v| a.max(v.abs()));
            let w = r.wgt.as_deref().unwrap_or_default();
            assert!(lo_max > 0.2 && g_max > 0.5, "lo {lo_max} g {g_max}");
            assert!(w.iter().any(|&v| (v - 1.0).abs() > 0.05), "wgt {w:?}");
        }
    }
}
