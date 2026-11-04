//! Load-time kernel picks: a site with several bit-equal ways to run one
//! kernel signature registers them here, and after the load, before any
//! capture, [`tune`] times each way on the model's own weights and keeps the
//! fastest only where it beats today's rule by the margin on every repeat
//! (`runtime::autotune::winner`). The picks are installed once on the card's
//! launchers ([`crate::model::GpuModel::autotune`]) and never change while it
//! serves; a card that installs none runs today's rule. A model supplies
//! nothing but its resident weights: the shapes come from them.
//!
//! The one site today is the MXFP8 gemv of [`STAGED_COLS`] to
//! [`STAGED_MAX_COLS`] columns over rows of a `k` the staged entries read (a
//! multiple of [`STAGED_K`]): its ways are [`Mcol`]. A signature ([`McolSig`]) is the
//! weight's rows and `k` and the column count; the ways are timed on
//! token-major outputs and the pick serves both layouts, which differ only in
//! where each row's few outputs land.
//!
//! A step reads every other weight of the model between two reads of one,
//! so each of its launches reads its weight from DRAM, whatever the
//! weight's size. The timing reads them so too: it rotates over the
//! resident weights of the signature in name order, then over windows of
//! the signature's rows cut from the other resident MXFP8 weights of its
//! `k`, until the rotation spans [`COLD_L2`] times the card's L2 (one weight
//! at least, all there are when they span less). A launch reads the weight
//! the last launch of its graph read least recently, and no way is timed
//! on weights an earlier one left in L2. A pick is a signature's, not a
//! model's: a model with more or fewer weights of the signature than the
//! load that timed it reads the same pick. A race takes
//! `ways × (REPEATS + 2)` samples, each about [`SAMPLE_US`] or one replay
//! of its graph when that is longer; the load prints the whole wall.
//!
//! Each pick is kept in the cost store (`runtime::costs::Store`) under the
//! card, the driver, the build and the signature (`runtime::autotune::key`),
//! so a later load of the same key reads it and times nothing. An entry that
//! does not decode, or names a way the signature does not race, is timed
//! again and replaced.

use std::collections::BTreeMap;
use std::fmt;
use std::time::{Duration, Instant};

use cuda_core::DeviceBuffer;
use runtime::autotune::{REPEATS, Times, key, median, winner};
use runtime::costs::{Source, Store};

use crate::graph::Graph;
use crate::planes::Format;
use crate::q8f32::{
    E4m3GemvArgs, E4m3Scales, E4m3Weight, GemvOut, STAGED_COLS, STAGED_K, STAGED_MAX_COLS,
};
use crate::site::{View, view};
use crate::tensor::DeviceTensor;
use crate::weights::{DevWeight, Weights};
use crate::{Gpu, GpuError};

/// The kernel family name an MXFP8 pick is kept and printed under.
pub const MCOL_KERNEL: &str = "mxfp8_gemv_mcol";

/// The bytes a race's rotation spans, in the card's L2 sizes (module doc).
pub const COLD_L2: usize = 2;

/// The card time a timed sample spans, at least [µs].
pub const SAMPLE_US: f64 = 2000.0;

/// The bytes a race's graph reads at least, in repeats of its rotation
/// (up to [`GRAPH_LAUNCHES`] launches), so a small weight's graph is not
/// all launch gaps.
const GRAPH_BYTES: usize = 64 << 20;

/// The most launches a race's graph repeats a short rotation to.
const GRAPH_LAUNCHES: usize = 16;

/// The bit-equal ways to run an MXFP8 gemv of `m` columns over rows the
/// staged entries read; each column's outputs are the one-column gemv's bit
/// for bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mcol {
    /// Today's rule: the staged entry at four columns, the wide stage past four.
    Rule,
    /// The unstaged `_mcol` entry.
    Plain,
    /// The first four columns staged, then the rest by the rule at their
    /// count: staged at four, else `_mcol`.
    Split,
}

impl Mcol {
    /// Every way, the rule first.
    pub const ALL: [Mcol; 3] = [Mcol::Rule, Mcol::Plain, Mcol::Split];

    /// The word the load log, the status and the store's key read.
    #[must_use]
    pub fn word(self) -> &'static str {
        match self {
            Mcol::Rule => "rule",
            Mcol::Plain => "plain",
            Mcol::Split => "split",
        }
    }

    /// The ways that launch differently at `m` columns, the rule first: from
    /// [`STAGED_COLS`] the rule and the plain walk, past it the split too;
    /// none below.
    #[must_use]
    pub fn racing(m: usize) -> &'static [Mcol] {
        match m {
            STAGED_COLS => &Mcol::ALL[..2],
            _ if m > STAGED_COLS && m <= STAGED_MAX_COLS => &Mcol::ALL,
            _ => &[],
        }
    }
}

/// An MXFP8 gemv's signature: the weight's rows and `k`, and the columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct McolSig {
    pub rows: usize,
    pub k: usize,
    pub m: usize,
}

impl fmt::Display for McolSig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rows={} k={} m={}", self.rows, self.k, self.m)
    }
}

/// A card's installed picks ([`crate::q8f32::Q8F32Kernels::install`]): the way each
/// signature runs, the rule where it names none.
#[derive(Debug, Default)]
pub struct Picks {
    mcol: BTreeMap<McolSig, Mcol>,
}

impl Picks {
    /// The way `sig` runs.
    pub(crate) fn mcol(&self, sig: McolSig) -> Mcol {
        self.mcol.get(&sig).copied().unwrap_or(Mcol::Rule)
    }
}

/// One signature's pick: the way, where it came from and the medians of the
/// rule and of the way [µs a launch] its race measured.
#[derive(Clone, Debug, PartialEq)]
pub struct Pick {
    pub sig: McolSig,
    pub way: Mcol,
    pub from: Source,
    pub rule_us: f64,
    pub way_us: f64,
}

/// What [`tune`] decided: every signature's pick and the wall it took.
#[derive(Debug)]
pub struct Tuned {
    pub picks: Vec<Pick>,
    pub wall: Duration,
}

impl Tuned {
    /// The table a card installs.
    #[must_use]
    pub fn table(&self) -> Picks {
        Picks {
            mcol: self
                .picks
                .iter()
                .filter(|p| p.way != Mcol::Rule)
                .map(|p| (p.sig, p.way))
                .collect(),
        }
    }
}

/// The values a pick is kept as.
const NAMES: [&str; 3] = ["way", "rule_us", "way_us"];

/// Time every MXFP8 signature of `weights` at each column count with more
/// than one way ([`Mcol::racing`]) on `gpu`, or read its pick from `store`
/// under `build`'s key (module doc). Load-time; installs nothing.
pub fn tune(gpu: &Gpu, weights: &Weights, build: &str, store: &Store) -> Result<Tuned, GpuError> {
    let t0 = Instant::now();
    let l2 = crate::raw_device_attribute(
        gpu.context().cu_device(),
        cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
        "cuDeviceGetAttribute(L2_CACHE_SIZE)",
    )?;
    let l2 = usize::try_from(l2)
        .map_err(|_| GpuError::shape("autotune::tune", format!("an L2 of {l2} bytes")))?;
    let groups = mcol_rotations(weights, COLD_L2 * l2)?;
    let mut picks = Vec::new();
    if let (Some(k), Some(rows)) = (
        groups.keys().map(|s| s.1).max(),
        groups.keys().map(|s| s.0).max(),
    ) {
        let stream = gpu.stream();
        let device = crate::host::costs::card_key(gpu.context())?;
        let x = DeviceBuffer::from_host(stream, &activations(STAGED_MAX_COLS * k))?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, STAGED_MAX_COLS * rows)?;
        for (&(rows, k), planes) in &groups {
            let ws: Vec<E4m3Weight<'_>> = planes
                .iter()
                .map(|(qs, e)| E4m3Weight {
                    qs,
                    e,
                    scales: E4m3Scales::Mxfp8,
                })
                .collect();
            for m in STAGED_COLS..=STAGED_MAX_COLS {
                let sig = McolSig { rows, k, m };
                let ways = Mcol::racing(m);
                let keep = |v: &[f64]| {
                    v[0].fract() == 0.0
                        && (0.0..ways.len() as f64).contains(&v[0])
                        && v[1..].iter().all(|t| t.is_finite() && *t > 0.0)
                };
                let id = key(&device, build, MCOL_KERNEL, &sig.to_string());
                let (v, from) =
                    store.get_or_retime(&id, &NAMES, keep, || race(gpu, &ws, sig, (&x, &mut y)))?;
                picks.push(Pick {
                    sig,
                    way: ways[v[0] as usize],
                    from,
                    rule_us: v[1],
                    way_us: v[2],
                });
            }
        }
    }
    Ok(Tuned {
        picks,
        wall: t0.elapsed(),
    })
}

/// One weight a race reads: the code and scale planes of a resident MXFP8
/// weight's rows.
type Planes<'w> = (View<'w, u32>, View<'w, u32>);

/// The planes each signature's race rotates over, by `(rows, k)`, over the
/// resident MXFP8 weights whose `k` the staged entries read: [`rotations`]
/// of `cold` bytes.
fn mcol_rotations(
    weights: &Weights,
    cold: usize,
) -> Result<BTreeMap<(usize, usize), Vec<Planes<'_>>>, GpuError> {
    let resident: Vec<(&str, &DeviceTensor<u32>, &DeviceTensor<u32>, usize)> = weights
        .names()
        .filter_map(|name| match weights.get(name) {
            Some(DevWeight::Mxfp8 { qs, e, k }) if k.is_multiple_of(STAGED_K) => {
                Some((name, qs, e, *k))
            }
            _ => None,
        })
        .collect();
    let shapes: Vec<(usize, usize)> = resident
        .iter()
        .map(|&(_, qs, _, k)| (qs.rows(), k))
        .collect();
    rotations(&shapes, cold)
        .into_iter()
        .map(|((rows, k), slots)| {
            let planes = slots
                .into_iter()
                .map(|(i, at)| {
                    let (name, qs, e, _) = resident[i];
                    let part = (name, Some((at, rows)));
                    Ok((
                        view("autotune::tune", part, qs)?,
                        view("autotune::tune", part, e)?,
                    ))
                })
                .collect::<Result<_, GpuError>>()?;
            Ok(((rows, k), planes))
        })
        .collect()
}

/// Each signature's rotation (module doc) over `shapes`, each weight's rows
/// and `k` in name order, as `(index in shapes, first row)` a launch of the
/// signature's rows: the signature's own weights, then the whole windows of
/// its rows the other weights of its `k` hold, each weight's from its first
/// row, until the launches read `cold` bytes; one at least, all there are
/// when they read less.
fn rotations(
    shapes: &[(usize, usize)],
    cold: usize,
) -> BTreeMap<(usize, usize), Vec<(usize, usize)>> {
    let mut out = BTreeMap::new();
    for &(rows, k) in shapes {
        out.entry((rows, k)).or_insert_with(|| {
            let own = shapes
                .iter()
                .enumerate()
                .filter(|&(_, &s)| s == (rows, k))
                .map(|(i, _)| (i, 0));
            let windows = shapes
                .iter()
                .enumerate()
                .filter(|&(_, &(r, kk))| kk == k && r != rows)
                .flat_map(|(i, &(r, _))| (0..r / rows).map(move |j| (i, j * rows)));
            let launches = cold.div_ceil(weight_bytes(rows, k)).max(1);
            own.chain(windows).take(launches).collect()
        });
    }
    out
}

/// An MXFP8 weight's bytes: a record a block, [`Format::code_bytes`] values.
fn weight_bytes(rows: usize, k: usize) -> usize {
    rows * k / Format::Mxfp8.code_bytes() * Format::Mxfp8.record_bytes()
}

/// `n` activations in about ±1, none zero, so no value-dependent path is
/// skipped.
fn activations(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / 8_388_608.0 - 1.0 + 1e-3)
        .collect()
}

/// One race (module doc): a graph a way of `sig` over the rotation `ws`,
/// each warmed, then [`REPEATS`] samples a way, the ways interleaved;
/// `[index of the pick in Mcol::racing, rule's median, pick's median]`.
fn race(
    gpu: &Gpu,
    ws: &[E4m3Weight<'_>],
    sig: McolSig,
    (x, y): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
) -> Result<Vec<f64>, GpuError> {
    let ways = Mcol::racing(sig.m);
    let stream = gpu.stream();
    let q8 = gpu.q8f32();
    let repeat = GRAPH_BYTES
        .div_ceil(ws.len() * weight_bytes(sig.rows, sig.k))
        .min(GRAPH_LAUNCHES.div_ceil(ws.len()))
        .max(1);
    let launches = repeat * ws.len();
    let mut graphs = Vec::with_capacity(ways.len());
    for &way in ways {
        graphs.push(gpu.capture(|st| {
            ws.iter().cycle().take(launches).try_for_each(|&w| {
                q8.enqueue_e4m3_gemv_way(
                    st,
                    E4m3GemvArgs {
                        w,
                        x,
                        m: sig.m,
                        out: GemvOut::TokenMajor,
                        y: &mut *y,
                    },
                    way,
                )
            })
        })?);
    }
    let start = new_event(gpu)?;
    let end = new_event(gpu)?;
    let time = |g: &Graph, replays: usize| -> Result<f64, GpuError> {
        start.record(stream)?;
        for _ in 0..replays {
            g.launch(stream)?;
        }
        end.record(stream)?;
        end.synchronize()?;
        let us = f64::from(start.elapsed_ms(&end)?) * 1e3;
        Ok(us / (replays * launches) as f64)
    };
    for g in &graphs {
        time(g, 1)?;
    }
    let graph_us = time(&graphs[0], 1)? * launches as f64;
    let replays = (SAMPLE_US / graph_us).ceil().clamp(1.0, 256.0) as usize;
    let mut samples = vec![Vec::with_capacity(REPEATS); ways.len()];
    for r in 0..REPEATS {
        for j in 0..ways.len() {
            let i = (j + r) % ways.len();
            samples[i].push(time(&graphs[i], replays)?);
        }
    }
    let t: Vec<Times> = samples
        .into_iter()
        .map(|s| s.try_into().expect("REPEATS samples a way"))
        .collect();
    let (i, us) = winner(&t);
    Ok(vec![i as f64, median(&t[0]), us])
}

/// A timing event on `gpu`'s context.
fn new_event(gpu: &Gpu) -> Result<cuda_core::CudaEvent, GpuError> {
    // A timed event: `new_event(None)` makes a sync-only one, whose
    // `elapsed_ms` the driver refuses.
    Ok(gpu
        .context()
        .new_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::q8f32::Q8F32Kernels;
    use crate::tensor::DeviceTensor;

    /// The table of a load that measured `ways`.
    fn table(ways: &[(McolSig, Mcol)]) -> Picks {
        Tuned {
            picks: ways
                .iter()
                .map(|&(sig, way)| Pick {
                    sig,
                    way,
                    from: Source::Measured,
                    rule_us: 2.0,
                    way_us: 1.0,
                })
                .collect(),
            wall: Duration::ZERO,
        }
        .table()
    }

    /// An empty table runs the rule at every signature; a table runs each
    /// tuned signature's way and the rule elsewhere, a kept rule taking no
    /// entry (mutant: the pick of another signature).
    #[test]
    fn a_table_runs_its_picks_and_the_rule_elsewhere() {
        let sig = |m| McolSig {
            rows: 640,
            k: 2560,
            m,
        };
        assert_eq!(Picks::default().mcol(sig(5)), Mcol::Rule);
        let t = table(&[
            (sig(4), Mcol::Rule),
            (sig(5), Mcol::Plain),
            (sig(7), Mcol::Split),
        ]);
        assert_eq!(t.mcol.len(), 2);
        assert_eq!(
            [4, 5, 6, 7].map(|m| t.mcol(sig(m))),
            [Mcol::Rule, Mcol::Plain, Mcol::Rule, Mcol::Split]
        );
        assert_eq!(Mcol::racing(3), &[] as &[Mcol]);
        assert_eq!(Mcol::racing(4), &[Mcol::Rule, Mcol::Plain]);
        assert_eq!(Mcol::racing(8), &Mcol::ALL);
    }

    /// A rotation reads its signature's weights, then the whole windows of
    /// its rows the other weights of its `k` hold, never another `k`'s,
    /// until it reads the cold bytes; one weight at least, all there are
    /// short of them (mutants: a window past a weight's rows, another `k`'s
    /// weight, the windows before the signature's own, no stop at the cold
    /// bytes, none at zero).
    #[test]
    fn a_rotation_reads_cold_bytes_of_its_rows_from_its_k() {
        let one = weight_bytes(96, 2560);
        let shapes = [(96, 2560), (256, 2560), (96, 6144), (96, 2560), (100, 2560)];
        let r = rotations(&shapes, 5 * one);
        assert_eq!(r.len(), 4);
        assert_eq!(r[&(96, 2560)], [(0, 0), (3, 0), (1, 0), (1, 96), (4, 0)]);
        assert_eq!(r[&(96, 6144)], [(2, 0)]);
        assert_eq!(r[&(256, 2560)], [(1, 0)]);
        assert_eq!(
            rotations(&shapes, 3 * one)[&(96, 2560)],
            [(0, 0), (3, 0), (1, 0)]
        );
        assert_eq!(rotations(&shapes, 0)[&(96, 2560)], [(0, 0)]);
        assert_eq!(
            rotations(&shapes, 1 << 40)[&(100, 2560)],
            [(4, 0), (1, 0), (1, 100)]
        );
    }

    /// An installed table steers `enqueue_e4m3_gemv`'s launch: a signature
    /// it names runs that way (plain, one `_mcol` node; split, the staged
    /// four then `_mcol`, two nodes), every other the rule (mutants: the
    /// launcher's table lookup dropped, its signature built from another
    /// field).
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_installed_picks_steer_the_launch() {
        let gpu = Gpu::new().expect("a Gpu on device 0");
        let q8 = Q8F32Kernels::load(gpu.context()).expect("the q8f32 kernels");
        let (rows, k) = (64, 2 * STAGED_K);
        let stream = gpu.stream();
        let qs = DeviceTensor::<u32>::zeroed(stream, rows, k / 4).expect("the codes");
        let e = DeviceTensor::<u32>::zeroed(stream, rows, k / 128).expect("the scales");
        let x = DeviceBuffer::<f32>::zeroed(stream, STAGED_MAX_COLS * k).expect("x");
        let mut y = DeviceBuffer::<f32>::zeroed(stream, STAGED_MAX_COLS * rows).expect("y");
        let sig = |m| McolSig { rows, k, m };
        q8.install(table(&[(sig(5), Mcol::Plain), (sig(7), Mcol::Split)]))
            .expect("the first install");
        let mut nodes = |m| -> Vec<String> {
            let w = E4m3Weight {
                qs: &qs,
                e: &e,
                scales: E4m3Scales::Mxfp8,
            };
            let a = E4m3GemvArgs {
                w,
                x: &x,
                m,
                out: GemvOut::TokenMajor,
                y: &mut y,
            };
            gpu.capture(|st| q8.enqueue_e4m3_gemv(st, a))
                .expect("the capture")
                .nodes()
                .expect("the node list")
                .into_iter()
                .map(|n| n.kernel.expect("a kernel node").name)
                .collect()
        };
        assert_eq!(nodes(5), ["mxfp8_gemv_mcol"]);
        assert_eq!(nodes(6), ["mxfp8_gemv_mcol_staged_wide"]);
        assert_eq!(nodes(7), ["mxfp8_gemv_mcol_staged", "mxfp8_gemv_mcol"]);
    }
}
