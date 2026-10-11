//! What the image-tower gates that hold a tower to llama.cpp mainline's mtmd taps (the clefvis sets)
//! have in common: the distance between a tap and the reference's f32, the controls a band is
//! drawn from (the reference's CPU twin, the bf16 rounding floor, the network's sensitivity to a
//! last-bit nudge of its input), the line a compared tap prints, the tap collector, ggml's vision
//! RoPE cache, the image files, the encoder file's host copies, and the two structure lines of an
//! encode (eager twice, reuse after the largest image). Shared by `gate_qwen3vl_tower` and
//! `gate_glm5v_tower`; each gate owns its band multiple, its tap names and its host rules.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use bloomery_gpu_gates::{GateError, verdict};
use bloomery_gpu_vision::chain::TapSink;
use bloomery_gpu_vision::{bf16_f32, f32_bf16};
use gguf::{GgmlType, Gguf};
use vision::Rgb8;

/// A sensitivity run flips the last bit of every `NOISE_EVERY`-th input value (a 1-ulp noise of
/// the order of rounding the patches to bf16).
const NOISE_EVERY: usize = 16;

// ------------------------------------------------------------ distances

#[derive(Clone, Copy)]
pub struct Dist {
    pub max_rel: f64,
    pub rms_rel: f64,
}

/// `ours` (bf16 bits) against `refv` (f32): `max|Δ| / max|ref|` and `rms(Δ) / rms(ref)`.
pub fn dist(ours: &[u16], refv: &[f32]) -> Dist {
    dist_f32(&to_f32(ours), refv)
}

pub fn dist_f32(ours: &[f32], refv: &[f32]) -> Dist {
    let (mut max_d, mut max_r, mut sd, mut sr) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in ours.iter().zip(refv) {
        let (x, y) = (f64::from(a), f64::from(b));
        max_d = max_d.max((x - y).abs());
        max_r = max_r.max(y.abs());
        sd += (x - y) * (x - y);
        sr += y * y;
    }
    Dist {
        max_rel: max_d / max_r,
        rms_rel: (sd / sr).sqrt(),
    }
}

/// The reference rounded to bf16 against itself: the floor of any bf16 tower's distance.
pub fn floor(refv: &[f32]) -> f64 {
    let r: Vec<f32> = refv.iter().map(|&v| bf16_f32(f32_bf16(v))).collect();
    dist_f32(&r, refv).rms_rel
}

pub fn to_f32(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&b| bf16_f32(b)).collect()
}

pub fn to_bf16(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| f32_bf16(x)).collect()
}

/// `x` with the last bit of every [`NOISE_EVERY`]-th value flipped, in place.
pub fn nudge(x: &mut [u16]) {
    for (i, v) in x.iter_mut().enumerate() {
        if i % NOISE_EVERY == 0 {
            *v ^= 1;
        }
    }
}

/// [`nudge`] of a copy of `x`: the input of a step's sensitivity run.
pub fn nudged(x: &[u16]) -> Vec<u16> {
    let mut v = x.to_vec();
    nudge(&mut v);
    v
}

/// The network's sensitivity at a tap: the distance of the nudged run's tensor from the run's own.
pub fn sens(nudged_run: &[u16], run: &[u16]) -> f64 {
    dist(nudged_run, &to_f32(run)).rms_rel
}

/// One compared tap's line: `rms(ours − ref) / rms(ref)` against `k ×` the larger of its controls.
#[allow(
    clippy::too_many_arguments,
    reason = "one tap's names, values and three controls, as the line prints them"
)]
pub fn band_check(
    k: f64,
    stem: &str,
    tap: &str,
    ours: &[u16],
    refv: &[f32],
    twin: f64,
    floor: f64,
    sens: Option<f64>,
) -> bool {
    if ours.len() != refv.len() {
        println!(
            "tap {stem:<14} {tap:<20} FAIL: {} values, the set has {}",
            ours.len(),
            refv.len()
        );
        return false;
    }
    let d = dist(ours, refv);
    let ctl = twin.max(floor).max(sens.unwrap_or(0.0));
    let pass = d.rms_rel <= k * ctl;
    println!(
        "tap {stem:<14} {tap:<20} max|rel| {:.3e}  rms rel {:.3e}  controls twin {twin:.2e} floor {floor:.2e} sens {}  ratio {:.2} (at most {k})  {}",
        d.max_rel,
        d.rms_rel,
        sens.map_or("-".to_string(), |s| format!("{s:.2e}")),
        d.rms_rel / ctl,
        verdict(pass)
    );
    pass
}

// ------------------------------------------------------------ taps

/// Collects the named tensors of one encode.
pub struct Collect {
    want: HashSet<String>,
    pub got: HashMap<String, Vec<u16>>,
}

impl Collect {
    pub fn new(want: HashSet<String>) -> Collect {
        Collect {
            want,
            got: HashMap::new(),
        }
    }

    /// A sink that asks for the same names and has taken none.
    pub fn again(&self) -> Collect {
        Collect::new(self.want.clone())
    }
}

impl TapSink for Collect {
    fn wants(&self, name: &str) -> bool {
        self.want.contains(name)
    }
    fn take(&mut self, name: &str, _cols: usize, bits: Vec<u16>) {
        self.got.insert(name.to_string(), bits);
    }
}

// ------------------------------------------------------------ independent host rules

/// ggml's vision RoPE cache of one patch `(y, x)`, from the loop of `ggml_rope_cache_init` /
/// `ggml_mrope_cache_init` (vision mode, `pairs` rotated pairs a head, the angles of the first half
/// of them the row `y`, of the second half the column `x`): `theta_scale = 10000^(−2/pairs)`; the
/// angle of a pair is the position times `theta_scale^k`, formed by multiplying the running angle
/// by `theta_scale` after every pair and started again at the second half's own position; `cos`
/// and `sin` of the f32 angle, correctly rounded.
fn ggml_cache(y: usize, x: usize, pairs: usize) -> Vec<f32> {
    let section = pairs / 2;
    let theta_scale = f64::from(10_000.0f32).powf(f64::from(-2.0f32 / pairs as f32)) as f32;
    let (mut theta_t, mut theta_h) = (y as f32, x as f32);
    let mut cache = Vec::with_capacity(2 * pairs);
    for pair in 0..pairs {
        if pair == section {
            theta_h = x as f32;
        }
        let theta = if pair < section { theta_t } else { theta_h };
        cache.push(f64::from(theta).cos() as f32);
        cache.push(f64::from(theta).sin() as f32);
        theta_t *= theta_scale;
        theta_h *= theta_scale;
    }
    cache
}

/// The RoPE table of an `n_h × n_w` grid in merge order (groups of 2×2 in raster order, the group's
/// patches in raster order inside it), row by row from [`ggml_cache`].
pub fn ggml_table(n_h: usize, n_w: usize, pairs: usize) -> Vec<f32> {
    let mut cs = Vec::with_capacity(n_h * n_w * 2 * pairs);
    for gy in 0..n_h / 2 {
        for gx in 0..n_w / 2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    cs.extend(ggml_cache(2 * gy + dy, 2 * gx + dx, pairs));
                }
            }
        }
    }
    cs
}

// ------------------------------------------------------------ files

/// The PNG of a test image, `tools/ref/clefvis/images/<name>.png`.
pub fn read_image(name: &str) -> Result<Rgb8, GateError> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/ref/clefvis/images");
    let bytes =
        std::fs::read(dir.join(format!("{name}.png"))).map_err(|e| format!("{name}: {e}"))?;
    Ok(Rgb8::from_png(&bytes)?)
}

/// The host copies of an encoder file's tensors: a matrix as the bf16 bits the card holds (an f32
/// or f16 tensor narrows; the tower's load refuses one that does not round-trip), a vector as f32.
pub struct HostFile<'a>(pub &'a Gguf);

impl HostFile<'_> {
    fn bytes(&self, name: &str) -> Result<(&[u8], GgmlType), GateError> {
        let t = self
            .0
            .find(name)
            .ok_or_else(|| format!("{name} not in the encoder file"))?;
        Ok((self.0.data(t)?, t.ty))
    }

    pub fn bf16(&self, name: String) -> Result<Vec<u16>, GateError> {
        let (bytes, ty) = self.bytes(&name)?;
        Ok(match ty {
            GgmlType::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32_bf16(f32::from_le_bytes(*c)))
                .collect(),
            GgmlType::F16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32_bf16(gguf::quant::half_to_f32(u16::from_le_bytes(*c))))
                .collect(),
            _ => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes(*c))
                .collect(),
        })
    }

    pub fn f32s(&self, name: String) -> Result<Vec<f32>, GateError> {
        Ok(self
            .bytes(&name)?
            .0
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }
}

// ------------------------------------------------------------ structure

/// An image's two plain encodes bit-identical to each other and to the tapped run, and the launches
/// of both the plain and the tapped run equal to the tower's own count. `first` is the plain
/// encode's rows and launches.
pub fn eager_twice(
    name: &str,
    tapped: (&[u16], usize),
    first: &(Vec<u16>, usize),
    second: &[u16],
    want_launches: usize,
) -> bool {
    let twice = first.0 == second && first.0 == tapped.0;
    let launches = first.1 == want_launches && tapped.1 == want_launches;
    println!(
        "image {name}: eager twice bit-identical and equal to the tapped run: {}; launches {} (want {want_launches}): {}",
        verdict(twice),
        first.1,
        verdict(launches)
    );
    twice && launches
}

/// An encode does not depend on the larger image encoded before it in the reused buffers: `again`
/// (the small image after the largest) is bit-identical to `first` (the small image's first
/// encode), and the largest's rows are `tokens` of `out_dim`.
pub fn reuse_after_largest(
    (small, largest): (&str, &str),
    (tokens, max_tokens, out_dim): (usize, usize, usize),
    (first, big, again): (&[u16], &[u16], &[u16]),
) -> bool {
    let pass = again == first && big.len() == tokens * out_dim;
    println!(
        "reuse {small} after {largest} ({tokens} of {max_tokens} tokens): bit-identical to its first encode: {}",
        verdict(pass)
    );
    pass
}
