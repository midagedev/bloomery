//! The weight rules: each type's scale and code distribution, and the
//! chunked, seed-keyed stream that writes a tensor's bytes.

use std::fmt;

use gguf::GgmlType;
use gguf::iq_tables::{IQ1S_GRID, IQ2XXS_GRID, KVALUES_IQ4NL};
use gguf::quant::{KVALUES_MXFP4, f32_to_f16_bits, half_to_f32, ue4m3_to_f32_half};

use super::FixtureError;

/// The smallest normal f16, `2^-14`.
const F16_MIN_NORMAL: f32 = 1.0 / 16384.0;
/// The largest finite f16.
const F16_MAX: f32 = 65504.0;

/// The range every generated block's `d` (and `dmin`) lies in. The only
/// range a rule's scales must keep to is the normal f16 values; inside them
/// the window is a family's choice ([`FixtureSpec::window`](super::FixtureSpec::window)).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Window {
    lo: f32,
    hi: f32,
}

impl Window {
    /// `[lo, hi]`; refused by name unless it lies inside the normal f16
    /// values `[2^-14, 65504]`.
    pub fn new(lo: f32, hi: f32) -> Result<Window, FixtureError> {
        if lo.is_finite() && hi.is_finite() && F16_MIN_NORMAL <= lo && lo <= hi && hi <= F16_MAX {
            Ok(Window { lo, hi })
        } else {
            Err(FixtureError::BadWindow { lo, hi })
        }
    }

    pub fn lo(&self) -> f32 {
        self.lo
    }

    pub fn hi(&self) -> f32 {
        self.hi
    }

    /// Whether `bits` is a normal f16 in the window.
    pub fn holds(&self, bits: u16) -> bool {
        normal_f16(bits) && (self.lo..=self.hi).contains(&half_to_f32(bits))
    }
}

impl fmt::Display for Window {
    /// As powers of two, for an error line: `[2^-13, 2^-10]`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[2^{}, 2^{}]", self.lo.log2(), self.hi.log2())
    }
}

/// Whether `bits` is a normal f16: not zero, subnormal, infinite or NaN.
fn normal_f16(bits: u16) -> bool {
    let exp = (bits >> 10) & 0x1f;
    exp != 0 && exp != 0x1f
}

/// A chunk of a tensor's stream is about this many bytes: whole blocks.
pub const CHUNK_TARGET: usize = 1 << 20;

/// The scale codes a block draws from: every value `v` of the type's field
/// values with `lo ≤ |v| ≤ hi`, uniformly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Band {
    pub lo: i32,
    pub hi: i32,
    values: Vec<i32>,
}

impl Band {
    /// The band of `field` (ascending) at `lo ≤ |v| ≤ hi`.
    fn of(field: &[i32], lo: i32, hi: i32) -> Band {
        let values = field
            .iter()
            .copied()
            .filter(|v| (lo..=hi).contains(&v.abs()))
            .collect();
        Band { lo, hi, values }
    }

    /// Whether `v` is one of the band's codes.
    pub fn holds(&self, v: i32) -> bool {
        self.values.binary_search(&v).is_ok()
    }

    /// Whether the band holds a negative code and a positive one.
    fn both_signs(&self) -> bool {
        self.values.first().is_some_and(|&v| v < 0) && self.values.last().is_some_and(|&v| v > 0)
    }

    fn mean_sq(&self) -> f64 {
        let s: f64 = self.values.iter().map(|&v| f64::from(v * v)).sum();
        s / self.values.len() as f64
    }

    fn draw(&self, r: u32) -> i32 {
        self.values[((u64::from(r) * self.values.len() as u64) >> 32) as usize]
    }
}

/// How a tensor's bytes are made. Every quantized rule holds one `d` (and
/// `dmin`) as f16 bits for the whole tensor; the codes and the per-sub-block
/// scales are random.
#[derive(Clone, Debug, PartialEq)]
pub enum Rule {
    /// `d·s·q`: `s` = the 6-bit scale − 32 from `band`, `q` uniform in −4..=3.
    Q3K { d: u16, band: Band },
    /// `d·sc·q − dmin·m`: `sc` from `band`, `m = sc`, `dmin = 7.5·d`, `q`
    /// uniform in 0..=15 — zero mean.
    Q4K { d: u16, dmin: u16, band: Band },
    /// `d·sc·q − dmin·m`: `sc` from `band` (at most 31), `m = 2·sc`,
    /// `dmin = 7.75·d`, `q` uniform in 0..=31 — zero mean.
    Q5K { d: u16, dmin: u16, band: Band },
    /// `d·sc·(q − 32)`: `sc` (i8) from `band`, `q` uniform in 0..=63.
    Q6K { d: u16, band: Band },
    /// `d·q`: `q` (i8) from `band`.
    Q8_0 { d: u16, band: Band },
    /// The legacy 32-value 5-bit blocks, `q` uniform in 0..=31: with `m`
    /// (Q5_1) `q·d + m`, `m = −15.5·d` — zero mean; without (Q5_0) `d·(q − 16)`
    /// — mean `−d/2`, no offset to zero it.
    Q5Legacy { d: u16, m: Option<u16> },
    /// `d·k`: `k` from `band`, a band of `kvalues_iq4nl`, stored as its index.
    Iq4Nl { d: u16, band: Band },
    /// `2^(E−128)·kvalues[c]`: `E` is `e_lo` with probability `p_lo / 2^32`,
    /// else `e_lo + 1`; `c` uniform over the 16 codes.
    Mxfp4 { e_lo: u8, p_lo: u64 },
    /// `d·(2s + 1)/8·g·(±1)`: per 32-value sub-block the scale `s` (4 bits) has
    /// `2s + 1` from `band`; each group of eight takes an index uniform over
    /// `IQ2XXS_GRID`'s 256 entries (`g` its bytes) and seven uniform sign bits,
    /// the eighth sign the parity that makes the group's negatives even.
    Iq2Xxs { d: u16, band: Band },
    /// `d·(2s + 1)·(g + δ)`: per 32-value sub-block the scale `s` (3 bits) has
    /// `2s + 1` from `band` and the delta `δ` is ±1/8 by one sign bit; each
    /// group of eight takes an 11-bit index uniform over `IQ1S_GRID`'s 2,048
    /// entries (`g` its signed bytes, −1, 0 or 1).
    Iq1S { d: u16, band: Band },
    /// Each of a block's four 16-value sub-blocks takes the UE4M3
    /// scale byte `s_lo` with probability `p_lo / 2^32`, else `s_lo + 1`; `c`
    /// uniform over the 16 codes. The block's dequant is `d·kvalues_fp4[c]`
    /// without the tensor's scale [`NVFP4_TENSOR_SCALE`] `S`, which the
    /// engine multiplies by.
    Nvfp4 { s_lo: u8, p_lo: u64 },
    /// Uniform in `[−half_width, half_width)`, rounded to the type.
    Uniform { ty: FloatTy, half_width: f32 },
    /// Every element `value` (F32).
    Const { value: f32 },
}

/// The float types a [`Rule::Uniform`] rounds to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatTy {
    F32,
    F16,
    BF16,
}

impl FloatTy {
    /// The ggml type it is.
    pub fn ggml(self) -> GgmlType {
        match self {
            FloatTy::F32 => GgmlType::F32,
            FloatTy::F16 => GgmlType::F16,
            FloatTy::BF16 => GgmlType::BF16,
        }
    }
}

impl Rule {
    /// One line: the scale, the band and the code distribution.
    pub fn describe(&self) -> String {
        let band = |b: &Band| {
            format!(
                "|scale| in [{}, {}] ({} codes, E[s^2] {:.2})",
                b.lo,
                b.hi,
                b.values.len(),
                b.mean_sq()
            )
        };
        match self {
            Rule::Q3K { d, band: b } => {
                format!("d {:.4e}, {}, q in -4..=3", half_to_f32(*d), band(b))
            }
            Rule::Q4K { d, dmin, band: b } => {
                format!(
                    "d {:.4e} dmin {:.4e}, {}, m = sc, q in 0..=15",
                    half_to_f32(*d),
                    half_to_f32(*dmin),
                    band(b)
                )
            }
            Rule::Q5K { d, dmin, band: b } => {
                format!(
                    "d {:.4e} dmin {:.4e}, {}, m = 2 sc, q in 0..=31",
                    half_to_f32(*d),
                    half_to_f32(*dmin),
                    band(b)
                )
            }
            Rule::Q6K { d, band: b } => {
                format!("d {:.4e}, {}, q in 0..=63", half_to_f32(*d), band(b))
            }
            Rule::Q8_0 { d, band: b } => format!("d {:.4e}, codes {}", half_to_f32(*d), band(b)),
            Rule::Q5Legacy { d, m: Some(m) } => format!(
                "d {:.4e} m {:.4e}, q in 0..=31",
                half_to_f32(*d),
                half_to_f32(*m)
            ),
            Rule::Q5Legacy { d, m: None } => {
                format!("d {:.4e}, value d·(q − 16), q in 0..=31", half_to_f32(*d))
            }
            Rule::Iq4Nl { d, band: b } => format!(
                "d {:.4e}, kvalues |k| in [{}, {}] ({} codes, E[k^2] {:.2})",
                half_to_f32(*d),
                b.lo,
                b.hi,
                b.values.len(),
                b.mean_sq()
            ),
            Rule::Iq2Xxs { d, band: b } => format!(
                "d {:.4e}, 2s+1 in [{}, {}] ({} codes, E[(2s+1)^2] {:.2}), grid index and 7 sign bits uniform",
                half_to_f32(*d),
                b.lo,
                b.hi,
                b.values.len(),
                b.mean_sq()
            ),
            Rule::Iq1S { d, band: b } => format!(
                "d {:.4e}, 2s+1 in [{}, {}] ({} codes, E[(2s+1)^2] {:.2}), grid index and delta sign uniform",
                half_to_f32(*d),
                b.lo,
                b.hi,
                b.values.len(),
                b.mean_sq()
            ),
            Rule::Mxfp4 { e_lo, p_lo } => format!(
                "E8M0 {e_lo} (d 2^{}) with p {:.4}, else {} (d 2^{})",
                i32::from(*e_lo) - 128,
                *p_lo as f64 / 4_294_967_296.0,
                e_lo + 1,
                i32::from(*e_lo) - 127
            ),
            Rule::Nvfp4 { s_lo, p_lo } => {
                let d = |b: u8| ue4m3_to_f32_half(b).expect("a byte below 0x7F is a scale");
                format!(
                    "UE4M3 scale {s_lo:#04x} (d {:.4e}) with p {:.4}, else {:#04x} (d {:.4e}), times the tensor scale {NVFP4_TENSOR_SCALE:.4e}; codes uniform over the 16 kvalues",
                    d(*s_lo),
                    *p_lo as f64 / 4_294_967_296.0,
                    s_lo + 1,
                    d(s_lo + 1)
                )
            }
            Rule::Uniform { ty, half_width } => {
                format!("{} uniform in ±{half_width:.4e}", ty.ggml())
            }
            Rule::Const { value } => format!("constant {value}"),
        }
    }
}

/// `E[q²]` of each quantized type's code distribution, and the ratio of its
/// minimum to its scale.
const Q3K_CODE_SQ: f64 = 5.5;
const Q4K_CODE_VAR: f64 = 21.25;
pub(super) const Q4K_MIN_PER_SCALE: i32 = 1;
const Q4K_DMIN_PER_D: f32 = 7.5;
const Q5K_CODE_VAR: f64 = 85.25;
pub(super) const Q5K_MIN_PER_SCALE: i32 = 2;
const Q5K_DMIN_PER_D: f32 = 7.75;
const Q6K_CODE_SQ: f64 = 341.5;
/// The legacy 5-bit blocks, Q5_0 (`block_q5_0`: f16 `d`, 32 five-bit codes,
/// value `d·(q − 16)`) and Q5_1 (`block_q5_1`: f16 `d`, f16 `m`, the same
/// codes, value `q·d + m`), differ in the offset alone, so one rule makes both
/// and `m` is its data: `q` uniform in 0..=31 has mean 15.5 and variance
/// 85.25. Q5_1's `m = −15.5·d` zeroes the mean, so `E[w²] = 85.25·d²`; Q5_0 has
/// no `m`, its values have mean `−d/2` (0.054 of the RMS) and
/// `E[w²] = (85.25 + 0.25)·d² = 85.5·d²`. `m` is an offset, not a scale:
/// `|m| / d = 15.5`, so a window narrower than that factor could never hold
/// both; only `d` is held to the window, and `m` to a normal f16. No band is
/// open (the block has no sub-block scales), so `d = σ/√E[w²/d²]`: a
/// [2^-13, 2^-10] window takes Q5_1 at K in about [12,300, 787,000], and
/// K = 640 needs `d = 2^-7.87`; a [2^-14, 2^-6] window takes Q5_0 at K in
/// about [48, 3.1 million].
const Q5_CODE_VAR: f64 = 85.25;
const Q5_MID: f32 = 15.5;
/// Q5_0's mean, `15.5 − 16`, squared: what its `E[w²]` holds beyond the
/// variance.
const Q5_0_MEAN_SQ: f64 = 0.25;
/// Q5_0's offset in codes: its value is `d·(q − 16)`.
pub(super) const Q5_0_OFFSET: f32 = 16.0;
/// IQ4_NL (`block_iq4_nl`: f16 `d`, 32 four-bit codes, value
/// `d·kvalues_iq4nl[c]`): the codes are table values, so a band of them is the
/// code distribution and its own `E[k²]` the code variance, as Q8_0's codes
/// are. Over all 16 codes `Σk² = 72,768`, so `E[k²] = 4,548` and `d = σ/67.4`
/// fits a [2^-13, 2^-10] window at K in about [231, 14,750]; `E[k] = −94/16 = −5.875`, a
/// mean of 0.09 of the RMS (Q3_K's −4..=3 has 0.21). A band of the larger |k|
/// reaches a smaller K, a band of the smaller a larger one; a band holds codes
/// of both signs, so no band is one constant.
const IQ4NL_CODE_SQ: f64 = 1.0;
/// IQ2_XXS (`block_iq2_xxs`: f16 `d`, per 32 values four grid-index bytes and a
/// u32 of 4×7 sign bits over a 4-bit scale `s`; value `d·(0.5 + s)/4·g·(±1)` with
/// `g` a byte of [`IQ2XXS_GRID`]'s entry): `(0.5 + s)/4 = (2s + 1)/8`, so a band
/// of the odd integers `2s + 1` is the scale distribution, as the K-quants' bands
/// are. The grid index is uniform over the 256 entries and the signs are
/// independent of it, so `E[w²] = d²·E[(2s+1)²]·E[g²]/64`; `E[g²]` is the mean of
/// the squared grid bytes over every entry, read from the table.
const IQ2XXS_SCALE_DEN_SQ: f64 = 64.0;
/// IQ2_XXS's sub-block scale codes `2s + 1`, `s` in 0..=15.
fn iq2xxs_scale_codes() -> Vec<i32> {
    (0..16).map(|s| 2 * s + 1).collect()
}
/// `E[g²]/64`: the grid bytes' mean square over the scale denominator.
fn iq2xxs_code_sq() -> f64 {
    let sum: f64 = IQ2XXS_GRID
        .iter()
        .flat_map(|g| g.to_le_bytes())
        .map(|b| f64::from(b) * f64::from(b))
        .sum();
    sum / (IQ2XXS_GRID.len() * 8) as f64 / IQ2XXS_SCALE_DEN_SQ
}
/// IQ1_S (`block_iq1_s`: f16 `d`, 32 grid-index bytes, eight u16 `qh` of three
/// high index bits a group, a 3-bit scale `s` and a delta sign a sub-block of 32
/// values; value `d·(2s + 1)·(g + δ)` with `g` a signed byte of
/// [`IQ1S_GRID`]'s entry and `δ = ±IQ1S_DELTA`): a band of the odd integers
/// `2s + 1` is the scale distribution. The grid index (11 bits) is uniform over
/// the 2,048 entries and the sign of `δ` is independent of it, so
/// `E[w²] = d²·E[(2s+1)²]·(E[g²] + δ²)`; `E[g²]` is the mean of the squared grid
/// bytes over every entry, read from the table.
const IQ1S_DELTA_SQ: f64 = 0.125 * 0.125;
/// IQ1_S's sub-block scale codes `2s + 1`, `s` in 0..=7.
fn iq1s_scale_codes() -> Vec<i32> {
    (0..8).map(|s| 2 * s + 1).collect()
}
/// `E[g²] + δ²`: the grid bytes' mean square and the delta's square.
fn iq1s_code_sq() -> f64 {
    let sum: f64 = IQ1S_GRID
        .iter()
        .flat_map(|g| g.to_le_bytes())
        .map(|b| f64::from(b as i8) * f64::from(b as i8))
        .sum();
    sum / (IQ1S_GRID.len() * 8) as f64 + IQ1S_DELTA_SQ
}
/// NVFP4 (`block_nvfp4`: four UE4M3 sub-block scale bytes, 64 four-bit codes,
/// value `S·d·kvalues_fp4[c]`; [`ue4m3_to_f32_half`] decodes `d`) carries a
/// per-tensor f32 scale `S`, a tensor of its own (`<name>.scale`) that the
/// block's dequant leaves out. The fixture fixes `S` for every NVFP4 tensor,
/// and its family names each `.scale` tensor's constant as this value: a power
/// of two, near the middle of the real file's (1.3e-4 to 1.6e-3, median 4e-4
/// [measured]), so `S·d` is exact. The window holds the product `S·d`.
pub const NVFP4_TENSOR_SCALE: f32 = 1.0 / 2048.0;
/// The largest scale byte whose successor is still a scale: 0x7F is E4M3's NaN.
const UE4M3_LAST_PAIR: u8 = 0x7D;

fn f16(x: f64) -> u16 {
    f32_to_f16_bits(x as f32)
}

/// The widest band of `field` (ascending; ties: the lowest `lo`) that
/// `fits` with its `d`, where `d = σ / √(E[v²]·code_sq)`.
fn widest_band(
    field: &[i32],
    code_sq: f64,
    sigma: f64,
    fits: impl Fn(&Band, u16) -> bool,
) -> Option<(Band, u16)> {
    let top = field.iter().map(|v| v.abs()).max()?;
    for width in (0..=top).rev() {
        for lo in 0..=top - width {
            let band = Band::of(field, lo, lo + width);
            if band.values.is_empty() {
                continue;
            }
            let e = band.mean_sq();
            if e == 0.0 {
                continue;
            }
            let d = f16(sigma / (e * code_sq).sqrt());
            if fits(&band, d) {
                return Some((band, d));
            }
        }
    }
    None
}

/// The integers `min..=max`: a scale or code field.
fn field(min: i32, max: i32) -> Vec<i32> {
    (min..=max).collect()
}

/// The rule of a quantized or float matrix of type `ty` whose rows are `k`
/// values long: element standard deviation `1/√k`, every `d` (and `dmin`) in
/// `window`.
pub fn rule_for(name: &str, ty: GgmlType, k: u64, window: Window) -> Result<Rule, FixtureError> {
    let sigma = 1.0 / (k as f64).sqrt();
    let no_scale = |what| FixtureError::NoScale {
        name: name.to_string(),
        ty,
        k,
        what,
        window,
    };
    let in_window = |_: &Band, d: u16| window.holds(d);
    let with_min = |dmin_per_d: f32| {
        move |_: &Band, d: u16| {
            window.holds(d) && window.holds(f32_to_f16_bits(half_to_f32(d) * dmin_per_d))
        }
    };
    let dmin_of = |d: u16, per: f32| f32_to_f16_bits(half_to_f32(d) * per);
    let uniform = |ty| Rule::Uniform {
        ty,
        half_width: (3.0f64.sqrt() * sigma) as f32,
    };
    match ty {
        GgmlType::Q3_K => widest_band(&field(-32, 31), Q3K_CODE_SQ, sigma, in_window)
            .map(|(band, d)| Rule::Q3K { d, band })
            .ok_or_else(|| no_scale("band of 6-bit scales")),
        GgmlType::Q4_K => widest_band(&field(0, 63), Q4K_CODE_VAR, sigma, with_min(Q4K_DMIN_PER_D))
            .map(|(band, d)| Rule::Q4K {
                d,
                dmin: dmin_of(d, Q4K_DMIN_PER_D),
                band,
            })
            .ok_or_else(|| no_scale("band of 6-bit scales")),
        GgmlType::Q5_K => widest_band(&field(0, 31), Q5K_CODE_VAR, sigma, with_min(Q5K_DMIN_PER_D))
            .map(|(band, d)| Rule::Q5K {
                d,
                dmin: dmin_of(d, Q5K_DMIN_PER_D),
                band,
            })
            .ok_or_else(|| no_scale("band of 6-bit scales")),
        GgmlType::Q6_K => widest_band(&field(-128, 127), Q6K_CODE_SQ, sigma, in_window)
            .map(|(band, d)| Rule::Q6K { d, band })
            .ok_or_else(|| no_scale("band of i8 scales")),
        GgmlType::Q8_0 => widest_band(&field(-127, 127), 1.0, sigma, in_window)
            .map(|(band, d)| Rule::Q8_0 { d, band })
            .ok_or_else(|| no_scale("band of i8 codes")),
        GgmlType::Q5_0 | GgmlType::Q5_1 => {
            let has_min = ty == GgmlType::Q5_1;
            let code_sq = Q5_CODE_VAR + if has_min { 0.0 } else { Q5_0_MEAN_SQ };
            let d = f16(sigma / code_sq.sqrt());
            let m = has_min.then(|| f32_to_f16_bits(-Q5_MID * half_to_f32(d)));
            if window.holds(d) && m.is_none_or(normal_f16) {
                Ok(Rule::Q5Legacy { d, m })
            } else {
                Err(no_scale("d of uniform 5-bit codes"))
            }
        }
        GgmlType::IQ4_NL => {
            let kvalues: Vec<i32> = KVALUES_IQ4NL.iter().map(|&k| i32::from(k)).collect();
            widest_band(&kvalues, IQ4NL_CODE_SQ, sigma, |b, d| {
                b.both_signs() && window.holds(d)
            })
            .map(|(band, d)| Rule::Iq4Nl { d, band })
            .ok_or_else(|| no_scale("band of kvalue codes"))
        }
        GgmlType::IQ2_XXS => widest_band(&iq2xxs_scale_codes(), iq2xxs_code_sq(), sigma, in_window)
            .map(|(band, d)| Rule::Iq2Xxs { d, band })
            .ok_or_else(|| no_scale("band of 4-bit scales")),
        GgmlType::IQ1_S => widest_band(&iq1s_scale_codes(), iq1s_code_sq(), sigma, in_window)
            .map(|(band, d)| Rule::Iq1S { d, band })
            .ok_or_else(|| no_scale("band of 3-bit scales")),
        GgmlType::MXFP4 => mxfp4_rule(sigma).ok_or_else(|| no_scale("E8M0 exponent pair")),
        GgmlType::NVFP4 => {
            nvfp4_rule(sigma, window).ok_or_else(|| no_scale("UE4M3 scale byte pair"))
        }
        GgmlType::F32 => Ok(uniform(FloatTy::F32)),
        GgmlType::F16 => Ok(uniform(FloatTy::F16)),
        GgmlType::BF16 => Ok(uniform(FloatTy::BF16)),
        _ => Err(FixtureError::UnsupportedType {
            name: name.to_string(),
            ty,
        }),
    }
}

/// `E[kvalue²]` of the 4-bit float codes MXFP4 and NVFP4 share, drawn uniformly.
fn fp4_code_sq() -> f64 {
    KVALUES_MXFP4
        .iter()
        .map(|&k| f64::from(k) * f64::from(k))
        .sum::<f64>()
        / KVALUES_MXFP4.len() as f64
}

/// MXFP4's scale is E8M0 (`2^(E−128)`, a power of two), not an f16: the
/// f16 window does not apply. `E[d²]` is matched exactly by mixing the two
/// exponents around `σ² / E[kvalue²]`; both must be normal E8M0 codes.
fn mxfp4_rule(sigma: f64) -> Option<Rule> {
    let t = sigma * sigma / fp4_code_sq();
    let e = (t.log2() / 2.0).floor();
    let lo = (2.0f64).powf(e);
    let p = (4.0 * lo * lo - t) / (3.0 * lo * lo);
    let byte = e + 128.0;
    if !(2.0..=253.0).contains(&byte) {
        return None;
    }
    Some(Rule::Mxfp4 {
        e_lo: byte as u8,
        p_lo: (p.clamp(0.0, 1.0) * 4_294_967_296.0) as u64,
    })
}

/// Whether UE4M3 scale byte `byte` is a scale whose product with
/// [`NVFP4_TENSOR_SCALE`] lies in `window`.
pub(super) fn nvfp4_scale_in_window(byte: u8, window: Window) -> bool {
    ue4m3_to_f32_half(byte)
        .is_ok_and(|d| (window.lo..=window.hi).contains(&(NVFP4_TENSOR_SCALE * d)))
}

/// NVFP4's sub-block scale is a UE4M3 byte, a scale per 16 values, and the
/// bytes' decodes grow with the byte. The `E[(S·d)²]` that gives the tensor
/// σ = 1/√K is matched exactly by mixing two adjacent bytes, the largest whose
/// `d` is at most the target and its successor, both in the window.
fn nvfp4_rule(sigma: f64, window: Window) -> Option<Rule> {
    let t = (sigma / f64::from(NVFP4_TENSOR_SCALE)).powi(2) / fp4_code_sq();
    let d_sq = |b: u8| {
        let d = f64::from(ue4m3_to_f32_half(b).expect("a byte below 0x7F is a scale"));
        d * d
    };
    let s_lo = (1..=UE4M3_LAST_PAIR).rev().find(|&b| d_sq(b) <= t)?;
    let (lo, hi) = (d_sq(s_lo), d_sq(s_lo + 1));
    if t >= hi || !nvfp4_scale_in_window(s_lo, window) || !nvfp4_scale_in_window(s_lo + 1, window) {
        return None;
    }
    let p = (hi - t) / (hi - lo);
    Some(Rule::Nvfp4 {
        s_lo,
        p_lo: (p.clamp(0.0, 1.0) * 4_294_967_296.0) as u64,
    })
}

// ------------------------------------------------------------------ generator

/// SplitMix64: one `u64` per step.
pub(super) struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn fill(&mut self, out: &mut [u8]) {
        let (words, tail) = out.as_chunks_mut::<8>();
        for w in words {
            *w = self.next().to_le_bytes();
        }
        if !tail.is_empty() {
            let last = self.next().to_le_bytes();
            let n = tail.len();
            tail.copy_from_slice(&last[..n]);
        }
    }
}

fn mix(x: u64) -> u64 {
    Rng(x).next()
}

/// FNV-1a of a tensor name.
fn name_key(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// The generator of chunk `chunk` of the tensor named `name` under `seed`.
pub(super) fn chunk_rng(seed: u64, name: &str, chunk: usize) -> Rng {
    Rng(mix(mix(mix(seed) ^ name_key(name)) ^ chunk as u64))
}

/// Bytes of one generated unit — a block, or an element of a float type.
pub(super) fn unit_bytes(ty: GgmlType) -> usize {
    ty.type_size()
        .expect("a planned tensor's type has a size: rule_for refuses every other type")
        as usize
}

/// A tensor's generator unit and chunk size.
pub(super) fn chunk_len(ty: GgmlType) -> usize {
    let u = unit_bytes(ty);
    (CHUNK_TARGET / u).max(1) * u
}

/// Q4_K/Q5_K's 6-bit (scale, min) pairs into their 12 bytes, the inverse of
/// ggml's `get_scale_min_k4`.
fn pack_scale_min_k4(sc: &[i32; 8], m: &[i32; 8], out: &mut [u8]) {
    out[..12].fill(0);
    for (j, (&s, &mm)) in sc.iter().zip(m).enumerate() {
        let (s, mm) = (s as u8, mm as u8);
        if j < 4 {
            out[j] = s;
            out[j + 4] = mm;
        } else {
            out[j + 4] = (s & 0x0f) | ((mm & 0x0f) << 4);
            out[j - 4] |= (s >> 4) << 6;
            out[j] |= (mm >> 4) << 6;
        }
    }
}

/// ggml's `get_scale_min_k4`: the (scale, min) of sub-block `j`.
pub(super) fn scale_min_k4(j: usize, q: &[u8]) -> (i32, i32) {
    if j < 4 {
        (i32::from(q[j] & 63), i32::from(q[j + 4] & 63))
    } else {
        let d = i32::from(q[j + 4] & 0x0f) | (i32::from(q[j - 4] >> 6) << 4);
        let m = i32::from(q[j + 4] >> 4) | (i32::from(q[j] >> 6) << 4);
        (d, m)
    }
}

/// Q3_K's sixteen 6-bit scales (`s + 32`) into their 12 bytes, as
/// `quantize_row_q3_K_ref` packs them.
fn pack_q3k_scales(l: &[u8; 16], out: &mut [u8]) {
    out[..12].fill(0);
    for (j, &v) in l.iter().enumerate() {
        if j < 8 {
            out[j] = v & 0x0f;
        } else {
            out[j - 8] |= (v & 0x0f) << 4;
        }
        out[8 + j % 4] |= (v >> 4) << (2 * (j / 4));
    }
}

/// Sub-block `j`'s Q3_K scale, `s = l − 32`.
pub(super) fn q3k_scale(sc: &[u8], j: usize) -> i32 {
    let low = (sc[j % 8] >> (4 * (j / 8))) & 0x0f;
    let high = (sc[8 + j % 4] >> (2 * (j / 4))) & 3;
    i32::from(low | (high << 4)) - 32
}

/// Two band draws per `u64`.
fn draws<const N: usize>(rng: &mut Rng, band: &Band) -> [i32; N] {
    let mut out = [0i32; N];
    for pair in out.chunks_mut(2) {
        let r = rng.next();
        pair[0] = band.draw(r as u32);
        if let Some(b) = pair.get_mut(1) {
            *b = band.draw((r >> 32) as u32);
        }
    }
    out
}

/// `out` as whole `N`-byte units. A remainder would keep a reused buffer's
/// earlier bytes, so it is a broken caller invariant, not a short fill.
fn units<const N: usize>(out: &mut [u8]) -> &mut [[u8; N]] {
    let len = out.len();
    let (units, rest) = out.as_chunks_mut::<N>();
    assert!(
        rest.is_empty(),
        "{len} bytes are not whole {N}-byte units: a chunk is whole units of its tensor's type"
    );
    units
}

/// Fill `out`, whole units of `rule`'s type, from `rng`.
pub(super) fn fill_units(rule: &Rule, rng: &mut Rng, out: &mut [u8]) {
    match rule {
        Rule::Q3K { d, band } => {
            for blk in units::<110>(out) {
                rng.fill(&mut blk[..96]);
                let s: [i32; 16] = draws(rng, band);
                let l: [u8; 16] = std::array::from_fn(|j| (s[j] + 32) as u8);
                pack_q3k_scales(&l, &mut blk[96..108]);
                blk[108..110].copy_from_slice(&d.to_le_bytes());
            }
        }
        Rule::Q4K { d, dmin, band } => {
            for blk in units::<144>(out) {
                blk[0..2].copy_from_slice(&d.to_le_bytes());
                blk[2..4].copy_from_slice(&dmin.to_le_bytes());
                let sc: [i32; 8] = draws(rng, band);
                let m = sc.map(|s| s * Q4K_MIN_PER_SCALE);
                pack_scale_min_k4(&sc, &m, &mut blk[4..16]);
                rng.fill(&mut blk[16..144]);
            }
        }
        Rule::Q5K { d, dmin, band } => {
            for blk in units::<176>(out) {
                blk[0..2].copy_from_slice(&d.to_le_bytes());
                blk[2..4].copy_from_slice(&dmin.to_le_bytes());
                let sc: [i32; 8] = draws(rng, band);
                let m = sc.map(|s| s * Q5K_MIN_PER_SCALE);
                pack_scale_min_k4(&sc, &m, &mut blk[4..16]);
                rng.fill(&mut blk[16..176]);
            }
        }
        Rule::Q6K { d, band } => {
            for blk in units::<210>(out) {
                rng.fill(&mut blk[..192]);
                let s: [i32; 16] = draws(rng, band);
                for (b, v) in blk[192..208].iter_mut().zip(s) {
                    *b = v as i8 as u8;
                }
                blk[208..210].copy_from_slice(&d.to_le_bytes());
            }
        }
        Rule::Q8_0 { d, band } => {
            for blk in units::<34>(out) {
                blk[0..2].copy_from_slice(&d.to_le_bytes());
                let q: [i32; 32] = draws(rng, band);
                for (b, v) in blk[2..].iter_mut().zip(q) {
                    *b = v as i8 as u8;
                }
            }
        }
        Rule::Q5Legacy { d, m } => {
            let head = 2 + 2 * usize::from(m.is_some());
            let unit = head + 20;
            assert!(
                out.len().is_multiple_of(unit),
                "{} bytes are not whole {unit}-byte units: a chunk is whole units of its tensor's type",
                out.len()
            );
            for blk in out.chunks_exact_mut(unit) {
                blk[0..2].copy_from_slice(&d.to_le_bytes());
                if let Some(m) = m {
                    blk[2..4].copy_from_slice(&m.to_le_bytes());
                }
                rng.fill(&mut blk[head..]);
            }
        }
        Rule::Iq4Nl { d, band } => {
            let mut code = [0u8; 256];
            for (c, &k) in KVALUES_IQ4NL.iter().enumerate() {
                code[(i32::from(k) + 128) as usize] = c as u8;
            }
            let code_of = |v: i32| code[(v + 128) as usize];
            for blk in units::<18>(out) {
                blk[0..2].copy_from_slice(&d.to_le_bytes());
                let v: [i32; 32] = draws(rng, band);
                for (j, b) in blk[2..].iter_mut().enumerate() {
                    *b = code_of(v[j]) | (code_of(v[j + 16]) << 4);
                }
            }
        }
        Rule::Iq2Xxs { d, band } => {
            for blk in units::<66>(out) {
                blk[0..2].copy_from_slice(&d.to_le_bytes());
                rng.fill(&mut blk[2..66]);
                let codes: [i32; 8] = draws(rng, band);
                for (w, c) in blk[2..].as_chunks_mut::<8>().0.iter_mut().zip(codes) {
                    w[7] = (w[7] & 0x0f) | (((c - 1) / 2) as u8) << 4;
                }
            }
        }
        Rule::Iq1S { d, band } => {
            for blk in units::<50>(out) {
                blk[0..2].copy_from_slice(&d.to_le_bytes());
                rng.fill(&mut blk[2..50]);
                let codes: [i32; 8] = draws(rng, band);
                // The scale `s` is bits 12..14 of a little-endian `qh`, so bits 4..6 of its high byte;
                // the delta sign and the index bits stay random.
                for (qh, c) in blk[34..50].as_chunks_mut::<2>().0.iter_mut().zip(codes) {
                    qh[1] = (qh[1] & 0x8f) | (((c - 1) / 2) as u8) << 4;
                }
            }
        }
        Rule::Mxfp4 { e_lo, p_lo } => {
            for blk in units::<17>(out) {
                let r = rng.next() & 0xffff_ffff;
                blk[0] = if r < *p_lo { *e_lo } else { e_lo + 1 };
                rng.fill(&mut blk[1..17]);
            }
        }
        Rule::Nvfp4 { s_lo, p_lo } => {
            let pick = |r: u64| if r < *p_lo { *s_lo } else { s_lo + 1 };
            for blk in units::<36>(out) {
                let (scales, codes) = blk.split_at_mut(4);
                for pair in scales.as_chunks_mut::<2>().0 {
                    let r = rng.next();
                    pair[0] = pick(r & 0xffff_ffff);
                    pair[1] = pick(r >> 32);
                }
                rng.fill(codes);
            }
        }
        Rule::Uniform { ty, half_width } => {
            let u = |r: u64| ((r >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * half_width;
            match ty {
                FloatTy::F32 => {
                    for e in units::<4>(out) {
                        *e = u(rng.next()).to_le_bytes();
                    }
                }
                FloatTy::F16 => {
                    for e in units::<2>(out) {
                        *e = f32_to_f16_bits(u(rng.next())).to_le_bytes();
                    }
                }
                FloatTy::BF16 => {
                    for e in units::<2>(out) {
                        *e = bf16_bits(u(rng.next())).to_le_bytes();
                    }
                }
            }
        }
        Rule::Const { value } => {
            for e in units::<4>(out) {
                *e = value.to_le_bytes();
            }
        }
    }
}

/// f32 → bf16 bits, round to nearest even (the values are finite).
fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
