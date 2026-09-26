//! The V4.1 gate fixture: a model file much smaller than the real one that
//! keeps every per-layer shape, so the engine's compile-time constants and
//! the layer kinds the step's launch table is built from all hold, and whose
//! whole weight set fits in host memory.
//!
//! The fixture is nine layers, one per layer kind the step distinguishes
//! ([`LAYER_MAP`]: fixture layer `f` takes source layer `LAYER_MAP[f]`), and
//! the source's globals. Every tensor keeps its source's dims and type id and
//! is renamed `blk.L.` → `blk.f.`; the one exception is an engram table,
//! whose rows follow the fixture's own hash primes. The metadata is the
//! source header's, with the per-layer keys remapped through the map (the
//! coverage table is [`arch_key_rule`]; a key it does not name is an error)
//! and five keys of its own: [`KEY_VERSION`], [`KEY_SEED`],
//! [`KEY_SOURCE_LAYERS`], [`KEY_SOURCE_SHA256`] and [`KEY_CARD_BUDGET`]. A
//! file holding only some of the planned tensors also carries [`KEY_SUBSET`];
//! the engine must not run such a file.
//!
//! Weights are random codes and scales written directly, no quantizer: every
//! matrix element has standard deviation `1/√K` (`K = dims[0]`), so a
//! projection of a unit-RMS input has unit RMS. [`rule_for`] derives each
//! type's scale from its block layout (the dequant ports in `gguf::quant`)
//! and picks the widest band of scale codes whose `d` (and `dmin`) is a
//! normal f16 in [`D_MIN`, `D_MAX`]. Gains are 1, sinks, biases and
//! hyper-connection bases 0, hyper-connection scales 1.
//!
//! The bytes are a function of the seed and the source header alone: each
//! tensor's stream is cut into chunks of about [`CHUNK_TARGET`] bytes, and a
//! chunk's generator is keyed by (seed, tensor name, chunk index), so any
//! thread count writes the same file.
//!
//! A DSpark draft fixture ([`DRAFT_FILE`]) is the real draft's tensors at
//! their shapes with random weights by the same rules, `target_layers`
//! moved to the fixture's last layers, and the same five keys.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::Instant;

use gguf::quant::{KVALUES_MXFP4, f32_to_f16_bits, half_to_f32};
use gguf::write::{Layout, TensorDecl, WriteError, Writer, file_alignment};
use gguf::{GgmlType, LoadError, Split, Value, dequant_row};
use sha2::{Digest, Sha256};

use super::hparams::{DenseStream, Hparams, LayerKind, Stream};
use crate::arch::dspark::{self, DraftHparams};
use crate::fileio;
use crate::placement::PlacementError;

/// Fixture layer `f` holds source layer `LAYER_MAP[f]`: one layer per kind
/// the step's launch table tells apart (window, window+engram, the r2 source
/// and reader, the r2 source with engram, the r1 source that ends the
/// compressed layers, the r1 readers with and without an indexer), and a
/// last reader whose top-k source is not its kv source.
pub const LAYER_MAP: [usize; 9] = [0, 1, 2, 3, 14, 20, 21, 24, 25];

/// The compress ratios the map must read from the source: the check that
/// the map still picks the kinds it was chosen for.
pub const FIXTURE_RATIOS: [u64; 9] = [0, 0, 2, 2, 2, 1, 1, 1, 1];

/// The format of the fixture's own keys.
pub const FIXTURE_VERSION: u32 = 1;
/// u32 [`FIXTURE_VERSION`]; its presence marks a fixture.
pub const KEY_VERSION: &str = "bloomery.fixture.version";
/// u64: the seed every weight stream is keyed by.
pub const KEY_SEED: &str = "bloomery.fixture.seed";
/// u32 array: [`LAYER_MAP`], the source layer of each fixture layer.
pub const KEY_SOURCE_LAYERS: &str = "bloomery.fixture.source_layers";
/// String: lowercase hex sha256 of the source's header bytes, every shard's
/// bytes before its data base, in split order.
pub const KEY_SOURCE_SHA256: &str = "bloomery.fixture.source_header_sha256";
/// u64: the card byte budget (`BLOOMERY_CARD_BUDGET`) the fixture's gate
/// placement plans under.
pub const KEY_CARD_BUDGET: &str = "bloomery.fixture.card_budget";
/// String array: present only on a file that holds some of the planned
/// tensors — their names, in file order.
pub const KEY_SUBSET: &str = "bloomery.fixture.subset";

/// The seed `generate` takes when none is given.
pub const DEFAULT_SEED: u64 = 1;
/// The card budget the fixture design derived for the gate placement:
/// 6,580 MiB keeps about as many experts a layer as the real gate placement.
pub const DEFAULT_CARD_BUDGET: u64 = 6580 << 20;
/// Tensor data bytes a shard holds at most, unless one tensor is larger.
pub const DEFAULT_SHARD_BYTES: u64 = 16 << 30;

/// The target shards are `<STEM>-0000i-of-0000N.gguf`.
pub const STEM: &str = "v41-fixture";
/// The draft fixture's path inside the fixture directory: a directory of its
/// own, so a reader that takes every `*.gguf` beside the target's shards (the
/// engram tables' `shards_in`) never meets it.
pub const DRAFT_FILE: &str = "draft/v41-fixture-draft.gguf";

/// Every generated block's `d` (and `dmin`) lies in [`D_MIN`, `D_MAX`], a
/// range of normal f16 values.
pub const D_MIN: f32 = 1.0 / 8192.0;
/// See [`D_MIN`].
pub const D_MAX: f32 = 1.0 / 1024.0;

/// [`D_MIN`, `D_MAX`] as powers of two, for an error line.
fn d_window() -> String {
    format!("[2^{}, 2^{}]", D_MIN.log2(), D_MAX.log2())
}

/// A chunk of a tensor's stream is about this many bytes: whole blocks.
pub const CHUNK_TARGET: usize = 1 << 20;

/// Engram hash primes are the smallest primes above this, as many as the
/// sites' buckets, in site order.
pub const ENGRAM_PRIME_FLOOR: u64 = 1 << 14;

const TARGET_ARCH: &str = "deepseek41";
const SPLIT_NO: &str = "split.no";
const SPLIT_COUNT: &str = "split.count";
const SPLIT_TENSORS: &str = "split.tensors.count";

#[derive(Debug, thiserror::Error)]
pub enum FixtureError {
    #[error(transparent)]
    Load(#[from] LoadError),
    #[error(transparent)]
    Placement(#[from] PlacementError),
    #[error("{path}: {op}: {source}")]
    Io {
        path: PathBuf,
        op: &'static str,
        source: io::Error,
    },
    #[error("{path}: writing: {source}")]
    Write { path: PathBuf, source: WriteError },
    #[error("the {set}'s metadata: {source}")]
    Alignment {
        set: &'static str,
        source: WriteError,
    },
    #[error("the source is {got:?}, not a {want} file")]
    Architecture {
        got: Option<String>,
        want: &'static str,
    },
    #[error("metadata {key}: {detail}")]
    Metadata { key: String, detail: String },
    #[error("metadata {key} is not covered by the fixture map")]
    UncoveredKey { key: String },
    #[error("source layer {layer} is past the source's {n_layer} layers")]
    NoSourceLayer { layer: usize, n_layer: usize },
    #[error("tensor {name}: {detail}")]
    Tensor { name: String, detail: String },
    #[error("tensor {name}: type {ty} has no fixture generator")]
    UnsupportedType { name: String, ty: GgmlType },
    #[error(
        "tensor {name}: no {what} of type {ty} at K = {k} puts d in {}",
        d_window()
    )]
    NoScale {
        name: String,
        ty: GgmlType,
        k: u64,
        what: &'static str,
    },
    #[error("tensor {name}: a 1-D {ty} tensor with no fill rule")]
    NoFillRule { name: String, ty: GgmlType },
    #[error("--tensors names {name}, which the plan does not hold")]
    UnknownTensor { name: String },
    #[error("{} shards, and no fixture layer spans a shard boundary", shards)]
    NoSpanningLayer { shards: usize },
    #[error("{path} exists; a fixture is never written over")]
    Exists { path: PathBuf },
    #[error("{path}: {need} bytes to write, {free} free")]
    Space { path: PathBuf, need: u64, free: u64 },
    #[error("{error}; and removing {tmp} failed: {cleanup}")]
    Abandoned {
        error: Box<FixtureError>,
        tmp: PathBuf,
        cleanup: io::Error,
    },
    #[error("the fixture's {key} is {got}, the source's is {want}")]
    SourceMismatch {
        key: &'static str,
        got: String,
        want: String,
    },
    #[error("the fixture's {what}: {detail}")]
    Mismatch { what: String, detail: String },
    #[error("tensor {name}: block {block}: {detail}")]
    Block {
        name: String,
        block: usize,
        detail: String,
    },
}

fn io_err(path: &Path, op: &'static str, source: io::Error) -> FixtureError {
    FixtureError::Io {
        path: path.to_path_buf(),
        op,
        source,
    }
}

fn meta(key: &str, detail: impl Into<String>) -> FixtureError {
    FixtureError::Metadata {
        key: key.to_string(),
        detail: detail.into(),
    }
}

// ---------------------------------------------------------------- weight rules

/// The scale codes a block draws from: every value `v` of the type's field
/// range `[min, max]` with `lo ≤ |v| ≤ hi`, uniformly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Band {
    pub lo: i32,
    pub hi: i32,
    values: Vec<i32>,
}

impl Band {
    fn new(min: i32, max: i32, lo: i32, hi: i32) -> Band {
        let values = (min..=max)
            .filter(|v| (lo..=hi).contains(&v.abs()))
            .collect();
        Band { lo, hi, values }
    }

    /// Whether `v` is one of the band's codes.
    pub fn holds(&self, v: i32) -> bool {
        self.values.binary_search(&v).is_ok()
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
    /// `2^(E−128)·kvalues[c]`: `E` is `e_lo` with probability `p_lo / 2^32`,
    /// else `e_lo + 1`; `c` uniform over the 16 codes.
    Mxfp4 { e_lo: u8, p_lo: u64 },
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
            Rule::Mxfp4 { e_lo, p_lo } => format!(
                "E8M0 {e_lo} (d 2^{}) with p {:.4}, else {} (d 2^{})",
                i32::from(*e_lo) - 128,
                *p_lo as f64 / 4_294_967_296.0,
                e_lo + 1,
                i32::from(*e_lo) - 127
            ),
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
const Q4K_MIN_PER_SCALE: i32 = 1;
const Q4K_DMIN_PER_D: f32 = 7.5;
const Q5K_CODE_VAR: f64 = 85.25;
const Q5K_MIN_PER_SCALE: i32 = 2;
const Q5K_DMIN_PER_D: f32 = 7.75;
const Q6K_CODE_SQ: f64 = 341.5;

/// Whether `bits` is a normal f16 in [`D_MIN`, `D_MAX`].
pub fn d_in_window(bits: u16) -> bool {
    let exp = (bits >> 10) & 0x1f;
    let v = half_to_f32(bits);
    exp != 0 && exp != 0x1f && (D_MIN..=D_MAX).contains(&v)
}

fn f16(x: f64) -> u16 {
    f32_to_f16_bits(x as f32)
}

/// The widest band of `[min, max]` (ties: the lowest `lo`) whose `d` passes
/// `fits`, where `d = σ / √(E[v²]·code_sq)`.
fn widest_band(
    min: i32,
    max: i32,
    code_sq: f64,
    sigma: f64,
    fits: impl Fn(u16) -> bool,
) -> Option<(Band, u16)> {
    let top = min.abs().max(max);
    for width in (0..=top).rev() {
        for lo in 0..=top - width {
            let band = Band::new(min, max, lo, lo + width);
            if band.values.is_empty() {
                continue;
            }
            let e = band.mean_sq();
            if e == 0.0 {
                continue;
            }
            let d = f16(sigma / (e * code_sq).sqrt());
            if fits(d) {
                return Some((band, d));
            }
        }
    }
    None
}

/// The rule of a quantized or float matrix of type `ty` whose rows are `k`
/// values long: element standard deviation `1/√k`.
pub fn rule_for(name: &str, ty: GgmlType, k: u64) -> Result<Rule, FixtureError> {
    let sigma = 1.0 / (k as f64).sqrt();
    let no_scale = |what| FixtureError::NoScale {
        name: name.to_string(),
        ty,
        k,
        what,
    };
    let with_min = |dmin_per_d: f32| {
        move |d: u16| d_in_window(d) && d_in_window(f32_to_f16_bits(half_to_f32(d) * dmin_per_d))
    };
    let dmin_of = |d: u16, per: f32| f32_to_f16_bits(half_to_f32(d) * per);
    let uniform = |ty| Rule::Uniform {
        ty,
        half_width: (3.0f64.sqrt() * sigma) as f32,
    };
    match ty {
        GgmlType::Q3_K => widest_band(-32, 31, Q3K_CODE_SQ, sigma, d_in_window)
            .map(|(band, d)| Rule::Q3K { d, band })
            .ok_or_else(|| no_scale("band of 6-bit scales")),
        GgmlType::Q4_K => widest_band(0, 63, Q4K_CODE_VAR, sigma, with_min(Q4K_DMIN_PER_D))
            .map(|(band, d)| Rule::Q4K {
                d,
                dmin: dmin_of(d, Q4K_DMIN_PER_D),
                band,
            })
            .ok_or_else(|| no_scale("band of 6-bit scales")),
        GgmlType::Q5_K => widest_band(0, 31, Q5K_CODE_VAR, sigma, with_min(Q5K_DMIN_PER_D))
            .map(|(band, d)| Rule::Q5K {
                d,
                dmin: dmin_of(d, Q5K_DMIN_PER_D),
                band,
            })
            .ok_or_else(|| no_scale("band of 6-bit scales")),
        GgmlType::Q6_K => widest_band(-128, 127, Q6K_CODE_SQ, sigma, d_in_window)
            .map(|(band, d)| Rule::Q6K { d, band })
            .ok_or_else(|| no_scale("band of i8 scales")),
        GgmlType::Q8_0 => widest_band(-127, 127, 1.0, sigma, d_in_window)
            .map(|(band, d)| Rule::Q8_0 { d, band })
            .ok_or_else(|| no_scale("band of i8 codes")),
        GgmlType::MXFP4 => mxfp4_rule(sigma).ok_or_else(|| no_scale("E8M0 exponent pair")),
        GgmlType::F32 => Ok(uniform(FloatTy::F32)),
        GgmlType::F16 => Ok(uniform(FloatTy::F16)),
        GgmlType::BF16 => Ok(uniform(FloatTy::BF16)),
        _ => Err(FixtureError::UnsupportedType {
            name: name.to_string(),
            ty,
        }),
    }
}

/// MXFP4's scale is E8M0 (`2^(E−128)`, a power of two), not an f16: the
/// f16 window does not apply. `E[d²]` is matched exactly by mixing the two
/// exponents around `σ² / E[kvalue²]`; both must be normal E8M0 codes.
fn mxfp4_rule(sigma: f64) -> Option<Rule> {
    let code_sq: f64 = KVALUES_MXFP4
        .iter()
        .map(|&k| f64::from(k) * f64::from(k))
        .sum::<f64>()
        / KVALUES_MXFP4.len() as f64;
    let t = sigma * sigma / code_sq;
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

/// The value of a 1-D F32 tensor, by its name without the `blk.N.` prefix:
/// gains 1, sinks, biases and hyper-connection bases 0, hyper-connection
/// scales 1.
fn const_rule(leaf: &str) -> Option<f32> {
    const RULES: [(&str, f32); 15] = [
        ("attn_norm.weight", 1.0),
        ("ffn_norm.weight", 1.0),
        ("attn_q_a_norm.weight", 1.0),
        ("attn_kv_a_norm.weight", 1.0),
        ("attn_compressor_norm.weight", 1.0),
        ("indexer.k_norm.weight", 1.0),
        ("output_norm.weight", 1.0),
        ("enc.output_norm.weight", 1.0),
        ("attn_sinks.weight", 0.0),
        ("exp_probs_b.bias", 0.0),
        ("exp_probs_b_vl.bias", 0.0),
        ("hc_attn_base.weight", 0.0),
        ("hc_ffn_base.weight", 0.0),
        ("hc_attn_scale.weight", 1.0),
        ("hc_ffn_scale.weight", 1.0),
    ];
    RULES.iter().find(|(n, _)| *n == leaf).map(|&(_, v)| v)
}

/// The rule of tensor `name` of type `ty` at `dims`.
fn fill_rule(name: &str, ty: GgmlType, dims: &[u64]) -> Result<Rule, FixtureError> {
    if dims.len() >= 2 {
        return rule_for(name, ty, dims[0]);
    }
    let leaf = split_layer(name).map_or(name, |(_, rest)| rest);
    match (ty, const_rule(leaf)) {
        (GgmlType::F32, Some(value)) => Ok(Rule::Const { value }),
        _ => Err(FixtureError::NoFillRule {
            name: name.to_string(),
            ty,
        }),
    }
}

/// `blk.<n>.<rest>` → `(n, rest)`.
fn split_layer(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let (n, rest) = rest.split_once('.')?;
    Some((n.parse().ok()?, rest))
}

// ------------------------------------------------------------------ generator

/// SplitMix64: one `u64` per step.
struct Rng(u64);

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
fn chunk_rng(seed: u64, name: &str, chunk: usize) -> Rng {
    Rng(mix(mix(mix(seed) ^ name_key(name)) ^ chunk as u64))
}

/// Bytes of one generated unit — a block, or an element of a float type.
fn unit_bytes(ty: GgmlType) -> usize {
    ty.type_size()
        .expect("a planned tensor's type has a size: rule_for refuses every other type")
        as usize
}

/// A tensor's generator unit and chunk size.
fn chunk_len(ty: GgmlType) -> usize {
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
fn scale_min_k4(j: usize, q: &[u8]) -> (i32, i32) {
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
fn q3k_scale(sc: &[u8], j: usize) -> i32 {
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
fn fill_units(rule: &Rule, rng: &mut Rng, out: &mut [u8]) {
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
        Rule::Mxfp4 { e_lo, p_lo } => {
            for blk in units::<17>(out) {
                let r = rng.next() & 0xffff_ffff;
                blk[0] = if r < *p_lo { *e_lo } else { e_lo + 1 };
                rng.fill(&mut blk[1..17]);
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

// ----------------------------------------------------------------------- plan

/// A file's metadata pairs, in file order.
pub type Kvs = Vec<(String, Value)>;

/// One tensor of a fixture file.
#[derive(Clone, Debug)]
pub struct PlannedTensor {
    /// Its name in the fixture.
    pub name: String,
    /// The source tensor it copies the shape of.
    pub source: String,
    /// The fixture layer, `None` for a global or a draft tensor.
    pub layer: Option<usize>,
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    pub nbytes: u64,
    pub rule: Rule,
}

impl PlannedTensor {
    /// Chunks its stream is cut into.
    pub fn chunks(&self) -> usize {
        (self.nbytes as usize).div_ceil(chunk_len(self.ty))
    }

    /// The byte range of chunk `c`.
    pub fn chunk_range(&self, c: usize) -> Range<usize> {
        let len = chunk_len(self.ty);
        c * len..((c + 1) * len).min(self.nbytes as usize)
    }

    /// Chunk `c`'s bytes under `seed` into `out` (its length).
    pub fn fill_chunk(&self, seed: u64, c: usize, out: &mut [u8]) {
        let mut rng = chunk_rng(seed, &self.name, c);
        fill_units(&self.rule, &mut rng, out);
    }

    /// The whole tensor under `seed` into `out` (`nbytes` long), its chunks
    /// spread over `threads` threads.
    pub fn fill(&self, seed: u64, threads: usize, out: &mut [u8]) {
        let len = chunk_len(self.ty);
        let mut lanes: Vec<Vec<_>> = (0..threads.max(1)).map(|_| Vec::new()).collect();
        let n = lanes.len();
        for (c, part) in out.chunks_mut(len).enumerate() {
            lanes[c % n].push((c, part));
        }
        std::thread::scope(|s| {
            for lane in lanes {
                s.spawn(move || {
                    for (c, part) in lane {
                        self.fill_chunk(seed, c, part);
                    }
                });
            }
        });
    }

    /// The element standard deviation it is generated at, `None` for a
    /// constant.
    pub fn sigma(&self) -> Option<f64> {
        match self.rule {
            Rule::Const { .. } => None,
            _ => Some(1.0 / (self.dims[0] as f64).sqrt()),
        }
    }
}

/// One file set: the target's shards or the draft's single file.
#[derive(Clone, Debug)]
pub struct FilePlan {
    /// The first file's metadata; a split set's keys carry their final values.
    pub kvs: Vec<(String, Value)>,
    pub tensors: Vec<PlannedTensor>,
    /// Each file's tensors, as index ranges into `tensors`.
    pub shards: Vec<Range<usize>>,
    /// File names, one per shard.
    pub files: Vec<String>,
    /// The split keys every later shard carries, in the first shard's order.
    split_keys: Vec<(String, Value)>,
    /// The alignment `kvs` sets, by the writer's rule.
    align: u64,
}

impl FilePlan {
    /// Each file's name and layout.
    pub fn layouts(&self) -> Result<Vec<(String, Layout)>, FixtureError> {
        let mut out = Vec::with_capacity(self.shards.len());
        for (i, range) in self.shards.iter().enumerate() {
            let kvs = if i == 0 {
                self.kvs.clone()
            } else {
                self.split_keys
                    .iter()
                    .map(|(k, v)| {
                        let v = if k == SPLIT_NO {
                            int_like(k, v, i as u64)?
                        } else {
                            v.clone()
                        };
                        Ok((k.clone(), v))
                    })
                    .collect::<Result<Vec<_>, FixtureError>>()?
            };
            let decls = self.tensors[range.clone()]
                .iter()
                .map(|t| TensorDecl {
                    name: t.name.clone(),
                    dims: t.dims.clone(),
                    type_id: t.ty.as_u32(),
                    nbytes: t.nbytes,
                })
                .collect();
            let layout = Layout::new(&kvs, decls).map_err(|source| FixtureError::Write {
                path: PathBuf::from(&self.files[i]),
                source,
            })?;
            out.push((self.files[i].clone(), layout));
        }
        Ok(out)
    }

    /// The fixture layers whose tensors lie in more than one shard.
    pub fn spanning_layers(&self) -> Vec<usize> {
        let mut first: HashMap<usize, usize> = HashMap::new();
        let mut spans = Vec::new();
        for (s, range) in self.shards.iter().enumerate() {
            for t in &self.tensors[range.clone()] {
                if let Some(l) = t.layer {
                    let at = *first.entry(l).or_insert(s);
                    if at != s && !spans.contains(&l) {
                        spans.push(l);
                    }
                }
            }
        }
        spans
    }

    /// Bytes a tensor takes in its file, padding included.
    pub fn padded(&self, t: &PlannedTensor) -> u64 {
        t.nbytes.next_multiple_of(self.align)
    }
}

/// The target's plan and, when a draft source is given, the draft's.
#[derive(Clone, Debug)]
pub struct Plan {
    pub target: FilePlan,
    pub draft: Option<FilePlan>,
    /// The fixture's engram primes per site, and the source's.
    pub engram_rows: Vec<(u64, u64)>,
}

/// What `plan`, `generate` and `verify` share.
#[derive(Clone, Debug)]
pub struct Options {
    pub seed: u64,
    pub card_budget: u64,
    pub shard_bytes: u64,
    /// Only these target tensors (a subset file), in plan order.
    pub tensors: Option<Vec<String>>,
    /// Only these draft tensors.
    pub draft_tensors: Option<Vec<String>>,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            seed: DEFAULT_SEED,
            card_budget: DEFAULT_CARD_BUDGET,
            shard_bytes: DEFAULT_SHARD_BYTES,
            tensors: None,
            draft_tensors: None,
        }
    }
}

/// `template`'s integer variant holding `v`.
fn int_like(key: &str, template: &Value, v: u64) -> Result<Value, FixtureError> {
    let bad = || meta(key, format!("{v} does not fit the source's {template:?}"));
    Ok(match template {
        Value::U8(_) => Value::U8(u8::try_from(v).map_err(|_| bad())?),
        Value::U16(_) => Value::U16(u16::try_from(v).map_err(|_| bad())?),
        Value::U32(_) => Value::U32(u32::try_from(v).map_err(|_| bad())?),
        Value::U64(_) => Value::U64(v),
        Value::I8(_) => Value::I8(i8::try_from(v).map_err(|_| bad())?),
        Value::I16(_) => Value::I16(i16::try_from(v).map_err(|_| bad())?),
        Value::I32(_) => Value::I32(i32::try_from(v).map_err(|_| bad())?),
        Value::I64(_) => Value::I64(i64::try_from(v).map_err(|_| bad())?),
        _ => return Err(meta(key, format!("is {template:?}, not an integer"))),
    })
}

fn items<'a>(key: &str, v: &'a Value) -> Result<&'a [Value], FixtureError> {
    match v {
        Value::Array(a) => Ok(a),
        _ => Err(meta(key, "is not an array")),
    }
}

fn unsigned(key: &str, v: &Value) -> Result<u64, FixtureError> {
    v.as_unsigned()
        .ok_or_else(|| meta(key, format!("{v:?} is not an unsigned integer")))
}

/// What the fixture does with source key `<arch>.<suffix>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeyRule {
    Copy,
    BlockCount,
    Ratios,
    PerLayer,
    EngramLayers,
    EngramPrimes,
    EngramOffsets,
    ZeroHashLayers,
}

/// The coverage table of the target's architecture keys; `None` is a key
/// the map does not cover.
fn arch_key_rule(suffix: &str) -> Option<KeyRule> {
    const COPY: [&str; 38] = [
        "context_length",
        "embedding_length",
        "attention.head_count",
        "attention.head_count_kv",
        "rope.scaling.type",
        "rope.scaling.factor",
        "rope.scaling.original_context_length",
        "rope.scaling.yarn_beta_fast",
        "rope.scaling.yarn_beta_slow",
        "rope.freq_base",
        "attention.layer_norm_rms_epsilon",
        "expert_count",
        "expert_used_count",
        "expert_gating_func",
        "attention.key_length",
        "attention.value_length",
        "rope.dimension_count",
        "attention.q_lora_rank",
        "attention.sliding_window",
        "expert_feed_forward_length",
        "expert_shared_count",
        "expert_weights_scale",
        "expert_weights_norm",
        "attention.indexer.head_count",
        "attention.indexer.key_length",
        "attention.indexer.top_k",
        "attention.output_group_count",
        "attention.output_lora_rank",
        "attention.compress_rope_freq_base",
        "hyper_connection.count",
        "hyper_connection.sinkhorn_iterations",
        "hyper_connection.epsilon",
        "engram.head_count",
        "engram.key_length",
        "engram.max_ngram_size",
        "engram.multipliers",
        "engram.token_map",
        "engram.pad_id",
    ];
    Some(match suffix {
        "block_count" => KeyRule::BlockCount,
        "attention.compress_ratios" => KeyRule::Ratios,
        "swiglu_clamp_exp" | "swiglu_clamp_shexp" => KeyRule::PerLayer,
        "engram.layer_ids" => KeyRule::EngramLayers,
        "engram.primes" => KeyRule::EngramPrimes,
        "engram.offsets" => KeyRule::EngramOffsets,
        "hash_layer_count" => KeyRule::ZeroHashLayers,
        s if COPY.contains(&s) => KeyRule::Copy,
        _ => return None,
    })
}

/// The first `n` primes above `floor`.
fn primes_above(floor: u64, n: usize) -> Vec<u64> {
    let is_prime = |p: u64| {
        p >= 2
            && (2..)
                .take_while(|d| d * d <= p)
                .all(|d| !p.is_multiple_of(d))
    };
    (floor + 1..).filter(|&p| is_prime(p)).take(n).collect()
}

/// The source's engram layout: per site its buckets' primes; the offsets
/// must be each site's exclusive prefix sums, as the fixture writes them.
struct EngramSource {
    layer_ids: Vec<usize>,
    cols: usize,
    primes: Vec<u64>,
    prime_tag: Value,
    offset_tag: Value,
}

fn engram_source(source: &Split) -> Result<EngramSource, FixtureError> {
    let key = |s: &str| format!("{TARGET_ARCH}.engram.{s}");
    let get = |s: &str| {
        source
            .value(&key(s))
            .ok_or_else(|| meta(&key(s), "is absent"))
    };
    let layer_ids = items(&key("layer_ids"), get("layer_ids")?)?
        .iter()
        .map(|v| unsigned(&key("layer_ids"), v).map(|l| l as usize))
        .collect::<Result<Vec<_>, _>>()?;
    if layer_ids.is_empty() {
        return Err(meta(&key("layer_ids"), "is empty"));
    }
    let heads = unsigned(&key("head_count"), get("head_count")?)? as usize;
    let ngram = unsigned(&key("max_ngram_size"), get("max_ngram_size")?)? as usize;
    let cols = ngram
        .checked_sub(1)
        .map(|n| n * heads)
        .filter(|&c| c > 0)
        .ok_or_else(|| meta(&key("max_ngram_size"), "leaves no bucket"))?;
    let p = items(&key("primes"), get("primes")?)?;
    let o = items(&key("offsets"), get("offsets")?)?;
    let want = layer_ids.len() * cols;
    if p.len() != want || o.len() != want {
        return Err(meta(
            &key("primes"),
            format!(
                "{} primes and {} offsets for {want} buckets",
                p.len(),
                o.len()
            ),
        ));
    }
    let primes = p
        .iter()
        .map(|v| unsigned(&key("primes"), v))
        .collect::<Result<Vec<_>, _>>()?;
    for (site, (ps, os)) in primes.chunks(cols).zip(o.chunks(cols)).enumerate() {
        let mut acc = 0u64;
        for (b, (&pr, ov)) in ps.iter().zip(os).enumerate() {
            if unsigned(&key("offsets"), ov)? != acc {
                return Err(meta(
                    &key("offsets"),
                    format!("site {site} bucket {b} is not its site's prefix sum {acc}"),
                ));
            }
            acc += pr;
        }
    }
    Ok(EngramSource {
        layer_ids,
        cols,
        primes,
        prime_tag: p[0].clone(),
        offset_tag: o[0].clone(),
    })
}

/// lowercase hex sha256 of every shard's bytes before its data base, in
/// split order.
pub fn header_sha256(split: &Split) -> String {
    let mut h = Sha256::new();
    for i in 0..split.shard_count() {
        let g = split
            .shard(i)
            .expect("a split has a reader for every shard index");
        h.update(g.header_bytes());
    }
    fileio::hex(&h.finalize())
}

/// The fixture's own keys.
fn fixture_keys(opts: &Options, source_sha: String) -> Vec<(String, Value)> {
    vec![
        (KEY_VERSION.to_string(), Value::U32(FIXTURE_VERSION)),
        (KEY_SEED.to_string(), Value::U64(opts.seed)),
        (
            KEY_SOURCE_LAYERS.to_string(),
            Value::Array(LAYER_MAP.iter().map(|&l| Value::U32(l as u32)).collect()),
        ),
        (KEY_SOURCE_SHA256.to_string(), Value::String(source_sha)),
        (KEY_CARD_BUDGET.to_string(), Value::U64(opts.card_budget)),
    ]
}

/// Keep only `subset` of `tensors` (every name must be planned) and record it.
fn apply_subset(
    tensors: Vec<PlannedTensor>,
    subset: Option<&Vec<String>>,
    kvs: &mut Vec<(String, Value)>,
) -> Result<Vec<PlannedTensor>, FixtureError> {
    let Some(names) = subset else {
        return Ok(tensors);
    };
    if let Some(n) = names
        .iter()
        .find(|n| !tensors.iter().any(|t| &t.name == *n))
    {
        return Err(FixtureError::UnknownTensor { name: n.clone() });
    }
    let kept: Vec<PlannedTensor> = tensors
        .into_iter()
        .filter(|t| names.contains(&t.name))
        .collect();
    kvs.push((
        KEY_SUBSET.to_string(),
        Value::Array(kept.iter().map(|t| Value::String(t.name.clone())).collect()),
    ));
    Ok(kept)
}

/// Rules are a function of (type, K); planned once per pair.
struct Rules(HashMap<(u32, u64), Rule>);

impl Rules {
    fn get(&mut self, name: &str, ty: GgmlType, dims: &[u64]) -> Result<Rule, FixtureError> {
        if dims.len() < 2 {
            return fill_rule(name, ty, dims);
        }
        let key = (ty.as_u32(), dims[0]);
        if let Some(r) = self.0.get(&key) {
            return Ok(r.clone());
        }
        let r = fill_rule(name, ty, dims)?;
        self.0.insert(key, r.clone());
        Ok(r)
    }
}

fn planned(
    rules: &mut Rules,
    name: String,
    source: &str,
    layer: Option<usize>,
    dims: Vec<u64>,
    ty: GgmlType,
) -> Result<PlannedTensor, FixtureError> {
    let rule = rules.get(&name, ty, &dims)?;
    let (_, blck, tsz) =
        gguf::ggml_type_info(ty.as_u32()).ok_or_else(|| FixtureError::UnsupportedType {
            name: name.clone(),
            ty,
        })?;
    if !dims[0].is_multiple_of(blck) {
        return Err(FixtureError::Tensor {
            name,
            detail: format!("ne[0] {} is not whole blocks of {blck}", dims[0]),
        });
    }
    let nbytes = tsz * (dims[0] / blck) * dims[1..].iter().product::<u64>();
    Ok(PlannedTensor {
        name,
        source: source.to_string(),
        layer,
        dims,
        ty,
        nbytes,
        rule,
    })
}

/// The fixture of `source` (the real first shard's split set) and, with
/// `draft`, of the real DSpark draft.
pub fn plan(source: &Split, draft: Option<&Split>, opts: &Options) -> Result<Plan, FixtureError> {
    if source.architecture() != Some(TARGET_ARCH) {
        return Err(FixtureError::Architecture {
            got: source.architecture().map(str::to_string),
            want: TARGET_ARCH,
        });
    }
    let n_layer = source
        .arch_get_u64("block_count")
        .ok_or_else(|| meta("block_count", "is absent"))? as usize;
    if let Some(&l) = LAYER_MAP.iter().find(|&&l| l >= n_layer) {
        return Err(FixtureError::NoSourceLayer { layer: l, n_layer });
    }
    let engram = engram_source(source)?;
    let fx_primes = primes_above(ENGRAM_PRIME_FLOOR, engram.primes.len());
    let site_rows = |p: &[u64]| {
        p.chunks(engram.cols)
            .map(|c| c.iter().sum::<u64>())
            .collect::<Vec<u64>>()
    };
    let (fx_rows, src_rows) = (site_rows(&fx_primes), site_rows(&engram.primes));
    let (mut kvs, split_keys) = target_kvs(source, n_layer, &engram, &fx_primes)?;
    kvs.extend(fixture_keys(opts, header_sha256(source)));
    let mut rules = Rules(HashMap::new());
    let tensors = target_tensors(source, &engram, &fx_rows, &src_rows, &mut rules)?;
    let tensors = apply_subset(tensors, opts.tensors.as_ref(), &mut kvs)?;
    let target = shard(kvs, split_keys, tensors, opts.shard_bytes)?;
    let draft = match draft {
        None => None,
        Some(d) => Some(plan_draft(d, n_layer, opts, &mut rules)?),
    };
    Ok(Plan {
        target,
        draft,
        engram_rows: fx_rows.into_iter().zip(src_rows).collect(),
    })
}

/// The target's metadata in the source's order, each key by
/// [`arch_key_rule`], and the source's split keys.
fn target_kvs(
    source: &Split,
    n_layer: usize,
    engram: &EngramSource,
    fx_primes: &[u64],
) -> Result<(Kvs, Kvs), FixtureError> {
    let arch_prefix = format!("{TARGET_ARCH}.");
    let mut kvs: Vec<(String, Value)> = Vec::new();
    let mut split_keys: Vec<(String, Value)> = Vec::new();
    for (k, v) in source.iter_kv() {
        if k.starts_with("split.") {
            split_keys.push((k.to_string(), v.clone()));
            kvs.push((k.to_string(), v.clone()));
            continue;
        }
        if k.starts_with("general.") || k.starts_with("tokenizer.") {
            kvs.push((k.to_string(), v.clone()));
            continue;
        }
        let rule = k
            .strip_prefix(&arch_prefix)
            .and_then(arch_key_rule)
            .ok_or_else(|| FixtureError::UncoveredKey { key: k.to_string() })?;
        let nv = match rule {
            KeyRule::Copy => v.clone(),
            KeyRule::BlockCount => int_like(k, v, LAYER_MAP.len() as u64)?,
            KeyRule::ZeroHashLayers => {
                if unsigned(k, v)? != 0 {
                    return Err(meta(k, "is not 0; hash-routed layers are not mapped"));
                }
                v.clone()
            }
            KeyRule::Ratios => mapped_ratios(k, v, n_layer)?,
            KeyRule::PerLayer => match v {
                Value::Array(a) if a.len() == n_layer => {
                    Value::Array(LAYER_MAP.iter().map(|&l| a[l].clone()).collect())
                }
                Value::Array(a) => {
                    return Err(meta(
                        k,
                        format!("has {} values for {n_layer} layers", a.len()),
                    ));
                }
                other => other.clone(),
            },
            KeyRule::EngramLayers => {
                let tag = items(k, v)?.first().ok_or_else(|| meta(k, "is empty"))?;
                let mut ids = Vec::with_capacity(engram.layer_ids.len());
                for &l in &engram.layer_ids {
                    let f = LAYER_MAP.iter().position(|&m| m == l).ok_or_else(|| {
                        meta(k, format!("lists layer {l}, which the map does not hold"))
                    })?;
                    ids.push(int_like(k, tag, f as u64)?);
                }
                Value::Array(ids)
            }
            KeyRule::EngramPrimes => Value::Array(
                fx_primes
                    .iter()
                    .map(|&p| int_like(k, &engram.prime_tag, p))
                    .collect::<Result<_, _>>()?,
            ),
            KeyRule::EngramOffsets => {
                let mut out = Vec::with_capacity(fx_primes.len());
                for site in fx_primes.chunks(engram.cols) {
                    let mut acc = 0u64;
                    for &p in site {
                        out.push(int_like(k, &engram.offset_tag, acc)?);
                        acc += p;
                    }
                }
                Value::Array(out)
            }
        };
        kvs.push((k.to_string(), nv));
    }
    Ok((kvs, split_keys))
}

/// `compress_ratios` at the map's layers, then the source's entries past its
/// `block_count`; the mapped entries must be [`FIXTURE_RATIOS`].
fn mapped_ratios(k: &str, v: &Value, n_layer: usize) -> Result<Value, FixtureError> {
    let a = items(k, v)?;
    if a.len() < n_layer {
        return Err(meta(
            k,
            format!("has {} values for {n_layer} layers", a.len()),
        ));
    }
    let mapped: Vec<Value> = LAYER_MAP.iter().map(|&l| a[l].clone()).collect();
    let got = mapped
        .iter()
        .map(|x| unsigned(k, x))
        .collect::<Result<Vec<_>, _>>()?;
    if got != FIXTURE_RATIOS {
        return Err(meta(
            k,
            format!("the map's layers read {got:?}, the fixture is built for {FIXTURE_RATIOS:?}"),
        ));
    }
    Ok(Value::Array(
        mapped
            .into_iter()
            .chain(a[n_layer..].iter().cloned())
            .collect(),
    ))
}

/// The globals in the source's order, then each fixture layer's tensors in
/// the source's order, renamed; an engram table's rows set to its site's
/// fixture primes.
fn target_tensors(
    source: &Split,
    engram: &EngramSource,
    fx_rows: &[u64],
    src_rows: &[u64],
    rules: &mut Rules,
) -> Result<Vec<PlannedTensor>, FixtureError> {
    let mut globals = Vec::new();
    let mut by_layer: Vec<Vec<PlannedTensor>> = vec![Vec::new(); LAYER_MAP.len()];
    for (_, t) in source.iter_tensors() {
        let Some((l, rest)) = split_layer(&t.name) else {
            globals.push(planned(
                rules,
                t.name.clone(),
                &t.name,
                None,
                t.dims.clone(),
                t.ty,
            )?);
            continue;
        };
        let Some(f) = LAYER_MAP.iter().position(|&m| m == l) else {
            continue;
        };
        let mut dims = t.dims.clone();
        if rest == "engram_embd.weight" {
            let bad = |detail: String| FixtureError::Tensor {
                name: t.name.clone(),
                detail,
            };
            let site = engram
                .layer_ids
                .iter()
                .position(|&x| x == l)
                .ok_or_else(|| {
                    bad("is the table of a layer engram.layer_ids does not list".into())
                })?;
            if dims.len() != 2 || dims[1] != src_rows[site] {
                let want = src_rows[site];
                return Err(bad(format!(
                    "has dims {dims:?}; its site's primes sum to {want}"
                )));
            }
            dims[1] = fx_rows[site];
        }
        let name = format!("blk.{f}.{rest}");
        by_layer[f].push(planned(rules, name, &t.name, Some(f), dims, t.ty)?);
    }
    if let Some(f) = by_layer.iter().position(Vec::is_empty) {
        return Err(FixtureError::Tensor {
            name: format!("blk.{}.*", LAYER_MAP[f]),
            detail: "the source holds no tensor of this mapped layer".into(),
        });
    }
    Ok(globals
        .into_iter()
        .chain(by_layer.into_iter().flatten())
        .collect())
}

/// Cut `tensors` into shards of at most `cap` data bytes each (a larger
/// tensor alone) and set the split keys.
fn shard(
    mut kvs: Vec<(String, Value)>,
    split_keys: Vec<(String, Value)>,
    tensors: Vec<PlannedTensor>,
    cap: u64,
) -> Result<FilePlan, FixtureError> {
    let align = file_alignment(&kvs).map_err(|source| FixtureError::Alignment {
        set: "target",
        source,
    })?;
    let mut shards: Vec<Range<usize>> = Vec::new();
    let (mut start, mut bytes) = (0usize, 0u64);
    for (i, t) in tensors.iter().enumerate() {
        let b = t.nbytes.next_multiple_of(align);
        if i > start && bytes + b > cap {
            shards.push(start..i);
            (start, bytes) = (i, 0);
        }
        bytes += b;
    }
    shards.push(start..tensors.len());
    let n = shards.len();
    for key in [SPLIT_NO, SPLIT_COUNT, SPLIT_TENSORS] {
        if !split_keys.iter().any(|(k, _)| k == key) {
            return Err(meta(key, "is absent from the source's first shard"));
        }
    }
    for (k, v) in &mut kvs {
        match k.as_str() {
            SPLIT_NO => *v = int_like(k, v, 0)?,
            SPLIT_COUNT => *v = int_like(k, v, n as u64)?,
            SPLIT_TENSORS => *v = int_like(k, v, tensors.len() as u64)?,
            _ => {}
        }
    }
    let split_keys = split_keys
        .into_iter()
        .map(|(k, v)| {
            let v = match k.as_str() {
                SPLIT_COUNT => int_like(&k, &v, n as u64)?,
                SPLIT_TENSORS => int_like(&k, &v, tensors.len() as u64)?,
                _ => v,
            };
            Ok((k, v))
        })
        .collect::<Result<Vec<_>, FixtureError>>()?;
    let files = (0..n)
        .map(|i| format!("{STEM}-{:05}-of-{n:05}.gguf", i + 1))
        .collect();
    let plan = FilePlan {
        kvs,
        tensors,
        shards,
        files,
        split_keys,
        align,
    };
    if n > 1 && plan.spanning_layers().is_empty() {
        return Err(FixtureError::NoSpanningLayer { shards: n });
    }
    Ok(plan)
}

/// The last `k` of `n` layers; `None` when `k > n`.
fn last_layers(n: usize, k: usize) -> Option<Range<usize>> {
    n.checked_sub(k).map(|first| first..n)
}

/// The draft fixture: the real draft's metadata with `target_layers` moved
/// to the fixture's last layers, its tensors at their shapes.
fn plan_draft(
    draft: &Split,
    n_layer: usize,
    opts: &Options,
    rules: &mut Rules,
) -> Result<FilePlan, FixtureError> {
    if !crate::arch::is_dflash(draft) {
        return Err(FixtureError::Architecture {
            got: draft.architecture().map(str::to_string),
            want: crate::arch::DFLASH,
        });
    }
    let tl_key = draft.arch_key("target_layers");
    let mut kvs = Vec::new();
    for (k, v) in draft.iter_kv() {
        if k.starts_with("split.") {
            return Err(meta(k, "the draft fixture is written as one file"));
        }
        if k != tl_key {
            kvs.push((k.to_string(), v.clone()));
            continue;
        }
        let a = items(k, v)?;
        let got = a
            .iter()
            .map(|x| unsigned(k, x).map(|l| l as usize))
            .collect::<Result<Vec<_>, _>>()?;
        let last: Vec<usize> = last_layers(n_layer, got.len())
            .ok_or_else(|| {
                meta(
                    k,
                    format!("names {} layers of a {n_layer}-layer source", got.len()),
                )
            })?
            .collect();
        if got.is_empty() || got != last {
            return Err(meta(
                k,
                format!("is {got:?}, not the source's last layers {last:?}"),
            ));
        }
        let n = LAYER_MAP.len();
        let tag = &a[0];
        let moved = last_layers(n, got.len())
            .ok_or_else(|| {
                meta(
                    k,
                    format!(
                        "names {} layers, more target layers than the fixture's {n}",
                        got.len()
                    ),
                )
            })?
            .map(|f| int_like(k, tag, f as u64))
            .collect::<Result<_, _>>()?;
        kvs.push((k.to_string(), Value::Array(moved)));
    }
    if !kvs.iter().any(|(k, _)| *k == tl_key) {
        return Err(meta(&tl_key, "is absent"));
    }
    let align = file_alignment(&kvs).map_err(|source| FixtureError::Alignment {
        set: "draft",
        source,
    })?;
    kvs.extend(fixture_keys(opts, header_sha256(draft)));
    let tensors = draft
        .iter_tensors()
        .map(|(_, t)| planned(rules, t.name.clone(), &t.name, None, t.dims.clone(), t.ty))
        .collect::<Result<Vec<_>, _>>()?;
    let tensors = apply_subset(tensors, opts.draft_tensors.as_ref(), &mut kvs)?;
    let n = tensors.len();
    Ok(FilePlan {
        kvs,
        tensors,
        shards: vec![Range { start: 0, end: n }],
        files: vec![DRAFT_FILE.to_string()],
        split_keys: Vec::new(),
        align,
    })
}

// ------------------------------------------------------------------- generate

/// One written tensor.
#[derive(Clone, Debug)]
pub struct TensorStat {
    pub name: String,
    pub ty: GgmlType,
    pub bytes: u64,
    pub file: String,
    /// Seconds generating its bytes, and writing them.
    pub gen_secs: f64,
    pub write_secs: f64,
}

/// What `generate` did.
#[derive(Clone, Debug)]
pub struct GenerateStats {
    pub out: PathBuf,
    pub tensors: usize,
    pub bytes: u64,
    pub file_bytes: u64,
    pub gen_secs: f64,
    pub write_secs: f64,
    pub sync_secs: f64,
    pub secs: f64,
}

/// Rename directory `from` to `to`, refused when `to` exists.
fn rename_new(from: &Path, to: &Path) -> Result<(), FixtureError> {
    fileio::rename_noreplace(from, to).map_err(|e| match e.kind() {
        io::ErrorKind::AlreadyExists => FixtureError::Exists {
            path: to.to_path_buf(),
        },
        _ => io_err(to, "rename", e),
    })
}

/// Write the fixture of `source` (and of `draft`) into directory `out`,
/// which must not exist: the files go into `<out>.tmp.<pid>`, each synced,
/// and the directory is renamed to `out` once every file is complete. On a
/// failure the temporary directory is removed.
pub fn generate(
    source: &Split,
    draft: Option<&Split>,
    out: &Path,
    opts: &Options,
    progress: &mut dyn FnMut(&TensorStat),
) -> Result<GenerateStats, FixtureError> {
    let t0 = Instant::now();
    if out.try_exists().map_err(|e| io_err(out, "stat", e))? {
        return Err(FixtureError::Exists {
            path: out.to_path_buf(),
        });
    }
    let plan = plan(source, draft, opts)?;
    let mut files = plan.target.layouts()?;
    let mut sets = vec![(&plan.target, files.len())];
    if let Some(d) = &plan.draft {
        let l = d.layouts()?;
        sets.push((d, l.len()));
        files.extend(l);
    }
    let need: u64 = files.iter().map(|(_, l)| l.file_len()).sum();
    let parent = match out.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let free = fileio::free_bytes(parent).map_err(|e| io_err(parent, "statvfs", e))?;
    if free < need {
        return Err(FixtureError::Space {
            path: parent.to_path_buf(),
            need,
            free,
        });
    }
    let mut tmp = out.as_os_str().to_owned();
    tmp.push(format!(".tmp.{}", std::process::id()));
    let tmp = PathBuf::from(tmp);
    std::fs::create_dir(&tmp).map_err(|e| match e.kind() {
        io::ErrorKind::AlreadyExists => FixtureError::Exists { path: tmp.clone() },
        _ => io_err(&tmp, "create the directory", e),
    })?;
    let written = write_files(&sets, files, &tmp, opts.seed, progress)
        .and_then(|stats| rename_new(&tmp, out).map(|()| stats))
        .and_then(|stats| {
            File::open(parent)
                .and_then(|d| d.sync_all())
                .map_err(|e| io_err(parent, "fsync", e))?;
            Ok(stats)
        });
    let w = match written {
        Ok(s) => s,
        Err(error) => {
            if let Err(cleanup) = std::fs::remove_dir_all(&tmp)
                && cleanup.kind() != io::ErrorKind::NotFound
            {
                return Err(FixtureError::Abandoned {
                    error: Box::new(error),
                    tmp,
                    cleanup,
                });
            }
            return Err(error);
        }
    };
    Ok(GenerateStats {
        out: out.to_path_buf(),
        tensors: w.tensors,
        bytes: w.bytes,
        file_bytes: need,
        gen_secs: w.gen_secs,
        write_secs: w.write_secs,
        sync_secs: w.sync_secs,
        secs: t0.elapsed().as_secs_f64(),
    })
}

/// What [`write_files`] wrote: tensors, their bytes, and the seconds spent
/// generating, writing and syncing them.
#[derive(Clone, Copy, Debug, Default)]
struct Written {
    tensors: usize,
    bytes: u64,
    gen_secs: f64,
    write_secs: f64,
    sync_secs: f64,
}

/// Every file of `sets` (each with its count of `files`) into `dir`, one
/// buffer of the largest tensor's size reused for every tensor.
fn write_files(
    sets: &[(&FilePlan, usize)],
    files: Vec<(String, Layout)>,
    dir: &Path,
    seed: u64,
    progress: &mut dyn FnMut(&TensorStat),
) -> Result<Written, FixtureError> {
    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let largest = sets
        .iter()
        .flat_map(|(p, _)| p.tensors.iter())
        .map(|t| t.nbytes as usize)
        .max()
        .unwrap_or(0);
    let mut buf = vec![0u8; largest];
    let mut done = Written::default();
    let mut files = files.into_iter();
    for (plan, n) in sets {
        for range in plan.shards.iter().take(*n) {
            let (name, layout) = files.next().expect("one layout per shard");
            let path = dir.join(&name);
            if let Some(sub) = path.parent() {
                std::fs::create_dir_all(sub).map_err(|e| io_err(sub, "create the directory", e))?;
            }
            let file = File::create_new(&path).map_err(|e| io_err(&path, "create", e))?;
            let werr = |source| FixtureError::Write {
                path: path.clone(),
                source,
            };
            let mut w =
                Writer::new(BufWriter::with_capacity(1 << 20, file), layout).map_err(werr)?;
            for t in &plan.tensors[range.clone()] {
                let out = &mut buf[..t.nbytes as usize];
                let g = Instant::now();
                t.fill(seed, threads, out);
                let gen_secs = g.elapsed().as_secs_f64();
                let wt = Instant::now();
                w.tensor(&t.name, out).map_err(werr)?;
                let write_secs = wt.elapsed().as_secs_f64();
                done.tensors += 1;
                done.bytes += t.nbytes;
                done.gen_secs += gen_secs;
                done.write_secs += write_secs;
                progress(&TensorStat {
                    name: t.name.clone(),
                    ty: t.ty,
                    bytes: t.nbytes,
                    file: name.clone(),
                    gen_secs,
                    write_secs,
                });
            }
            let file = w
                .finish()
                .map_err(werr)?
                .into_inner()
                .map_err(|e| io_err(&path, "flush", e.into_error()))?;
            let s = Instant::now();
            file.sync_all().map_err(|e| io_err(&path, "fsync", e))?;
            done.sync_secs += s.elapsed().as_secs_f64();
        }
    }
    Ok(done)
}

// --------------------------------------------------------------------- checks

/// What a checked sample of a tensor held.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sample {
    pub blocks: usize,
    pub values: usize,
    pub sum_sq: f64,
}

impl Sample {
    pub fn rms(&self) -> f64 {
        (self.sum_sq / self.values.max(1) as f64).sqrt()
    }

    fn add(&mut self, o: Sample) {
        self.blocks += o.blocks;
        self.values += o.values;
        self.sum_sq += o.sum_sq;
    }
}

/// `x` is within `1e-3` of an integer in `range`.
fn code_in(x: f32, range: std::ops::RangeInclusive<i32>) -> bool {
    let r = x.round();
    (x - r).abs() < 1e-3 && range.contains(&(r as i32))
}

/// Check `bytes` (whole units of `t`'s type; the first is unit
/// `first_unit` of the tensor) against `t`'s rule: every block's `d` and
/// `dmin` are the rule's and lie in the window, every sub-block scale is in
/// the rule's band, and `gguf::dequant_row` of the block is the rule's
/// formula at in-range codes. Returns the sample's dequantized sums.
pub fn check_units(
    t: &PlannedTensor,
    bytes: &[u8],
    first_unit: usize,
) -> Result<Sample, FixtureError> {
    let unit = unit_bytes(t.ty);
    if !bytes.len().is_multiple_of(unit) {
        return Err(FixtureError::Tensor {
            name: t.name.clone(),
            detail: format!(
                "a sample of {} bytes from unit {first_unit} is not whole {unit}-byte units",
                bytes.len()
            ),
        });
    }
    let per = if matches!(t.rule, Rule::Uniform { .. } | Rule::Const { .. }) {
        1
    } else {
        t.ty.blck_size()
            .expect("a planned tensor's type has a block: rule_for refuses every other type")
            as usize
    };
    let mut y = vec![0f32; per];
    let mut s = Sample::default();
    for (i, blk) in bytes.chunks_exact(unit).enumerate() {
        let fault = match dequant_row(t.ty, blk, &mut y) {
            Err(e) => Some(e.to_string()),
            Ok(()) => block_fault(&t.rule, blk, &y),
        };
        if let Some(detail) = fault {
            return Err(FixtureError::Block {
                name: t.name.clone(),
                block: first_unit + i,
                detail,
            });
        }
        s.blocks += 1;
        s.values += y.len();
        s.sum_sq += y.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>();
    }
    Ok(s)
}

/// The f16 at byte `o` of `blk`, when it is `want` and in the window.
fn scale_at(blk: &[u8], o: usize, what: &str, want: u16) -> Result<f32, String> {
    let got = u16::from_le_bytes([blk[o], blk[o + 1]]);
    if got != want || !d_in_window(got) {
        return Err(format!(
            "{what} {} (bits {got:#06x}), the rule's {} in {}",
            half_to_f32(got),
            half_to_f32(want),
            d_window()
        ));
    }
    Ok(half_to_f32(got))
}

/// What is wrong with block `blk`, dequantized to `y`, under `rule`.
fn block_fault(rule: &Rule, blk: &[u8], y: &[f32]) -> Option<String> {
    let r = match rule {
        Rule::Q3K { d, band } => scale_at(blk, 108, "d", *d).and_then(|d| {
            let scales: [i32; 16] = std::array::from_fn(|j| q3k_scale(&blk[96..108], j));
            signed_fault(&scales, band, d, y, -4..=3)
        }),
        Rule::Q6K { d, band } => scale_at(blk, 208, "d", *d).and_then(|d| {
            let scales: [i32; 16] = std::array::from_fn(|j| i32::from(blk[192 + j] as i8));
            signed_fault(&scales, band, d, y, -32..=31)
        }),
        Rule::Q4K { d, dmin, band } => min_fault(blk, y, (*d, *dmin), band, Q4K_MIN_PER_SCALE, 15),
        Rule::Q5K { d, dmin, band } => min_fault(blk, y, (*d, *dmin), band, Q5K_MIN_PER_SCALE, 31),
        Rule::Q8_0 { d, band } => scale_at(blk, 0, "d", *d).and_then(|d| {
            match y.iter().find(|&&v| {
                let q = (v / d).round() as i32;
                v / d != q as f32 || !band.holds(q)
            }) {
                Some(v) => Err(format!("dequantizes {v}, not d·q with q in the band")),
                None => Ok(()),
            }
        }),
        Rule::Mxfp4 { e_lo, .. } => {
            if blk[0] != *e_lo && blk[0] != e_lo + 1 {
                Err(format!(
                    "E8M0 exponent {} is neither {e_lo} nor {}",
                    blk[0],
                    e_lo + 1
                ))
            } else {
                let d = gguf::quant::e8m0_to_f32_half(blk[0]);
                match y
                    .iter()
                    .find(|&&v| !KVALUES_MXFP4.iter().any(|&k| f32::from(k) * d == v))
                {
                    Some(v) => Err(format!("dequantizes {v}, not 2^(E-128)·kvalue")),
                    None => Ok(()),
                }
            }
        }
        Rule::Uniform { half_width, .. } => {
            let limit = half_width * (1.0 + 1.0 / 128.0);
            if y[0].is_finite() && y[0].abs() <= limit {
                Ok(())
            } else {
                Err(format!("{} is outside ±{half_width}", y[0]))
            }
        }
        Rule::Const { value } => {
            if y[0] == *value {
                Ok(())
            } else {
                Err(format!("{} is not {value}", y[0]))
            }
        }
    };
    r.err()
}

/// Sixteen signed sub-block scales in `band`, and each value of sub-block
/// `j` is `d·scale·q` with `q` in `codes`.
fn signed_fault(
    scales: &[i32; 16],
    band: &Band,
    d: f32,
    y: &[f32],
    codes: std::ops::RangeInclusive<i32>,
) -> Result<(), String> {
    for (j, (&sc, vals)) in scales.iter().zip(y.as_chunks::<16>().0).enumerate() {
        if !band.holds(sc) {
            return Err(format!("sub-block {j} scale {sc} is outside the band"));
        }
        for &v in vals {
            let ok = if sc == 0 {
                v == 0.0
            } else {
                code_in(v / (d * sc as f32), codes.clone())
            };
            if !ok {
                return Err(format!("sub-block {j} dequantizes {v}, not d·{sc}·q"));
            }
        }
    }
    Ok(())
}

/// Q4_K/Q5_K: eight (scale, min) pairs with `min = per_scale·scale` and the
/// scale in `band`; each value of sub-block `j` is `d·sc·q − dmin·m` with
/// `q` in `0..=top`.
fn min_fault(
    blk: &[u8],
    y: &[f32],
    (d, dmin): (u16, u16),
    band: &Band,
    per_scale: i32,
    top: i32,
) -> Result<(), String> {
    let d = scale_at(blk, 0, "d", d)?;
    let dmin = scale_at(blk, 2, "dmin", dmin)?;
    for (j, vals) in y.as_chunks::<32>().0.iter().enumerate() {
        let (sc, m) = scale_min_k4(j, &blk[4..16]);
        if !band.holds(sc) || m != per_scale * sc {
            return Err(format!(
                "sub-block {j} scale {sc} min {m} is outside the rule"
            ));
        }
        for &v in vals {
            let ok = if sc == 0 {
                v == 0.0
            } else {
                code_in((v + dmin * m as f32) / (d * sc as f32), 0..=top)
            };
            if !ok {
                return Err(format!(
                    "sub-block {j} dequantizes {v}, not d·{sc}·q − dmin·{m}"
                ));
            }
        }
    }
    Ok(())
}

/// The chunks a check samples: the first, the middle and the last.
pub fn sample_chunks(t: &PlannedTensor) -> Vec<usize> {
    let n = t.chunks();
    let mut c = vec![0, n / 2, n - 1];
    c.dedup();
    c
}

/// `t`'s sampled chunks in `data` (its bytes in a file) equal the
/// generator's under `seed` and pass [`check_units`]; the dequantized RMS of
/// a random tensor is within ±10 % of `1/√K`.
pub fn check_tensor(t: &PlannedTensor, data: &[u8], seed: u64) -> Result<Sample, FixtureError> {
    if data.len() as u64 != t.nbytes {
        return Err(FixtureError::Tensor {
            name: t.name.clone(),
            detail: format!("holds {} bytes, the plan {}", data.len(), t.nbytes),
        });
    }
    let unit = unit_bytes(t.ty);
    let mut s = Sample::default();
    let mut want = Vec::new();
    for c in sample_chunks(t) {
        let r = t.chunk_range(c);
        want.resize(r.len(), 0);
        t.fill_chunk(seed, c, &mut want);
        if let Some(at) = want.iter().zip(&data[r.clone()]).position(|(a, b)| a != b) {
            return Err(FixtureError::Tensor {
                name: t.name.clone(),
                detail: format!(
                    "byte {} differs from the generator's under seed {seed}",
                    r.start + at
                ),
            });
        }
        s.add(check_units(t, &data[r.clone()], r.start / unit)?);
    }
    rms_within(t, &s)?;
    Ok(s)
}

/// The sample's RMS against `t`'s `1/√K`, ±10 %.
pub fn rms_within(t: &PlannedTensor, s: &Sample) -> Result<(), FixtureError> {
    let Some(sigma) = t.sigma() else {
        return Ok(());
    };
    let ratio = s.rms() / sigma;
    if (0.9..=1.1).contains(&ratio) {
        Ok(())
    } else {
        Err(FixtureError::Tensor {
            name: t.name.clone(),
            detail: format!(
                "dequantized RMS {:.4e} is {ratio:.3}× its 1/√K {sigma:.4e}",
                s.rms()
            ),
        })
    }
}

/// Source layer `l`'s index in the fixture.
fn fixture_layer(l: usize, what: &str) -> Result<usize, FixtureError> {
    LAYER_MAP
        .iter()
        .position(|&m| m == l)
        .ok_or_else(|| FixtureError::Mismatch {
            what: what.to_string(),
            detail: format!("reads source layer {l}, which the map does not hold"),
        })
}

/// Source kind `k` with its layer references moved through the map.
fn remap_kind(k: &LayerKind, what: &str) -> Result<LayerKind, FixtureError> {
    let mut out = *k;
    if let Some(s) = k.stream {
        out.stream = Some(Stream {
            ratio: s.ratio,
            kv_source: fixture_layer(s.kv_source, what)?,
            index_key_source: fixture_layer(s.index_key_source, what)?,
            topk_source: fixture_layer(s.topk_source, what)?,
        });
    }
    if let Some(d) = k.dense {
        out.dense = Some(DenseStream {
            ratio: d.ratio,
            kv_source: fixture_layer(d.kv_source, what)?,
        });
    }
    Ok(out)
}

/// The engine reads `fixture`'s header as nine layers whose kinds are the
/// source layers' kinds, their layer references moved through the map.
pub fn check_kinds(fixture: &Split, source: &Split) -> Result<Hparams, FixtureError> {
    let fx = Hparams::read(fixture)?;
    let src = Hparams::read(source)?;
    if fx.n_layer != LAYER_MAP.len() {
        return Err(FixtureError::Mismatch {
            what: "layer count".into(),
            detail: format!("{}, not {}", fx.n_layer, LAYER_MAP.len()),
        });
    }
    for (f, &l) in LAYER_MAP.iter().enumerate() {
        let what = format!("layer {f} (source {l})");
        let want = remap_kind(&src.layers[l], &what)?;
        if fx.layers[f] != want {
            return Err(FixtureError::Mismatch {
                what,
                detail: format!("kind {:?}, the source's {want:?}", fx.layers[f]),
            });
        }
    }
    Ok(fx)
}

/// The draft fixture reads as a draft whose target layers are the fixture's
/// last ones; with both files whole, its inventory against `target` holds.
pub fn check_draft(
    draft: &Split,
    target: &Split,
    whole: bool,
) -> Result<DraftHparams, FixtureError> {
    let hp = DraftHparams::read(draft)?;
    let n = LAYER_MAP.len();
    let want: Vec<usize> = last_layers(n, hp.target_layers.len())
        .ok_or_else(|| FixtureError::Mismatch {
            what: "draft target_layers".into(),
            detail: format!(
                "{:?} is more target layers than the fixture's {n}",
                hp.target_layers
            ),
        })?
        .collect();
    if hp.target_layers != want {
        return Err(FixtureError::Mismatch {
            what: "draft target_layers".into(),
            detail: format!("{:?}, not {want:?}", hp.target_layers),
        });
    }
    if whole {
        dspark::inventory(draft, &hp, target)?.check()?;
    }
    Ok(hp)
}

/// One verified file set.
#[derive(Clone, Debug)]
pub struct VerifyStats {
    pub tensors: usize,
    pub blocks: usize,
    pub subset: bool,
}

/// The fixture keys of `split`, read back.
fn read_options(split: &Split) -> Result<Options, FixtureError> {
    let get = |k: &str| split.value(k).ok_or_else(|| meta(k, "is absent"));
    let version = get(KEY_VERSION)?.as_u64();
    if version != Some(u64::from(FIXTURE_VERSION)) {
        return Err(meta(
            KEY_VERSION,
            format!("is {version:?}, not {FIXTURE_VERSION}"),
        ));
    }
    let layers = items(KEY_SOURCE_LAYERS, get(KEY_SOURCE_LAYERS)?)?
        .iter()
        .map(|v| unsigned(KEY_SOURCE_LAYERS, v).map(|l| l as usize))
        .collect::<Result<Vec<_>, _>>()?;
    if layers != LAYER_MAP {
        return Err(meta(
            KEY_SOURCE_LAYERS,
            format!("is {layers:?}, not {LAYER_MAP:?}"),
        ));
    }
    let subset = match split.value(KEY_SUBSET) {
        None => None,
        Some(v) => Some(
            items(KEY_SUBSET, v)?
                .iter()
                .map(|n| {
                    n.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| meta(KEY_SUBSET, "holds a non-string"))
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
    };
    Ok(Options {
        seed: unsigned(KEY_SEED, get(KEY_SEED)?)?,
        card_budget: unsigned(KEY_CARD_BUDGET, get(KEY_CARD_BUDGET)?)?,
        shard_bytes: DEFAULT_SHARD_BYTES,
        tensors: subset,
        draft_tensors: None,
    })
}

/// `file`'s recorded source sha against `source`'s header.
fn same_source(file: &Split, source: &Split) -> Result<(), FixtureError> {
    let got = file
        .value(KEY_SOURCE_SHA256)
        .and_then(Value::as_str)
        .ok_or_else(|| meta(KEY_SOURCE_SHA256, "is absent or not a string"))?;
    let want = header_sha256(source);
    if got != want {
        return Err(FixtureError::SourceMismatch {
            key: KEY_SOURCE_SHA256,
            got: got.to_string(),
            want,
        });
    }
    Ok(())
}

/// `file` against `plan`: its metadata equals the plan's (split keys aside,
/// which `Split::open` checked), and its tensors are the plan's, in order,
/// at the plan's dims and types, and each passes [`check_tensor`].
fn check_file(
    file: &Split,
    plan: &FilePlan,
    seed: u64,
    progress: &mut dyn FnMut(&PlannedTensor, &Sample),
) -> Result<VerifyStats, FixtureError> {
    let got: Vec<(&str, &Value)> = file
        .iter_kv()
        .filter(|(k, _)| !k.starts_with("split."))
        .collect();
    let want: Vec<(&str, &Value)> = plan
        .kvs
        .iter()
        .filter(|(k, _)| !k.starts_with("split."))
        .map(|(k, v)| (k.as_str(), v))
        .collect();
    if got.len() != want.len() {
        return Err(FixtureError::Mismatch {
            what: "metadata".into(),
            detail: format!("{} keys, the plan {}", got.len(), want.len()),
        });
    }
    if let Some((g, w)) = got.iter().zip(&want).find(|(g, w)| g != w) {
        return Err(FixtureError::Mismatch {
            what: format!("metadata {}", w.0),
            detail: format!("the file holds {} = {:?}", g.0, short(g.1)),
        });
    }
    let tensors: Vec<_> = file.iter_tensors().collect();
    if tensors.len() != plan.tensors.len() {
        return Err(FixtureError::Mismatch {
            what: "tensor count".into(),
            detail: format!("{}, the plan {}", tensors.len(), plan.tensors.len()),
        });
    }
    let mut blocks = 0;
    for ((s, info), t) in tensors.into_iter().zip(&plan.tensors) {
        if info.name != t.name || info.dims != t.dims || info.ty != t.ty {
            return Err(FixtureError::Mismatch {
                what: format!("tensor {}", t.name),
                detail: format!("the file holds {} {} {:?}", info.name, info.ty, info.dims),
            });
        }
        let g = file.shard(s).expect("a found tensor's shard exists");
        let sample = check_tensor(t, g.data(info)?, seed)?;
        blocks += sample.blocks;
        progress(t, &sample);
    }
    Ok(VerifyStats {
        tensors: plan.tensors.len(),
        blocks,
        subset: file.value(KEY_SUBSET).is_some(),
    })
}

/// A value for an error line: arrays by length.
fn short(v: &Value) -> String {
    match v {
        Value::Array(a) => format!("[{} items]", a.len()),
        other => format!("{other:?}"),
    }
}

/// Verify the fixture whose first shard `fixture` opens against `source`,
/// and, when given, the draft fixture against the real draft. A whole
/// target also passes [`check_kinds`]; a whole draft [`check_draft`] with
/// its inventory.
pub fn verify(
    fixture: &Split,
    source: &Split,
    draft: Option<(&Split, &Split)>,
    progress: &mut dyn FnMut(&PlannedTensor, &Sample),
) -> Result<(VerifyStats, Option<VerifyStats>), FixtureError> {
    let mut opts = read_options(fixture)?;
    same_source(fixture, source)?;
    let draft_opts = match draft {
        None => None,
        Some((d, real)) => {
            let o = read_options(d)?;
            same_source(d, real)?;
            if (o.seed, o.card_budget) != (opts.seed, opts.card_budget) {
                return Err(FixtureError::Mismatch {
                    what: "draft keys".into(),
                    detail: format!(
                        "seed {} budget {}, the target's {} {}",
                        o.seed, o.card_budget, opts.seed, opts.card_budget
                    ),
                });
            }
            Some(o)
        }
    };
    opts.shard_bytes = u64::MAX;
    opts.draft_tensors = draft_opts.and_then(|o| o.tensors);
    let plan = plan(source, draft.map(|(_, real)| real), &opts)?;
    let whole = opts.tensors.is_none();
    if whole {
        check_kinds(fixture, source)?;
    }
    let target = check_file(fixture, &plan.target, opts.seed, progress)?;
    let draft_stats = match (draft, &plan.draft) {
        (Some((d, _)), Some(dp)) => {
            check_draft(d, fixture, whole && opts.draft_tensors.is_none())?;
            Some(check_file(d, dp, opts.seed, progress)?)
        }
        _ => None,
    };
    Ok((target, draft_stats))
}
