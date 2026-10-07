//! Checking a written fixture against its plan: the metadata, every tensor's
//! sampled chunks byte for byte, each sampled block against its rule, and the
//! dequantized RMS.

use gguf::quant::{KVALUES_MXFP4, half_to_f32};
use gguf::{Split, Value, dequant_row};

use super::fill::{
    Band, Q4K_MIN_PER_SCALE, Q5K_MIN_PER_SCALE, Rule, Window, q3k_scale, scale_min_k4, unit_bytes,
};
use super::plan::{FilePlan, PlannedTensor, header_sha256, items, plan, unsigned};
use super::spec::{FixtureSpec, Options};
use super::{
    DEFAULT_SHARD_BYTES, FIXTURE_VERSION, FixtureError, KEY_CARD_BUDGET, KEY_SEED,
    KEY_SOURCE_LAYERS, KEY_SOURCE_SHA256, KEY_SUBSET, KEY_VERSION, meta,
};

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
/// `dmin` are the rule's and lie in `window`, every sub-block scale is in
/// the rule's band, and `gguf::dequant_row` of the block is the rule's
/// formula at in-range codes. Returns the sample's dequantized sums.
pub fn check_units(
    t: &PlannedTensor,
    bytes: &[u8],
    first_unit: usize,
    window: Window,
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
            Ok(()) => block_fault(&t.rule, blk, &y, window),
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

/// The f16 at byte `o` of `blk`, when it is `want` and in `window`.
fn scale_at(blk: &[u8], o: usize, what: &str, want: u16, window: Window) -> Result<f32, String> {
    let got = u16::from_le_bytes([blk[o], blk[o + 1]]);
    if got != want || !window.holds(got) {
        return Err(format!(
            "{what} {} (bits {got:#06x}), the rule's {} in {window}",
            half_to_f32(got),
            half_to_f32(want),
        ));
    }
    Ok(half_to_f32(got))
}

/// What is wrong with block `blk`, dequantized to `y`, under `rule` and
/// `window`.
fn block_fault(rule: &Rule, blk: &[u8], y: &[f32], window: Window) -> Option<String> {
    let scale_at = |o, what, want| scale_at(blk, o, what, want, window);
    let r = match rule {
        Rule::Q3K { d, band } => scale_at(108, "d", *d).and_then(|d| {
            let scales: [i32; 16] = std::array::from_fn(|j| q3k_scale(&blk[96..108], j));
            signed_fault(&scales, band, d, y, -4..=3)
        }),
        Rule::Q6K { d, band } => scale_at(208, "d", *d).and_then(|d| {
            let scales: [i32; 16] = std::array::from_fn(|j| i32::from(blk[192 + j] as i8));
            signed_fault(&scales, band, d, y, -32..=31)
        }),
        Rule::Q4K { d, dmin, band } => {
            min_fault(blk, y, (*d, *dmin), window, band, Q4K_MIN_PER_SCALE, 15)
        }
        Rule::Q5K { d, dmin, band } => {
            min_fault(blk, y, (*d, *dmin), window, band, Q5K_MIN_PER_SCALE, 31)
        }
        Rule::Q8_0 { d, band } => scale_at(0, "d", *d).and_then(|d| {
            match y.iter().find(|&&v| {
                let q = (v / d).round() as i32;
                v / d != q as f32 || !band.holds(q)
            }) {
                Some(v) => Err(format!("dequantizes {v}, not d·q with q in the band")),
                None => Ok(()),
            }
        }),
        Rule::Q5_1 { d, m } => scale_at(0, "d", *d).and_then(|d| {
            let got = u16::from_le_bytes([blk[2], blk[3]]);
            if got != *m {
                return Err(format!(
                    "m {} (bits {got:#06x}), the rule's {}",
                    half_to_f32(got),
                    half_to_f32(*m)
                ));
            }
            let m = half_to_f32(got);
            match y.iter().find(|&&v| !code_in((v - m) / d, 0..=31)) {
                Some(v) => Err(format!("dequantizes {v}, not q·d + m with q in 0..=31")),
                None => Ok(()),
            }
        }),
        Rule::Iq4Nl { d, band } => scale_at(0, "d", *d).and_then(|d| {
            match y.iter().find(|&&v| {
                let k = (v / d).round() as i32;
                d * k as f32 != v || !band.holds(k)
            }) {
                Some(v) => Err(format!(
                    "dequantizes {v}, not d·k with k a kvalue in the band"
                )),
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

/// Q4_K/Q5_K: `d` and `dmin` the rule's and in `window`, eight (scale, min)
/// pairs with `min = per_scale·scale` and the scale in `band`; each value of
/// sub-block `j` is `d·sc·q − dmin·m` with `q` in `0..=top`.
fn min_fault(
    blk: &[u8],
    y: &[f32],
    (d, dmin): (u16, u16),
    window: Window,
    band: &Band,
    per_scale: i32,
    top: i32,
) -> Result<(), String> {
    let d = scale_at(blk, 0, "d", d, window)?;
    let dmin = scale_at(blk, 2, "dmin", dmin, window)?;
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
/// generator's under `seed` and pass [`check_units`] under `window`; the
/// dequantized RMS of a random tensor is within ±10 % of `1/√K`.
pub fn check_tensor(
    t: &PlannedTensor,
    data: &[u8],
    seed: u64,
    window: Window,
) -> Result<Sample, FixtureError> {
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
        s.add(check_units(t, &data[r.clone()], r.start / unit, window)?);
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

/// One verified file set.
#[derive(Clone, Debug)]
pub struct VerifyStats {
    pub tensors: usize,
    pub blocks: usize,
    pub subset: bool,
}

/// The fixture keys of `split`, read back; its layer map must be `spec`'s.
fn read_options(spec: &FixtureSpec, split: &Split) -> Result<Options, FixtureError> {
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
    if layers != spec.layers {
        return Err(meta(
            KEY_SOURCE_LAYERS,
            format!("is {layers:?}, not {:?}", spec.layers),
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
    window: Window,
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
        let sample = check_tensor(t, g.data(info)?, seed, window)?;
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

/// Verify the fixture of `spec` whose first shard `fixture` opens against
/// `source`, and, when given, the draft fixture against the real draft. A
/// whole target also passes the family's kinds check; a whole draft the
/// draft's check with its inventory.
pub fn verify(
    spec: &FixtureSpec,
    fixture: &Split,
    source: &Split,
    draft: Option<(&Split, &Split)>,
    progress: &mut dyn FnMut(&PlannedTensor, &Sample),
) -> Result<(VerifyStats, Option<VerifyStats>), FixtureError> {
    let mut opts = read_options(spec, fixture)?;
    same_source(fixture, source)?;
    let draft_opts = match draft {
        None => None,
        Some((d, real)) => {
            let o = read_options(spec, d)?;
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
    let plan = plan(spec, source, draft.map(|(_, real)| real), &opts)?;
    let whole = opts.tensors.is_none();
    if whole {
        spec.family.check_kinds(spec, fixture, source)?;
    }
    let target = check_file(fixture, &plan.target, opts.seed, spec.window, progress)?;
    let draft_stats = match (draft, &plan.draft, &spec.draft) {
        (Some((d, _)), Some(dp), Some(ds)) => {
            ds.rules
                .check(spec, d, fixture, whole && opts.draft_tensors.is_none())?;
            Some(check_file(d, dp, opts.seed, spec.window, progress)?)
        }
        _ => None,
    };
    Ok((target, draft_stats))
}
