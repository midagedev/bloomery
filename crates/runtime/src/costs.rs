//! The prices the expert stream and the split walk read, as a pure function of what a load
//! detected: the card's rates from its driver attributes ([`CardRates::from_attributes`]) and the
//! model's expert shape ([`ExpertShape`]), the host union's costs fitted from four timings
//! ([`fit_union`]), and the cost store that keeps a timing under its device's key ([`Store`]).
//! No card or model name enters it; the device crate reads the attributes and times the host
//! (`bloomery_gpu::host::costs`).
//!
//! **Computed from the device.** The card's side of a streamed expert and of a walk's front is
//! priced from the card's int8 MMA rate (multiprocessors, clock and compute capability) and DRAM
//! rate (memory clock and bus width) and the model's own shape: a column's MACs at the int8 rate
//! times the route GEMM's share of it ([`MMA_SHARE`]), the bytes a column's glue (quantize,
//! SwiGLU, accumulate) moves and an expert's weights read once a unit, both at the DRAM rate
//! times the kernels' read share ([`READ_SHARE`]). The shares are the kernels' own, the same on
//! every card. The values are a function of the device, so two loads on it price alike and
//! nothing is stored.
//!
//! **Measured on this load's host.** The lane's rate, the host union's costs and the host's cost
//! of one enqueue are timings. A stream pick moves a token's experts between the card's sum and
//! the host's, which changes the output's bits, so a timing that differed between two loads
//! would make their greedy tokens differ. A measured value is therefore kept under its key in the
//! cost store, and every later load of the key reads it instead of timing again: the first load
//! times and writes, the next ones read the same bits.

use std::fmt::{self, Write as _};
use std::path::{Path, PathBuf};

/// A price the rule cannot read, by name.
#[derive(Clone, Debug, PartialEq)]
pub struct CostError(pub String);

impl fmt::Display for CostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CostError {}

/// The share of the card's int8 MMA peak a route GEMM over a ring half reaches: a fifth, the
/// share the MiniMax and Muse Spark rows were derived at, which A6000's fitted Qwen3.8 column
/// cost (0.16 a column, in µs) and GB10's prompt GEMMs agree with.
pub const MMA_SHARE: f64 = 0.2;

/// The share of the card's DRAM rate the route GEMMs' weight reads and the glue's row reads
/// reach: GB10's prompt routed GEMMs (2,048-row runs over 512 experts, 40 columns an expert)
/// read their weights at 0.69 of it once the columns' share is taken off, and the derived
/// MiniMax and Muse Spark rows at 0.85.
pub const READ_SHARE: f64 = 0.85;

/// The card's two rates a price divides by, from the driver's attributes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CardRates {
    /// Dense int8 MMA operations a microsecond at the card's clock.
    pub int8_ops_per_us: f64,
    /// DRAM bytes a microsecond: the memory clock's two transfers a cycle over the bus.
    pub dram_b_per_us: f64,
}

impl CardRates {
    /// The rates of a card of `sms` multiprocessors at `clock_khz`, compute capability `cc`,
    /// whose memory runs at `mem_clock_khz` over a bus of `bus_bits`. Refused by name: a zero
    /// rate attribute or compute capability major (a driver that does not report it), and
    /// any negative driver attribute, which cannot describe a rate, and a compute
    /// capability whose integer MMA rate is not priced.
    pub fn from_attributes(
        sms: i32,
        clock_khz: i32,
        cc: (i32, i32),
        (mem_clock_khz, bus_bits): (i32, i32),
    ) -> Result<CardRates, CostError> {
        let attribute = |v, name| {
            u32::try_from(v)
                .map_err(|_| CostError(format!("card attribute {name}: negative value {v}")))
        };
        let sms = attribute(sms, "multiprocessors")?;
        let clock_khz = attribute(clock_khz, "clock_khz")?;
        let cc = (
            attribute(cc.0, "compute capability major")?,
            attribute(cc.1, "compute capability minor")?,
        );
        let mem_clock_khz = attribute(mem_clock_khz, "mem_clock_khz")?;
        let bus_bits = attribute(bus_bits, "bus_bits")?;
        if cc.0 == 0 {
            return Err(CostError(format!(
                "card compute capability {}.{}: the driver reports a zero major",
                cc.0, cc.1
            )));
        }
        if sms == 0 || clock_khz == 0 || mem_clock_khz == 0 || bus_bits == 0 {
            return Err(CostError(format!(
                "a card of {sms} multiprocessors at {clock_khz} kHz, memory at \
                     {mem_clock_khz} kHz over {bus_bits} bits: the driver reports a zero"
            )));
        }
        Ok(CardRates {
            int8_ops_per_us: f64::from(sms) * int8_ops_per_clock(cc)? * f64::from(clock_khz) * 1e-3,
            dram_b_per_us: 2.0 * f64::from(mem_clock_khz) * 1e-3 * f64::from(bus_bits) / 8.0,
        })
    }
}

/// Dense int8 MMA operations one multiprocessor of compute capability `cc` retires a clock:
/// the tensor cores' rate the kernels' `mma.sync` int8 shapes run at.
fn int8_ops_per_clock(cc: (u32, u32)) -> Result<f64, CostError> {
    match cc {
        (8, 0) => Ok(4096.0),
        (9, 0) | (10, 0 | 3) => Ok(8192.0),
        (8, 6 | 9) | (12, 0 | 1) => Ok(2048.0),
        _ => Err(CostError(format!(
            "card compute capability {}.{}: unpriced int8 MMA rate",
            cc.0, cc.1
        ))),
    }
}

/// The shape of one routed expert the card's prices count: its gate, up and down of `hidden`
/// by `ff`, the router's `top_k` picks a column, and the `units` a walk's group reads a
/// streamed expert's weights for (one unit's route GEMM reads them once).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertShape {
    pub hidden: usize,
    pub ff: usize,
    pub top_k: usize,
    pub units: usize,
}

impl ExpertShape {
    /// MACs one column of the expert takes: gate, up and down.
    fn macs_per_col(self) -> f64 {
        3.0 * self.hidden as f64 * self.ff as f64
    }

    /// Bytes one column's glue moves around the expert's GEMMs: the down row the accumulate
    /// reads, the gate and up rows SwiGLU reads and the q8 row it writes, and the column's
    /// share of its row's input quantize and the shared add (one a row over its `top_k`
    /// picks).
    fn glue_b_per_col(self) -> f64 {
        let (h, f) = (self.hidden as f64, self.ff as f64);
        let q8 = 34.0 / 32.0;
        h * 4.0 + 2.0 * f * 4.0 + f * q8 + (2.0 * h * 4.0 + h * q8) / self.top_k as f64
    }
}

/// The card's fixed and per-column cost of one streamed expert of `expert_b` bytes and shape
/// `shape` on a card of `rates`, in µs: its weights read once a unit, and a column's MACs and
/// glue bytes.
#[must_use]
pub fn card_costs(rates: CardRates, shape: ExpertShape, expert_b: u64) -> (f64, f64) {
    let read = rates.dram_b_per_us * READ_SHARE;
    let fixed = expert_b as f64 * shape.units as f64 / read;
    let per_col = 2.0 * shape.macs_per_col() / (rates.int8_ops_per_us * MMA_SHARE)
        + shape.glue_b_per_col() / read;
    (fixed, per_col)
}

/// The host union's costs a fit gives ([`fit_union`]), in µs: a call's own, an expert's
/// weight listing, a column of one expert, and what a split adds an expert it lists in
/// both calls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnionCosts {
    pub call_us: f64,
    pub listing_us: f64,
    pub per_col_us: f64,
    pub split_us: f64,
}

/// Resolve a deferred union's shape and quant identity on first use. Both the loader and
/// consumer must succeed before the request is consumed; a load with no host union consumes
/// it without calling the consumer. An absent request does no work.
pub fn resolve_union_costs<T, E>(
    request: &mut Option<(ExpertShape, String)>,
    load: impl FnOnce(ExpertShape, &str) -> Result<Option<T>, E>,
    apply: impl FnOnce(&T) -> Result<(), E>,
) -> Result<(), E> {
    let Some((shape, tag)) = request.as_ref() else {
        return Ok(());
    };
    if let Some(costs) = load(*shape, tag)? {
        apply(&costs)?;
    }
    *request = None;
    Ok(())
}

/// The four timed union calls [`fit_union`] reads, in µs: `n` experts a column each
/// (`one`), `n / 2` experts a column each (`half`), `n` experts `cols` columns each
/// (`wide`) and `n` experts `cols / 2` columns each (`halved`, one of a split's two calls
/// over `wide`'s columns).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnionTimes {
    pub one: f64,
    pub half: f64,
    pub wide: f64,
    pub halved: f64,
}

/// The union's costs from its timings ([`UnionTimes`]): each call's time a call's cost plus
/// each expert's listing and columns, and a split's two calls over `wide`'s columns less
/// `wide` itself the second call's cost plus what each expert adds. A listing or a split
/// share the timings put under 0 is 0 (a union whose experts cost their columns alone, as
/// a scalar one does). Refused by name: an even `cols` of at least 2 is wanted, a timing that
/// is not finite, and a column cost that is not finite and above 0 (timings too short to
/// separate).
pub fn fit_union(n: usize, cols: usize, t: UnionTimes) -> Result<UnionCosts, CostError> {
    let (n_f, half_n) = (n as f64, (n / 2) as f64);
    let per_expert_one = (t.one - t.half) / (n_f - half_n);
    let per_col_us = (t.wide - t.one) / (n_f * (cols as f64 - 1.0));
    let listing_us = (per_expert_one - per_col_us).max(0.0);
    let call_us = (t.one - n_f * per_expert_one).max(0.0);
    let split_us = ((2.0 * t.halved - t.wide - call_us) / n_f).max(0.0);
    let timed = [t.one, t.half, t.wide, t.halved];
    if n < 2
        || cols < 2
        || !cols.is_multiple_of(2)
        || timed.iter().any(|v| !v.is_finite())
        || !(per_col_us.is_finite() && per_col_us > 0.0)
    {
        return Err(CostError(format!(
            "a union fit of {n} experts at 1 and {cols} columns from {t:?}: column {per_col_us:.3}"
        )));
    }
    Ok(UnionCosts {
        call_us,
        listing_us,
        per_col_us,
        split_us,
    })
}

/// Where a load's measured values came from, for its record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// Read from the cost store under the load's key.
    Stored,
    /// Timed by this load and written to the store.
    Measured,
}

impl Source {
    /// The record's word.
    #[must_use]
    pub fn word(self) -> &'static str {
        match self {
            Source::Stored => "stored",
            Source::Measured => "measured",
        }
    }
}

/// The host CPU's identity read from `path` (the device reader passes `/proc/cpuinfo`).
/// A read failure or a file naming no CPU is refused before it becomes a persistent cache key.
fn cpu_model(path: &Path) -> Result<String, CostError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| CostError(format!("read {}: {e}", path.display())))?;
    cpu_model_of(&text).ok_or_else(|| CostError(format!("{} names no CPU model", path.display())))
}

/// The host portion of a cost key. A failed or zero online-CPU count must not become a
/// persistent identity; `path` supplies the CPU model through its existing reader, and the
/// caller supplies the effective choices of the kernels it is about to calibrate.
pub fn cpu_key(
    path: &Path,
    cpus: std::ffi::c_long,
    kernels: HostKernels,
) -> Result<String, CostError> {
    if cpus <= 0 {
        return Err(CostError(format!("online CPUs {cpus}: must be positive")));
    }
    Ok(format!(
        "cpu={} cpus={cpus} kernels={kernels:?}",
        cpu_model(path)?
    ))
}

/// Effective choices of the host kernels the calibration runs, after lever and ISA selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostKernels {
    pub q4k_avx512: bool,
    pub lane_prefetch: bool,
    pub steal: bool,
    pub defer_quant: bool,
}

/// The CPU model `cpuinfo` names: its first `model name` (x86), else its cores' distinct
/// `CPU implementer` and `CPU part` pairs in order (aarch64, which names no model). `None`
/// when it names neither.
fn cpu_model_of(cpuinfo: &str) -> Option<String> {
    let field = |l: &str, name: &str| {
        l.strip_prefix(name)
            .and_then(|l| l.split_once(':'))
            .map(|(_, v)| v.trim().replace(' ', "_"))
            .filter(|v| !v.is_empty())
    };
    if let Some(m) = cpuinfo.lines().find_map(|l| field(l, "model name")) {
        return Some(m);
    }
    let mut parts: Vec<String> = Vec::new();
    let mut implementer = None;
    for l in cpuinfo.lines() {
        if let Some(v) = field(l, "CPU implementer") {
            implementer = Some(v);
        } else if let (Some(i), Some(p)) = (&implementer, field(l, "CPU part")) {
            let pair = format!("{i}:{p}");
            if !parts.contains(&pair) {
                parts.push(pair);
            }
        }
    }
    (!parts.is_empty()).then(|| parts.join(","))
}

/// The lane calibration namespace: device/host identity, actual pinned staging slots
/// (zero for pageable), configured fill workers, source layout/transform identity, and each
/// probed expert's part sizes and held/paged source. A tiny gate probe and a model probe must
/// not share a persisted rate, nor equal-size layouts with different transforms.
#[must_use]
pub fn lane_key<'a>(
    device: &str,
    source: &str,
    staging: usize,
    fills: usize,
    probe: impl IntoIterator<Item = (&'a [usize], bool)>,
) -> String {
    let geometry: Vec<_> = probe.into_iter().collect();
    format!("{device} source={source} staging={staging} fills={fills} probe={geometry:?}")
}

/// The host layout of every routed tensor, including layers with no card stack. Tensor type
/// ids, layer indices, dimensions and host repacking all affect which host kernels run.
#[must_use]
pub fn union_source_tag<'a>(
    tensors: impl IntoIterator<Item = (Option<usize>, u32, &'a [u64])>,
    r8: bool,
) -> String {
    let quants: Vec<_> = tensors.into_iter().collect();
    format!("quants={quants:?} r8={r8}")
}

/// The union calibration namespace: device/host identity, the resident worker pool's width
/// (the union runs on it, so another width times other costs), layer `layer`'s `experts` host
/// experts at `shape`, the probe of `probe.0` experts at `probe.1` columns, and the caller's
/// quant identity. `tags.1`, when registered, identifies the whole union and takes precedence
/// over `tags.0`, the whole source's host layout.
#[must_use]
pub fn union_key(
    device: &str,
    threads: usize,
    layer: usize,
    experts: usize,
    shape: ExpertShape,
    probe: (usize, usize),
    tags: (&str, Option<&str>),
) -> String {
    let (n, cols) = probe;
    let tag = tags.1.unwrap_or(tags.0);
    format!(
        "{device} union threads={threads} layer={layer} experts={experts} hidden={} ff={} top_k={} probe={n}x{cols} {tag}",
        shape.hidden, shape.ff, shape.top_k
    )
}

/// Refuses the values `get_or` gave under `key` ([`Store::get_or`]) that no price takes: any
/// under 0, or one named in `positive` not above 0. The store checks only that a value is
/// finite, so a kept entry out of range is refused here on every load until deleted.
pub fn check_range(
    key: &str,
    names: &[&str],
    values: &[f64],
    positive: &[&str],
) -> Result<(), CostError> {
    for (name, &v) in names.iter().zip(values) {
        if v < 0.0 || (positive.contains(name) && v <= 0.0) {
            return Err(CostError(format!(
                "cost store: {name} {v} under {key} is out of range; delete the entry to recalibrate"
            )));
        }
    }
    Ok(())
}

/// The cost store: one file a key under its directory, `$BLOOMERY_COSTS` when set, else
/// `~/.cache/bloomery/costs`. A file holds its key and each value as its `f64` bits, so a
/// load that reads it prices with the bits the measuring load priced with. A key's first
/// measurement stands until its file is deleted; the next load of the key then times again.
pub struct Store {
    dir: Option<PathBuf>,
}

impl Store {
    /// The store every load reads and writes.
    #[must_use]
    pub fn shared() -> Store {
        let dir = std::env::var_os("BLOOMERY_COSTS")
            .filter(|d| !d.is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME")
                    .filter(|h| !h.is_empty())
                    .map(|h| PathBuf::from(h).join(".cache/bloomery/costs"))
            });
        Store { dir }
    }

    /// A store under `dir`.
    #[must_use]
    pub fn at(dir: PathBuf) -> Store {
        Store { dir: Some(dir) }
    }

    /// The `names.len()` values kept under `key`, or, when none are, `measure`'s, written under
    /// it: with where they came from. An invalid file under the key's name, including one
    /// holding another key or other names, is refused until deleted. The key's lock file is
    /// held from the read to the write, so two loads that miss the same key at once take one
    /// measurement: the second reads the
    /// first's. `measure`'s error is returned whole. Refused by name: no directory (neither
    /// `BLOOMERY_COSTS` nor `HOME`), and a store that cannot be locked, read or written, since a
    /// value the store does not keep would let the next load price with other bits.
    pub fn get_or<E: From<CostError>>(
        &self,
        key: &str,
        names: &[&str],
        measure: impl FnOnce() -> Result<Vec<f64>, E>,
    ) -> Result<(Vec<f64>, Source), E> {
        let dir = self.dir.as_ref().ok_or_else(|| {
            E::from(CostError(
                "cost store: no directory (BLOOMERY_COSTS or HOME)".into(),
            ))
        })?;
        let bad = |what: &str, e: std::io::Error| {
            E::from(CostError(format!(
                "cost store {}: {what}: {e}",
                dir.display()
            )))
        };
        std::fs::create_dir_all(dir).map_err(|e| bad("create", e))?;
        let path = dir.join(file_name(key));
        let lock = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(path.with_extension("lock"))
            .map_err(|e| bad("open the lock", e))?;
        lock.lock().map_err(|e| bad("lock", e))?;
        match std::fs::read_to_string(&path) {
            Ok(t) => {
                if let Some(v) = decode(&t, key, names) {
                    return Ok((v, Source::Stored));
                }
                return Err(E::from(CostError(format!(
                    "cost store {}: invalid cached entry for {key}; delete it to recalibrate",
                    path.display()
                ))));
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(bad("read", e)),
        }
        let values = measure()?;
        if values.len() != names.len() || values.iter().any(|v| !v.is_finite()) {
            return Err(E::from(CostError(format!(
                "cost store {}: invalid measured values for {key}: {values:?}, names {names:?}",
                path.display()
            ))));
        }
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        std::fs::write(&tmp, encode(key, names, &values)).map_err(|e| bad("write", e))?;
        std::fs::rename(&tmp, &path).map_err(|e| bad("rename", e))?;
        Ok((values, Source::Measured))
    }
}

/// The file a key is kept in: its FNV-1a hash.
fn file_name(key: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in key.bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    format!("{h:016x}.costs")
}

fn encode(key: &str, names: &[&str], values: &[f64]) -> String {
    let mut t = format!("key\t{key}\n");
    for (n, v) in names.iter().zip(values) {
        let _ = writeln!(t, "{n}\t{:016x}\t{v}", v.to_bits());
    }
    t
}

/// The values of `names` in a file `text` of `key`; `None` for another key, a missing name or
/// duplicate or extra name, malformed row, or a value that is not finite.
fn decode(text: &str, key: &str, names: &[&str]) -> Option<Vec<f64>> {
    let mut lines = text.lines();
    if lines.next()?.strip_prefix("key\t")? != key {
        return None;
    }
    let kept: Vec<(&str, f64)> = lines
        .map(|l| {
            let mut f = l.split('\t');
            let (n, bits, _) = (f.next()?, f.next()?, f.next()?);
            if f.next().is_some() {
                return None;
            }
            Some((n, f64::from_bits(u64::from_str_radix(bits, 16).ok()?)))
        })
        .collect::<Option<_>>()?;
    if kept.len() != names.len() {
        return None;
    }
    names
        .iter()
        .map(|n| {
            kept.iter()
                .find(|(k, _)| k == n)
                .map(|&(_, v)| v)
                .filter(|v| v.is_finite())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deferred price is used once, and a loader or consumer error keeps the request for
    /// a retry (mutant: take the request before either fallible operation).
    #[test]
    fn deferred_union_costs_commit_after_their_consumer() {
        let shape = ExpertShape {
            hidden: 32,
            ff: 64,
            top_k: 2,
            units: 1,
        };
        let planned = Some((shape, "quant".to_string()));
        let mut request = planned.clone();
        let e = resolve_union_costs::<UnionCosts, _>(
            &mut request,
            |_, _| Err(CostError("load refused".into())),
            |_| panic!("failed loads are not consumed"),
        )
        .unwrap_err();
        assert_eq!(e.0, "load refused");
        assert_eq!(request, planned);
        let prices = UnionCosts {
            call_us: 4.0,
            listing_us: 1.0,
            per_col_us: 2.0,
            split_us: 3.0,
        };
        let e = resolve_union_costs(
            &mut request,
            |s, t| {
                assert_eq!((s, t), (shape, "quant"));
                Ok(Some(prices))
            },
            |c| {
                assert_eq!(*c, prices);
                Err(CostError("consumer refused".into()))
            },
        )
        .unwrap_err();
        assert_eq!(e.0, "consumer refused");
        assert_eq!(request, planned);
        resolve_union_costs::<_, CostError>(
            &mut request,
            |_, _| Ok(Some(prices)),
            |c| {
                assert_eq!(*c, prices);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(request, None);
        resolve_union_costs::<UnionCosts, CostError>(
            &mut request,
            |_, _| panic!("a consumed request is not loaded again"),
            |_| panic!("a consumed request is not applied again"),
        )
        .unwrap();
        request = planned;
        resolve_union_costs::<UnionCosts, CostError>(
            &mut request,
            |_, _| Ok(None),
            |_| panic!("a load with no union has no prices to apply"),
        )
        .unwrap();
        assert_eq!(request, None);
    }

    /// The RTX A6000 the Qwen3.8, Qwen3.6, MiniMax and Muse Spark rows were fitted or derived
    /// on: 84 SMs at 1,800 MHz, sm_86, GDDR6 at 8,001 MHz over 384 bits.
    fn a6000() -> CardRates {
        CardRates::from_attributes(84, 1_800_000, (8, 6), (8_001_000, 384)).unwrap()
    }

    /// The rates of the card kinds the rows were set on, against their data sheets, in TOPS and
    /// GB/s: A6000 309.7 and 768, RTX 4090 660.6 and 1,008, RTX 5090 838 and 1,792,
    /// GB10 at its run clock 237 and 273
    /// (mutants: the bus in bits, one transfer a cycle). Negative attributes and a zero rate
    /// attribute or compute capability major are refused; a zero minor is valid.
    #[test]
    fn the_rates_are_the_data_sheets() {
        let close = |a: f64, b: f64| (a - b).abs() <= 0.005 * b;
        let a = a6000();
        assert!(close(a.int8_ops_per_us, 309.66e6), "{a:?}");
        assert!(close(a.dram_b_per_us, 768.1e3), "{a:?}");
        let r5090 = CardRates::from_attributes(170, 2_407_000, (12, 0), (14_001_000, 512)).unwrap();
        assert!(close(r5090.int8_ops_per_us, 838.0e6), "{r5090:?}");
        assert!(close(r5090.dram_b_per_us, 1_792.1e3), "{r5090:?}");
        let ada = CardRates::from_attributes(128, 2_520_000, (8, 9), (10_501_000, 384)).unwrap();
        assert!(close(ada.int8_ops_per_us, 660.6e6), "{ada:?}");
        assert!(close(ada.dram_b_per_us, 1_008.1e3), "{ada:?}");
        let gb10 = CardRates::from_attributes(48, 2_411_000, (12, 1), (4_266_000, 256)).unwrap();
        assert!(close(gb10.int8_ops_per_us, 237.0e6), "{gb10:?}");
        assert!(close(gb10.dram_b_per_us, 273.0e3), "{gb10:?}");
        for i in 0..6 {
            let mut attributes = [48, 2_411_000, 12, 1, 4_266_000, 256];
            attributes[i] = -1;
            let [sms, clock, major, minor, memory, bus] = attributes;
            let e =
                CardRates::from_attributes(sms, clock, (major, minor), (memory, bus)).unwrap_err();
            assert!(e.0.contains("negative value -1"), "{e}");
        }
        for i in [0, 1, 2, 4, 5] {
            let mut attributes = [48, 2_411_000, 12, 1, 4_266_000, 256];
            attributes[i] = 0;
            let [sms, clock, major, minor, memory, bus] = attributes;
            let e =
                CardRates::from_attributes(sms, clock, (major, minor), (memory, bus)).unwrap_err();
            assert!(e.0.contains("zero"), "{e}");
            if i == 2 {
                assert!(e.0.contains("compute capability"), "{e}");
            }
        }
        for cc in [(8, 0), (9, 0), (10, 0), (10, 3)] {
            assert!(CardRates::from_attributes(48, 2_411_000, cc, (4_266_000, 256)).is_ok());
        }
        for cc in [
            (7, 5),
            (8, 5),
            (8, 7),
            (9, 1),
            (10, 1),
            (10, 2),
            (10, 4),
            (11, 0),
            (12, 2),
            (13, 0),
        ] {
            let e = CardRates::from_attributes(48, 2_411_000, cc, (4_266_000, 256)).unwrap_err();
            assert!(
                e.0.contains(&format!("compute capability {}.{}", cc.0, cc.1))
                    && e.0.contains("unpriced int8 MMA rate"),
                "{e}"
            );
        }
    }

    /// The card's costs at A6000's attributes against the rows the families carried, and a
    /// GB10 run against its prompt pass's kernel times [derived]. A column's MACs at the MMA
    /// share are within 15 % of every row (Qwen3.8 0.16, Qwen3.6 0.10, MiniMax 2.0, Muse Spark
    /// 1.6); with the glue's bytes the whole column is within 15 % of MiniMax's and Muse
    /// Spark's and 17 % over Qwen3.8's (0.187), whose row counted the route's glue in its
    /// fitted slope. The fixed cost is within 15 % of the two rows derived as a read of the
    /// weights (MiniMax 90 at two units, Muse Spark 20); Qwen3.8's fitted 13.0 holds the
    /// route's run overhead over its experts, which the read prices at 4.7; the whole run that
    /// row was read from (fixed and columns together) prices within 15 % of it. On GB10 (48
    /// SMs at its run clock, its DRAM rate) a 2,048-row run over 512 resident experts of 3.072 MB
    /// at 40 columns each prices 12.65 against the pass's 14.75 (ms) a run of routed GEMMs
    /// (12.65) and the glue kernels the glue's bytes count (accumulate, SwiGLU, shared add,
    /// input quantize and remap, 2.10) (mutants: the MACs of two parts, the glue left out, the
    /// weights read once for two units).
    #[test]
    fn the_card_costs_are_the_rows_at_a6000() {
        let a = a6000();
        let off = |got: f64, row: f64| (got - row) / row;
        let within = |got: f64, row: f64| off(got, row).abs() <= 0.15;
        let shape = |hidden, ff, units| ExpertShape {
            hidden,
            ff,
            top_k: 8,
            units,
        };
        let glue = |s: ExpertShape| s.glue_b_per_col() / (a.dram_b_per_us * READ_SHARE);
        let qwen38 = ExpertShape {
            top_k: 10,
            ..shape(2560, 640, 1)
        };
        let (f38, c38) = card_costs(a, qwen38, 3_072_000);
        assert!(within(c38 - glue(qwen38), 0.16), "Qwen3.8 MACs {c38}");
        assert!((off(c38, 0.16) - 0.17).abs() < 0.01, "Qwen3.8 column {c38}");
        assert!((f38 - 4.7).abs() < 0.1, "Qwen3.8 fixed {f38}");
        // The run the fitted row was read from: the card route's 1.3 (ms) a 2,048-row run (its
        // 13.0 a card expert over 100 of them) and 0.16 a card column, at the P 4096 pick's
        // 17.4k card columns a run (704k host columns down to 105k over 17.2 layers).
        let cols = 17_400.0;
        let run = 100.0 * f38 + cols * c38;
        assert!(within(run, 1_300.0 + cols * 0.16), "Qwen3.8 run {run} µs");
        let q36 = shape(2048, 512, 1);
        let (_, c36) = card_costs(a, q36, 1_769_472);
        assert!(within(c36 - glue(q36), 0.10), "Qwen3.6 MACs {c36}");
        // MiniMax's NVFP4 expert: 6,144 by 3,072 a part at 4.5 bits.
        let (fmm, cmm) = card_costs(a, shape(6144, 3072, 2), 31_850_496);
        assert!(within(cmm, 2.0) && within(fmm, 90.0), "MiniMax {fmm} {cmm}");
        // Muse Spark's IQ2_XXS expert: 4,096 by 4,096 a part at 66 bytes a 256.
        let (fms, cms) = card_costs(a, shape(4096, 4096, 1), 12_976_128);
        assert!(
            within(cms, 1.6) && within(fms, 20.0),
            "Muse Spark {fms} {cms}"
        );
        let gb10 = CardRates::from_attributes(48, 2_411_000, (12, 1), (4_266_000, 256)).unwrap();
        let (f, c) = card_costs(gb10, qwen38, 3_072_000);
        let run_ms = (512.0 * f + 20_480.0 * c) / 1e3;
        assert!(within(run_ms, 14.75), "GB10 run {run_ms} ms");
    }

    /// The fit reads back the costs three timings were made of, and refuses timings that
    /// separate no column cost or are not finite (mutants: the half's experts as `n`, the
    /// column's count off by one, the finite timing check dropped).
    #[test]
    fn the_union_fit_reads_back_its_costs() {
        let (call, listing, col, split) = (40.0, 20.0, 3.5, 6.0);
        let t = |n: f64, c: f64| call + n * (listing + col * c);
        let times = UnionTimes {
            one: t(8.0, 1.0),
            half: t(4.0, 1.0),
            wide: t(8.0, 16.0),
            halved: (t(8.0, 16.0) + call + 8.0 * split) / 2.0,
        };
        let got = fit_union(8, 16, times).unwrap();
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        assert!(
            close(got.call_us, call) && close(got.listing_us, listing),
            "{got:?}"
        );
        assert!(
            close(got.per_col_us, col) && close(got.split_us, split),
            "{got:?}"
        );
        // A scalar union: its experts cost their columns alone, the noise under 0 read as 0.
        let scalar = |n: f64, c: f64| n * 150.0 * c;
        let got = fit_union(
            8,
            16,
            UnionTimes {
                one: scalar(8.0, 1.0) - 30.0,
                half: scalar(4.0, 1.0),
                wide: scalar(8.0, 16.0),
                halved: scalar(8.0, 8.0) - 30.0,
            },
        )
        .unwrap();
        assert!(got.listing_us == 0.0 && got.split_us == 0.0, "{got:?}");
        let flat = UnionTimes {
            one: 100.0,
            half: 100.0,
            wide: 100.0,
            halved: 100.0,
        };
        assert!(fit_union(8, 16, flat).is_err());
        // A timing that is not finite would leave a finite fit of zeroed shares.
        for i in 0..4 {
            let mut t = [times.one, times.half, times.wide, times.halved];
            t[i] = f64::NAN;
            let [one, half, wide, halved] = t;
            let bad = UnionTimes {
                one,
                half,
                wide,
                halved,
            };
            assert!(fit_union(8, 16, bad).is_err(), "sample {i} NaN");
        }
    }

    /// A first load measures and writes, the next reads the same bits without measuring, and a
    /// corrupt or colliding entry is refused without measuring (mutants: the store read
    /// skipped, a decode failure recalibrated, the bits rounded through the decimal text).
    #[test]
    fn a_second_load_reads_the_first_loads_bits() {
        let dir = std::env::temp_dir().join(format!("bloomery-costs-test-{}", std::process::id()));
        let store = Store::at(dir.clone());
        let names = ["a", "b"];
        let v = vec![0.1 + 0.2, 1.0 / 3.0];
        let (first, src) = store
            .get_or("k1", &names, || Ok::<_, CostError>(v.clone()))
            .unwrap();
        assert_eq!((first.clone(), src), (v.clone(), Source::Measured));
        let (second, src) = store
            .get_or::<CostError>("k1", &names, || panic!("a stored key is not measured"))
            .unwrap();
        assert_eq!(src, Source::Stored);
        assert!(
            second
                .iter()
                .zip(&first)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        let valid = encode("k1", &names, &v);
        for text in [
            encode("k2", &names, &v),
            "key\tk1\na\tgarbled\t0\nb\t3ff0000000000000\t1\n".into(),
            "key\tk1\na\t3ff0000000000000\t1\n".into(),
            encode("k1", &names, &[f64::INFINITY, 1.0]),
            encode("k1", &names, &[1.0, f64::NAN]),
            valid.replace("a\t3fd3333333333334", "a"),
            format!("{valid}extra\t3ff0000000000000\t1\n"),
            format!("{valid}bad row\n"),
            valid.replace("b\t", "a\t"),
        ] {
            std::fs::write(dir.join(file_name("k1")), &text).unwrap();
            let e = store
                .get_or::<CostError>("k1", &names, || panic!("an invalid entry is not measured"))
                .unwrap_err();
            assert!(
                e.0.contains("invalid cached entry for k1; delete it to recalibrate"),
                "{e}"
            );
            assert_eq!(
                std::fs::read_to_string(dir.join(file_name("k1"))).unwrap(),
                text
            );
        }
        for values in [vec![1.0], vec![1.0, f64::NAN], vec![1.0, f64::INFINITY]] {
            let e = store
                .get_or("bad-measure", &names, || Ok::<_, CostError>(values))
                .unwrap_err();
            assert!(e.0.contains("invalid measured values"), "{e}");
            assert!(!dir.join(file_name("bad-measure")).exists());
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    /// CPU identification names both ISA families, and refuses an unreadable or unnamed host
    /// before any cache key can use it (mutants: `unknown` on a read or parse failure).
    #[test]
    fn calibration_cpu_identity_is_named_or_refused() {
        let path =
            std::env::temp_dir().join(format!("bloomery-cpuinfo-test-{}", std::process::id()));
        let x86 = "processor\t: 0\nmodel name\t: AMD Ryzen 9 9950X\n\
                   processor\t: 1\nmodel name\t: AMD Ryzen 9 9950X\n";
        let arm = "processor\t: 0\nCPU implementer\t: 0x41\nCPU part\t: 0xd87\n\
                   processor\t: 1\nCPU implementer\t: 0x41\nCPU part\t: 0xd87\n\
                   processor\t: 2\nCPU implementer\t: 0x41\nCPU part\t: 0xd85\n";
        let base = HostKernels {
            q4k_avx512: false,
            lane_prefetch: false,
            steal: true,
            defer_quant: true,
        };
        for (text, want) in [(x86, "AMD_Ryzen_9_9950X"), (arm, "0x41:0xd87,0x41:0xd85")] {
            std::fs::write(&path, text).unwrap();
            assert_eq!(cpu_model(&path).unwrap(), want);
            assert_eq!(
                cpu_key(&path, 1, base).unwrap(),
                format!("cpu={want} cpus=1 kernels={base:?}")
            );
            let mut keys = std::collections::HashSet::new();
            for bits in 0..16 {
                let kernels = HostKernels {
                    q4k_avx512: bits & 1 != 0,
                    lane_prefetch: bits & 2 != 0,
                    steal: bits & 4 != 0,
                    defer_quant: bits & 8 != 0,
                };
                let key = cpu_key(&path, 1, kernels).unwrap();
                assert!(
                    keys.insert(key),
                    "different host kernels collided: {kernels:?}"
                );
            }
            for count in [-1, 0] {
                let e = cpu_key(&path, count, base).unwrap_err();
                assert!(
                    e.0.contains("online CPUs") && e.0.contains("positive"),
                    "{e}"
                );
            }
        }
        let e = cpu_model(&path.join("cpuinfo")).unwrap_err();
        assert!(e.0.starts_with("read ") && e.0.contains("cpuinfo"), "{e}");
        for text in [
            "",
            "processor: 0\n",
            "model name: \n",
            "CPU implementer: \nCPU part: \n",
        ] {
            std::fs::write(&path, text).unwrap();
            let e = cpu_model(&path).unwrap_err();
            assert!(e.0.contains("names no CPU model"), "{e}");
        }
        std::fs::remove_file(path).unwrap();
    }

    /// A tiny probe cannot seed a model rate; every transfer condition gets its own entry,
    /// and repeating it reads its first bits (mutant: key the lane by device/host alone).
    #[test]
    fn calibration_lanes_keep_different_transfer_conditions_apart() {
        let dir = std::env::temp_dir().join(format!("bloomery-lane-keys-{}", std::process::id()));
        let store = Store::at(dir.clone());
        let tiny = [4, 4, 4];
        let model = [1_024_000; 3];
        let planes = [
            1_024_000, 16_000, 4, 1_024_000, 16_000, 4, 1_024_000, 16_000, 4,
        ];
        let mut keys = Vec::new();
        for (i, (source, staging, fills, parts, paged, count)) in [
            ("q4", 1, 1, tiny.as_slice(), false, 1),
            ("q4", 1, 1, model.as_slice(), false, 1),
            ("q4", 0, 1, model.as_slice(), false, 1),
            ("q4", 2, 1, model.as_slice(), false, 1),
            ("q4", 1, 2, model.as_slice(), false, 1),
            ("q4", 1, 1, model.as_slice(), true, 1),
            ("q4", 1, 1, model.as_slice(), false, 2),
            ("nvfp4-blocks", 1, 1, planes.as_slice(), false, 1),
            ("nvfp4-rows", 1, 1, planes.as_slice(), false, 1),
        ]
        .into_iter()
        .enumerate()
        {
            let key = lane_key(
                "card=test cpu=test",
                source,
                staging,
                fills,
                std::iter::repeat_n((parts, paged), count),
            );
            assert!(!keys.contains(&key), "probe conditions collided: {key}");
            let v = i as f64 + 0.5;
            let (got, src) = store
                .get_or(&key, &["b_per_us"], || Ok::<_, CostError>(vec![v]))
                .unwrap();
            assert_eq!((got, src), (vec![v], Source::Measured));
            let (got, src) = store
                .get_or::<CostError>(&key, &["b_per_us"], || {
                    panic!("a repeated model probe is stored")
                })
                .unwrap();
            assert_eq!((got, src), (vec![v], Source::Stored));
            keys.push(key);
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// Pool width and quant identity separate calibrations. A registered identity wins over
    /// source layout, so stream and walk read the same prices; identical first-layer layouts
    /// cannot hide a different later-layer quant (mutants: width or registered identity lost).
    #[test]
    fn union_keys_keep_calibrations_apart() {
        let dir = std::env::temp_dir().join(format!("bloomery-union-keys-{}", std::process::id()));
        let store = Store::at(dir.clone());
        let shape = ExpertShape {
            hidden: 2048,
            ff: 768,
            top_k: 10,
            units: 1,
        };
        let key = |threads, fallback, quant| {
            union_key(
                "card=test cpu=test",
                threads,
                3,
                256,
                shape,
                (256, 2),
                (fallback, quant),
            )
        };
        let first = "parts=[4, 4, 4]";
        assert_eq!(key(1, first, Some("q4")), key(1, "walk", Some("q4")));
        let dims = &[2048, 768, 256][..];
        let source = |last, r8| union_source_tag([(Some(0), 1, dims), (Some(3), last, dims)], r8);
        let q4 = source(12, false);
        let q6 = source(14, false);
        let r8 = source(12, true);
        let wider = union_source_tag(
            [(Some(0), 1, dims), (Some(3), 12, &[4096, 768, 256])],
            false,
        );
        let names = ["call_us"];
        for (threads, fallback, quant, v) in [
            (1, first, Some("q4"), 1.5),
            (32, first, Some("q4"), 2.5),
            (1, first, Some("q6"), 3.5),
            (1, q4.as_str(), None, 4.5),
            (1, q6.as_str(), None, 5.5),
            (1, wider.as_str(), None, 6.5),
            (1, r8.as_str(), None, 7.5),
        ] {
            let key = key(threads, fallback, quant);
            let (got, src) = store
                .get_or(&key, &names, || Ok::<_, CostError>(vec![v]))
                .unwrap();
            assert_eq!((got, src), (vec![v], Source::Measured), "{key}");
            let (got, src) = store
                .get_or::<CostError>(&key, &names, || panic!("a repeated calibration is stored"))
                .unwrap();
            assert_eq!((got, src), (vec![v], Source::Stored), "{key}");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// A kept price out of range is refused: a lane rate of 0 or under 0, and a union with a
    /// negative share or a column cost of 0; 0 shares and in-range values pass (mutants: the
    /// sign check dropped, `positive` read as at least 0).
    #[test]
    fn prices_out_of_range_are_refused() {
        let union = [
            "call_us",
            "listing_us",
            "per_col_us",
            "split_us",
            "enqueue_us",
        ];
        let ok = [40.0, 0.0, 3.5, 0.0, 2.0];
        assert_eq!(check_range("k", &union, &ok, &["per_col_us"]), Ok(()));
        for (i, v) in [(1, -0.1), (2, 0.0), (2, -3.0), (4, -1.0)] {
            let mut bad = ok;
            bad[i] = v;
            let e = check_range("k", &union, &bad, &["per_col_us"]).unwrap_err();
            assert!(e.0.contains(union[i]), "{e}");
        }
        assert_eq!(
            check_range("k", &["b_per_us"], &[1.5], &["b_per_us"]),
            Ok(())
        );
        for v in [0.0, -1.5] {
            assert!(check_range("k", &["b_per_us"], &[v], &["b_per_us"]).is_err());
        }
    }

    /// Two loads that miss one key at once take one measurement and price with its bits, and a
    /// store that cannot keep a value refuses it instead of handing it out (mutants: the lock
    /// dropped, a failed write returning the measured values).
    #[test]
    fn concurrent_misses_share_one_measurement_and_an_unkept_one_is_refused() {
        let dir = std::env::temp_dir().join(format!("bloomery-costs-race-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let measured = std::sync::atomic::AtomicUsize::new(0);
        let got: Vec<(Vec<f64>, Source)> = std::thread::scope(|s| {
            let runs: Vec<_> = (0..2)
                .map(|i| {
                    let (dir, measured) = (dir.clone(), &measured);
                    s.spawn(move || {
                        Store::at(dir)
                            .get_or("race", &["v"], || {
                                measured.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                std::thread::sleep(std::time::Duration::from_millis(50));
                                Ok::<_, CostError>(vec![f64::from(i) + 0.5])
                            })
                            .unwrap()
                    })
                })
                .collect();
            runs.into_iter().map(|r| r.join().unwrap()).collect()
        });
        assert_eq!(measured.into_inner(), 1, "{got:?}");
        assert_eq!(got[0].0[0].to_bits(), got[1].0[0].to_bits(), "{got:?}");
        let mut sources: Vec<_> = got.iter().map(|g| g.1.word()).collect();
        sources.sort_unstable();
        assert_eq!(sources, ["measured", "stored"]);
        // A store whose directory is a file keeps nothing, so nothing is priced from it.
        let file = dir.join("not-a-dir");
        std::fs::write(&file, "").unwrap();
        let e = Store::at(file.join("costs"))
            .get_or("k", &["v"], || Ok::<_, CostError>(vec![1.0]))
            .unwrap_err();
        assert!(e.0.starts_with("cost store "), "{e}");
        let tmp = dir
            .join(file_name("write-failure"))
            .with_extension(format!("tmp{}", std::process::id()));
        std::fs::create_dir(&tmp).unwrap();
        let e = Store::at(dir.clone())
            .get_or("write-failure", &["v"], || Ok::<_, CostError>(vec![1.0]))
            .unwrap_err();
        assert!(e.0.contains(": write:"), "{e}");
        let _ = std::fs::remove_dir_all(dir);
    }
}
