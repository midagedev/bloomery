//! The router dials' host rules: the resident bonus and the top-k cut of a routed decode step
//! (`BLOOMERY_ROUTE_RESIDENT_BONUS`, `BLOOMERY_ROUTE_TOPK`), in the arithmetic the device runs.
//! A gate's host side holds the device's picks and weights to these bit for bit; a launcher
//! takes its refusals ([`RouteDialError`]) from here.
//!
//! - **Bonus.** The selection value of expert `e` is `score[e] + (bias[e] + λ · r[e])`,
//!   `r[e]` 1 for an expert on the stage card, 0 otherwise. The top `k` are taken over those values,
//!   an equal value going to the larger expert id (the router's own tie rule). The weights of
//!   the picked set come from the unbiased scores ([`cut_weights`] with the full `k`).
//! - **Cut.** The router ranks `k` picks; the first `k'` run. Their weights are
//!   `score_j / divisor(Σ_{j<k'} score_j) · scale`, the sum in f64 in slot order narrowed to
//!   f32, the divisor the router instance's guard (`floor` selects Flint's floor, else the
//!   `1e-20` addend). The slots from `k'` weigh 0. A router whose weights carry a learned
//!   per-expert scale (ERNIE 5.5) takes [`cut_weights_scaled`]: the same quotient, then the picked
//!   expert's scale.
//! - **Mask.** A sidecar bans experts per layer ([`RouteMask`]). A banned expert's selection bias
//!   is [`BAN`] added to the file's, so the router's own top-k leaves it out and the weights
//!   renormalize over the picked set; a layer that bans so many that fewer than `k` remain is
//!   refused. The mask is one more user of the bias the bonus rides.
//! - **Census.** The picks that ran per (layer, expert) and the sum of their gate weights, as a
//!   text table ([`CensusTable`]) that `tools/route-mask.py` reads.

use std::collections::BTreeMap;
use std::fmt;
use std::fmt::Write as _;

/// What a banned expert adds to its selection value: far below any router score (V4.1's √softplus
/// and ERNIE's sigmoid sum are small non-negative numbers), finite so the router's non-finite
/// guard stays quiet, and still far below the bonus (at most 100) added to it.
pub const BAN: f32 = -1.0e30;

/// A dial a load or a launch refuses, by name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RouteDialError {
    /// `BLOOMERY_ROUTE_TOPK` above the model's own `k`.
    TopkAboveK { topk: usize, k: usize },
    /// A dial beside a feature whose rows would pick without it.
    Beside {
        dial: &'static str,
        other: &'static str,
    },
}

impl fmt::Display for RouteDialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RouteDialError::TopkAboveK { topk, k } => write!(
                f,
                "BLOOMERY_ROUTE_TOPK={topk}: the model routes {k} experts a token; it takes 1 to {k}"
            ),
            RouteDialError::Beside { dial, other } => write!(
                f,
                "{dial} beside {other}: {other}'s rows pick without the dial, so the two paths \
                 would route one token differently; unset one"
            ),
        }
    }
}

impl std::error::Error for RouteDialError {}

/// The cut `topk` (`None`: the model's own `k`) as a count of the `k` slots a token routes, or the
/// refusal. A cut equal to `k` is `None`: the router's own weights, no launch.
pub fn cut_of(topk: Option<usize>, k: usize) -> Result<Option<usize>, RouteDialError> {
    match topk {
        None => Ok(None),
        Some(t) if t == 0 || t > k => Err(RouteDialError::TopkAboveK { topk: t, k }),
        Some(t) if t == k => Ok(None),
        Some(t) => Ok(Some(t)),
    }
}

/// The router's guard on the weight stage's divisor: Flint's `2^-14` floor, else the `1e-20`
/// addend (`route_core::floor_divisor` and `renorm_divisor`).
#[inline]
fn divisor(sum: f32, floor: bool) -> f32 {
    if floor {
        sum.max(1.0 / 16384.0)
    } else {
        sum + 1e-20
    }
}

/// The weights of a token's `k` picked slots when the first `cut` run: `scores` are the picks'
/// unbiased scores in slot (rank) order; slot `j < cut` weighs
/// `scores[j] / divisor(Σ_{i<cut} scores[i]) · scale`, every later slot 0. With `cut` the whole
/// `scores.len()` these are the router's own weights.
#[must_use]
pub fn cut_weights(scores: &[f32], cut: usize, scale: f32, floor: bool) -> Vec<f32> {
    let cut = cut.min(scores.len());
    let sum: f64 = scores[..cut].iter().map(|&s| f64::from(s)).sum();
    let div = divisor(sum as f32, floor);
    scores
        .iter()
        .enumerate()
        .map(|(j, &s)| if j < cut { s / div * scale } else { 0.0 })
        .collect()
}

/// [`cut_weights`] for a router whose kept weights take a learned per-expert scale: `scales` the
/// picked slots' scales in slot order (the scale of the expert each slot picked). Slot `j < cut`
/// weighs `(scores[j] / divisor(Σ_{i<cut} scores[i]) · 1) · scales[j]`, the quotient formed first
/// (the references scale the quotient, never the score); every later slot 0.
#[must_use]
pub fn cut_weights_scaled(scores: &[f32], scales: &[f32], cut: usize, floor: bool) -> Vec<f32> {
    assert_eq!(scores.len(), scales.len(), "one scale per picked slot");
    cut_weights(scores, cut, 1.0, floor)
        .into_iter()
        .zip(scales)
        .enumerate()
        .map(|(j, (w, &s))| {
            if j < cut.min(scores.len()) {
                w * s
            } else {
                0.0
            }
        })
        .collect()
}

/// The top `k` experts by selection value `scores[e] + (bias[e] + lambda · [resident[e]])`, in rank
/// order, an equal value to the larger id; a NaN value never wins.
#[must_use]
pub fn biased_picks(
    scores: &[f32],
    bias: &[f32],
    resident: &[bool],
    lambda: f32,
    k: usize,
) -> Vec<u32> {
    assert!(
        scores.len() == bias.len() && scores.len() == resident.len(),
        "one score, bias and residency per expert"
    );
    let value = |e: usize| {
        scores[e]
            + if resident[e] {
                bias[e] + lambda
            } else {
                bias[e]
            }
    };
    let mut order: Vec<usize> = (0..scores.len()).filter(|&e| !value(e).is_nan()).collect();
    order.sort_by(|&a, &b| {
        value(b)
            .partial_cmp(&value(a))
            .expect("NaN values are filtered out")
            .then(b.cmp(&a))
    });
    order.into_iter().take(k).map(|e| e as u32).collect()
}

/// A banned-experts sidecar or a census file a load refuses, by name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MaskError {
    /// A line that is not `<layer> <id> ...`.
    Malformed { line: usize, why: String },
    /// A layer not below the model's layer count.
    LayerOutOfRange {
        line: usize,
        layer: usize,
        n_layer: usize,
    },
    /// An id not below the model's expert count.
    IdOutOfRange {
        line: usize,
        id: u32,
        n_expert: usize,
    },
    /// A layer given by two lines.
    DuplicateLayer { line: usize, layer: usize },
    /// A layer that bans so many experts that fewer than the model's `k` remain.
    TooFewLeft {
        layer: usize,
        banned: usize,
        n_expert: usize,
        k: usize,
    },
    /// A layer with no routed block (a dense layer).
    NotRouted { layer: usize },
}

impl fmt::Display for MaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MaskError::Malformed { line, why } => {
                write!(f, "BLOOMERY_ROUTE_MASK line {line}: {why}")
            }
            MaskError::LayerOutOfRange {
                line,
                layer,
                n_layer,
            } => write!(
                f,
                "BLOOMERY_ROUTE_MASK line {line}: layer {layer} of a model of {n_layer} layers"
            ),
            MaskError::IdOutOfRange { line, id, n_expert } => write!(
                f,
                "BLOOMERY_ROUTE_MASK line {line}: expert {id} of a router of {n_expert} experts"
            ),
            MaskError::DuplicateLayer { line, layer } => write!(
                f,
                "BLOOMERY_ROUTE_MASK line {line}: layer {layer} is given by an earlier line"
            ),
            MaskError::TooFewLeft {
                layer,
                banned,
                n_expert,
                k,
            } => write!(
                f,
                "BLOOMERY_ROUTE_MASK: layer {layer} bans {banned} of {n_expert} experts, leaving \
                 fewer than the {k} the router picks a token"
            ),
            MaskError::NotRouted { layer } => write!(
                f,
                "BLOOMERY_ROUTE_MASK: layer {layer} has no routed block (a dense layer)"
            ),
        }
    }
}

impl std::error::Error for MaskError {}

/// The shape a mask is read against: the model's layers, routed experts and picks a token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskShape {
    pub n_layer: usize,
    pub n_expert: usize,
    pub k: usize,
}

/// The experts a sidecar bans, per layer ([`RouteMask::parse`]). A layer with no line bans none.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RouteMask {
    /// Layer to its banned ids, ascending, each once, never empty.
    layers: BTreeMap<usize, Vec<u32>>,
}

impl RouteMask {
    /// The sidecar's text: one line a layer, `<layer> <id> <id> ...`, whitespace-separated
    /// decimal numbers; blank lines and lines starting with `#` are skipped. Refused by name: a
    /// token that is not a whole number, a line with a layer and no id, a layer or an id out of
    /// range, a layer given twice, an id twice in a line, and a layer left with fewer than `k`
    /// experts.
    pub fn parse(text: &str, shape: MaskShape) -> Result<RouteMask, MaskError> {
        let mut layers = BTreeMap::new();
        for (i, raw) in text.lines().enumerate() {
            let line = i + 1;
            let raw = raw.trim();
            if raw.is_empty() || raw.starts_with('#') {
                continue;
            }
            let bad = |why: String| MaskError::Malformed { line, why };
            let mut words = raw.split_whitespace();
            let first = words.next().unwrap_or_default();
            let layer: usize = first
                .parse()
                .map_err(|_| bad(format!("{first:?} is no layer number")))?;
            if layer >= shape.n_layer {
                return Err(MaskError::LayerOutOfRange {
                    line,
                    layer,
                    n_layer: shape.n_layer,
                });
            }
            if layers.contains_key(&layer) {
                return Err(MaskError::DuplicateLayer { line, layer });
            }
            let mut ids: Vec<u32> = Vec::new();
            for w in words {
                let id: u32 = w
                    .parse()
                    .map_err(|_| bad(format!("{w:?} is no expert id")))?;
                if id as usize >= shape.n_expert {
                    return Err(MaskError::IdOutOfRange {
                        line,
                        id,
                        n_expert: shape.n_expert,
                    });
                }
                if ids.contains(&id) {
                    return Err(bad(format!("expert {id} is given twice")));
                }
                ids.push(id);
            }
            if ids.is_empty() {
                return Err(bad(format!("layer {layer} has no expert id")));
            }
            ids.sort_unstable();
            if shape.n_expert - ids.len() < shape.k {
                return Err(MaskError::TooFewLeft {
                    layer,
                    banned: ids.len(),
                    n_expert: shape.n_expert,
                    k: shape.k,
                });
            }
            layers.insert(layer, ids);
        }
        Ok(RouteMask { layers })
    }

    /// The layers the mask names, ascending.
    pub fn layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.layers.keys().copied()
    }

    /// Layer `layer`'s banned ids, ascending; empty for a layer with no line.
    #[must_use]
    pub fn banned(&self, layer: usize) -> &[u32] {
        self.layers.get(&layer).map_or(&[], Vec::as_slice)
    }

    /// Experts banned over all layers.
    #[must_use]
    pub fn total(&self) -> usize {
        self.layers.values().map(Vec::len).sum()
    }

    /// Whether no layer bans anything.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// The owner's check that every named layer routes: `routed(layer)` false is refused by name.
    pub fn check_routed(&self, routed: impl Fn(usize) -> bool) -> Result<(), MaskError> {
        match self.layers().find(|&l| !routed(l)) {
            Some(layer) => Err(MaskError::NotRouted { layer }),
            None => Ok(()),
        }
    }

    /// Layer `layer`'s selection bias: the file's `base` with [`BAN`] added to each banned
    /// expert's entry. A layer with no line returns `base` as it is.
    #[must_use]
    pub fn masked_bias(&self, layer: usize, base: &[f32]) -> Vec<f32> {
        let mut out = base.to_vec();
        for &id in self.banned(layer) {
            out[id as usize] += BAN;
        }
        out
    }
}

/// The picks that ran per (layer, expert) and the sum of their gate weights: the census a run
/// writes ([`CensusTable::render`]) and `tools/route-mask.py` reads.
#[derive(Clone, Debug, PartialEq)]
pub struct CensusTable {
    pub n_layer: usize,
    pub n_expert: usize,
    pub n_used: usize,
    picks: Vec<u64>,
    weight: Vec<f64>,
}

/// The first line of a census file.
pub const CENSUS_TITLE: &str = "# route-census v1";

impl CensusTable {
    /// An empty table of `n_layer` layers of `n_expert` experts, `n_used` picks a token.
    #[must_use]
    pub fn new(n_layer: usize, n_expert: usize, n_used: usize) -> CensusTable {
        CensusTable {
            n_layer,
            n_expert,
            n_used,
            picks: vec![0; n_layer * n_expert],
            weight: vec![0.0; n_layer * n_expert],
        }
    }

    /// The table of a device's counters, `n_layer · n_expert` words each, layer-major. A length
    /// off the shape is refused by name.
    pub fn of_device(
        shape: (usize, usize, usize),
        picks: &[u32],
        weight: &[f32],
    ) -> Result<CensusTable, MaskError> {
        let (n_layer, n_expert, n_used) = shape;
        let mut t = CensusTable::new(n_layer, n_expert, n_used);
        if picks.len() < t.picks.len() || weight.len() < t.weight.len() {
            return Err(MaskError::Malformed {
                line: 0,
                why: format!(
                    "counters of {} and {} words for {n_layer} layers of {n_expert} experts",
                    picks.len(),
                    weight.len()
                ),
            });
        }
        for i in 0..t.picks.len() {
            t.picks[i] = u64::from(picks[i]);
            t.weight[i] = f64::from(weight[i]);
        }
        Ok(t)
    }

    /// The picks and weight sum of one cell.
    #[must_use]
    pub fn cell(&self, layer: usize, expert: usize) -> (u64, f64) {
        let at = layer * self.n_expert + expert;
        (self.picks[at], self.weight[at])
    }

    /// Add `picks` and `weight` to one cell.
    pub fn add(&mut self, layer: usize, expert: usize, picks: u64, weight: f64) {
        let at = layer * self.n_expert + expert;
        self.picks[at] += picks;
        self.weight[at] += weight;
    }

    /// Every cell of `other` added to this one; a table of another shape is refused by name.
    pub fn merge(&mut self, other: &CensusTable) -> Result<(), MaskError> {
        if (self.n_layer, self.n_expert, self.n_used)
            != (other.n_layer, other.n_expert, other.n_used)
        {
            return Err(MaskError::Malformed {
                line: 0,
                why: format!(
                    "a census of {}x{} k={} merged into one of {}x{} k={}",
                    other.n_layer,
                    other.n_expert,
                    other.n_used,
                    self.n_layer,
                    self.n_expert,
                    self.n_used
                ),
            });
        }
        for i in 0..self.picks.len() {
            self.picks[i] += other.picks[i];
            self.weight[i] += other.weight[i];
        }
        Ok(())
    }

    /// The file's text: the title, the three shape lines, then `<layer> <expert> <picks>
    /// <weight_sum>` for every cell with a pick, layer-major. The weight sum is the shortest
    /// text that reads back as the same `f64`.
    #[must_use]
    pub fn render(&self) -> String {
        let mut out = format!(
            "{CENSUS_TITLE}\nn_layer {}\nn_expert {}\nn_used {}\n",
            self.n_layer, self.n_expert, self.n_used
        );
        for layer in 0..self.n_layer {
            for e in 0..self.n_expert {
                let (p, w) = self.cell(layer, e);
                if p != 0 {
                    let _ = writeln!(out, "{layer} {e} {p} {w}");
                }
            }
        }
        out
    }

    /// A table read back from [`CensusTable::render`]'s text. Refused by name: a missing title or
    /// shape line, a cell line that is not four numbers, a cell out of range.
    pub fn parse(text: &str) -> Result<CensusTable, MaskError> {
        let bad = |line: usize, why: String| MaskError::Malformed { line, why };
        let mut lines = text.lines().enumerate().map(|(i, l)| (i + 1, l.trim()));
        match lines.next() {
            Some((_, CENSUS_TITLE)) => {}
            _ => return Err(bad(1, format!("the first line is not {CENSUS_TITLE:?}"))),
        }
        let mut shape = [0usize; 3];
        for (slot, key) in ["n_layer", "n_expert", "n_used"].into_iter().enumerate() {
            let (n, l) = lines
                .next()
                .ok_or_else(|| bad(0, format!("no {key} line")))?;
            let v = l
                .strip_prefix(key)
                .map(str::trim)
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| bad(n, format!("expected `{key} <number>`")))?;
            shape[slot] = v;
        }
        let mut t = CensusTable::new(shape[0], shape[1], shape[2]);
        for (n, l) in lines {
            if l.is_empty() || l.starts_with('#') {
                continue;
            }
            let w: Vec<&str> = l.split_whitespace().collect();
            let [layer, e, p, wt] = w[..] else {
                return Err(bad(
                    n,
                    "expected `<layer> <expert> <picks> <weight_sum>`".to_string(),
                ));
            };
            let (Ok(layer), Ok(e), Ok(p), Ok(wt)) = (
                layer.parse::<usize>(),
                e.parse::<usize>(),
                p.parse::<u64>(),
                wt.parse::<f64>(),
            ) else {
                return Err(bad(
                    n,
                    "a cell with a field that is not a number".to_string(),
                ));
            };
            if layer >= t.n_layer || e >= t.n_expert {
                return Err(bad(
                    n,
                    format!("cell ({layer}, {e}) outside {}x{}", t.n_layer, t.n_expert),
                ));
            }
            t.add(layer, e, p, wt);
        }
        Ok(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The full cut is the router's own rule; a cut renormalizes over the first `cut` slots and
    /// zeroes the rest; the weights of the run slots sum to `scale` up to the guard.
    #[test]
    fn a_cut_renormalizes_the_first_slots_and_zeroes_the_rest() {
        let s = [0.9f32, 0.6, 0.3, 0.2, 0.1, 0.05];
        let full = cut_weights(&s, 6, 1.5, false);
        let total: f32 = full.iter().sum();
        assert!((total - 1.5).abs() < 1e-5, "{total}");
        let w = cut_weights(&s, 4, 1.5, false);
        assert_eq!(&w[4..], &[0.0, 0.0]);
        let sum4 = f64::from(s[0]) + f64::from(s[1]) + f64::from(s[2]) + f64::from(s[3]);
        assert_eq!(w[0], s[0] / (sum4 as f32 + 1e-20) * 1.5);
        let kept: f32 = w.iter().sum();
        assert!((kept - 1.5).abs() < 1e-5, "{kept}");
        assert!(w[0] > full[0], "fewer slots weigh more each");
        let one = cut_weights(&s, 1, 1.5, false);
        assert_eq!(one[0], s[0] / (f64::from(s[0]) as f32 + 1e-20) * 1.5);
    }

    /// A floor guard keeps an all-underflow cut finite: 0 weights, not 0 / 0.
    #[test]
    fn a_floor_guard_weighs_an_all_zero_cut_zero() {
        let w = cut_weights(&[0.0, 0.0, 0.0], 2, 1.0, true);
        assert_eq!(w, [0.0, 0.0, 0.0]);
        let w = cut_weights(&[0.0, 0.0, 0.0], 2, 1.0, false);
        assert_eq!(w, [0.0, 0.0, 0.0]);
    }

    /// The cut refuses a count past `k` and zero by name; a cut of `k` is no cut.
    #[test]
    fn the_cut_of_a_count() {
        assert_eq!(cut_of(None, 6), Ok(None));
        assert_eq!(cut_of(Some(6), 6), Ok(None));
        assert_eq!(cut_of(Some(4), 6), Ok(Some(4)));
        assert_eq!(
            cut_of(Some(7), 6),
            Err(RouteDialError::TopkAboveK { topk: 7, k: 6 })
        );
        assert!(cut_of(Some(0), 6).is_err());
    }

    /// With λ 0 the picks are the file's; a modest λ moves a resident just under the cut line in;
    /// a λ above the whole spread makes every pick resident while `k` residents exist; ties go to
    /// the larger id.
    #[test]
    fn a_bonus_steers_the_picks_toward_the_card() {
        let n = 16usize;
        let scores: Vec<f32> = (0..n).map(|e| 3.0 - 0.1 * e as f32).collect();
        let bias = vec![0.0f32; n];
        let mut resident = vec![false; n];
        for e in [7usize, 9, 11, 12, 14, 15] {
            resident[e] = true;
        }
        let plain = biased_picks(&scores, &bias, &resident, 0.0, 6);
        assert_eq!(plain, [0, 1, 2, 3, 4, 5]);
        let off_by_zero = biased_picks(&scores, &bias, &vec![false; n], 5.0, 6);
        assert_eq!(off_by_zero, plain, "no resident: no change");
        let mild = biased_picks(&scores, &bias, &resident, 0.6, 6);
        assert!(mild.contains(&7) && mild.len() == 6, "{mild:?}");
        let mut got = biased_picks(&scores, &bias, &resident, 10.0, 6);
        got.sort_unstable();
        assert_eq!(got, [7, 9, 11, 12, 14, 15]);
        let tie = biased_picks(&[1.0, 1.0, 1.0], &[0.0; 3], &[false; 3], 0.0, 2);
        assert_eq!(tie, [2, 1]);
        let nan = biased_picks(&[f32::NAN, 1.0], &[0.0; 2], &[false; 2], 0.0, 2);
        assert_eq!(nan, [1]);
    }
    const SHAPE: MaskShape = MaskShape {
        n_layer: 8,
        n_expert: 16,
        k: 4,
    };

    /// A mask reads one line a layer, skips blank and comment lines, sorts the ids, and puts
    /// [`BAN`] on each banned expert's bias only.
    #[test]
    fn a_mask_reads_its_lines_and_biases_the_banned() {
        let m = RouteMask::parse("# keep\n\n3 9 2 7\n5 0\n", SHAPE).unwrap();
        assert_eq!(m.layers().collect::<Vec<_>>(), [3, 5]);
        assert_eq!(m.banned(3), [2, 7, 9]);
        assert_eq!(m.banned(4), [] as [u32; 0]);
        assert_eq!(m.total(), 4);
        let base = vec![0.25f32; 16];
        let b = m.masked_bias(3, &base);
        assert_eq!(b[2], 0.25 + BAN);
        assert_eq!(b[3], 0.25);
        assert_eq!(m.masked_bias(4, &base), base);
        assert!(m.check_routed(|l| l != 5).is_err());
        assert!(m.check_routed(|_| true).is_ok());
    }

    /// Each malformed sidecar is refused by name, never read as a smaller mask.
    #[test]
    fn a_malformed_mask_is_refused_by_name() {
        let err = |t: &str| RouteMask::parse(t, SHAPE).unwrap_err();
        assert!(matches!(err("x 1"), MaskError::Malformed { line: 1, .. }));
        assert!(matches!(err("3"), MaskError::Malformed { line: 1, .. }));
        assert!(matches!(err("3 1 q"), MaskError::Malformed { .. }));
        assert!(matches!(err("3 1 1"), MaskError::Malformed { .. }));
        assert!(matches!(err("-1 1"), MaskError::Malformed { .. }));
        assert!(matches!(
            err("8 1"),
            MaskError::LayerOutOfRange { layer: 8, .. }
        ));
        assert!(matches!(
            err("1 2\n2 16"),
            MaskError::IdOutOfRange {
                line: 2,
                id: 16,
                ..
            }
        ));
        assert!(matches!(
            err("1 2\n1 3"),
            MaskError::DuplicateLayer { line: 2, layer: 1 }
        ));
        // 16 experts, k 4: 13 banned leaves 3.
        let ids: Vec<String> = (0..13).map(|i| i.to_string()).collect();
        assert!(matches!(
            err(&format!("1 {}", ids.join(" "))),
            MaskError::TooFewLeft {
                layer: 1,
                banned: 13,
                k: 4,
                ..
            }
        ));
        let ids: Vec<String> = (0..12).map(|i| i.to_string()).collect();
        assert!(RouteMask::parse(&format!("1 {}", ids.join(" ")), SHAPE).is_ok());
    }

    /// With a mask no banned id is picked, by the same top-k the router takes, at any λ: the
    /// bonus does not lift a banned resident expert back into the picks.
    #[test]
    fn a_banned_expert_is_never_picked() {
        let n = 16usize;
        let scores: Vec<f32> = (0..n).map(|e| 3.0 - 0.1 * e as f32).collect();
        let m = RouteMask::parse(
            "0 0 1 2",
            MaskShape {
                n_layer: 1,
                n_expert: n,
                k: 6,
            },
        )
        .unwrap();
        let bias = m.masked_bias(0, &vec![0.0f32; n]);
        let mut resident = vec![false; n];
        for e in [0usize, 1, 9, 12] {
            resident[e] = true;
        }
        for lambda in [0.0f32, 0.5, 4.0, 100.0] {
            let picks = biased_picks(&scores, &bias, &resident, lambda, 6);
            assert_eq!(picks.len(), 6, "{lambda}: {picks:?}");
            assert!(picks.iter().all(|&p| p > 2), "{lambda}: {picks:?}");
        }
        let plain = biased_picks(&scores, &bias, &vec![false; n], 0.0, 6);
        assert_eq!(plain, [3, 4, 5, 6, 7, 8]);
    }

    /// The scaled cut is the plain cut's quotient times each slot's scale; at the full count and
    /// scales of 1 it is the router's own weights.
    #[test]
    fn a_scaled_cut_scales_the_quotient() {
        let s = [0.9f32, 0.6, 0.3, 0.2];
        let sc = [2.0f32, 0.5, 1.5, 3.0];
        let full = cut_weights_scaled(&s, &sc, 4, false);
        let plain = cut_weights(&s, 4, 1.0, false);
        for j in 0..4 {
            assert_eq!(full[j], plain[j] * sc[j]);
        }
        assert_eq!(
            cut_weights_scaled(&s, &[1.0; 4], 4, false),
            cut_weights(&s, 4, 1.0, false)
        );
        let c = cut_weights_scaled(&s, &sc, 2, false);
        assert_eq!(&c[2..], &[0.0, 0.0]);
        assert_eq!(c[1], cut_weights(&s, 2, 1.0, false)[1] * 0.5);
    }

    /// A census renders and reads back cell for cell, merges by addition and refuses another
    /// shape and a bad file by name.
    #[test]
    fn a_census_round_trips_and_merges() {
        let mut a = CensusTable::new(3, 8, 2);
        a.add(0, 1, 5, 2.5);
        a.add(2, 7, 1, 0.1f32 as f64);
        let text = a.render();
        assert!(text.starts_with("# route-census v1\nn_layer 3\nn_expert 8\nn_used 2\n"));
        assert_eq!(CensusTable::parse(&text).unwrap(), a);
        let mut b = CensusTable::new(3, 8, 2);
        b.add(0, 1, 2, 0.5);
        b.merge(&a).unwrap();
        assert_eq!(b.cell(0, 1), (7, 3.0));
        assert!(b.merge(&CensusTable::new(3, 9, 2)).is_err());
        assert!(CensusTable::parse("n_layer 3\n").is_err());
        assert!(CensusTable::parse(&text.replace("2 7 1", "2 8 1")).is_err());
        let dev = CensusTable::of_device((1, 2, 1), &[3, 0], &[1.5, 0.0]).unwrap();
        assert_eq!(dev.cell(0, 0), (3, 1.5));
        assert!(CensusTable::of_device((2, 2, 1), &[3, 0], &[1.5, 0.0]).is_err());
    }
}
