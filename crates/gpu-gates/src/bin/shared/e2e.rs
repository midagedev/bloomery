//! The e2e gates' shared runner: the numeric compare against ik's taps, the
//! per-layer error tables, the free and forced arms' flip accounting, the
//! step-set prologue and the clause reporting helpers. A gate includes it by
//! `#[path]` (as `shared/gate_card.rs` is) and holds its taps and ik names,
//! its pins, its structure clause, its row read-backs and its clauses.

use std::time::Instant;

use bloomery_gpu::GpuError;
use bloomery_gpu::route_core::sigmoid;
use bloomery_gpu_gates::flip::Flip;
use bloomery_gpu_gates::tier;
use bloomery_gpu_gates::{
    GateError, RefManifest, bits_equal, data_dir, ik_q8_2, ref_tensor_logical_in, verdict,
};
use refset::family::Family;

/// `‖got − want‖ / ‖want‖` in f64, normalised by `want`; infinite on a NaN or
/// an infinity in either vector or a length mismatch, so none passes a band.
pub fn rel(got: &[f32], want: &[f32]) -> f64 {
    if got.len() != want.len() || got.iter().chain(want).any(|v| !v.is_finite()) {
        return f64::INFINITY;
    }
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (&x, &y) in got.iter().zip(want) {
        num += (f64::from(x) - f64::from(y)).powi(2);
        den += f64::from(y).powi(2);
    }
    let r = (num / den.max(f64::MIN_POSITIVE)).sqrt();
    if r.is_nan() { f64::INFINITY } else { r }
}

/// `max|got − want| / max|want|` in f64, the max-norm of [`rel`]: the
/// numerator alone when `want` is all zeros; infinite on a non-finite value
/// or a length mismatch.
pub fn rel_max(got: &[f32], want: &[f32]) -> f64 {
    if got.len() != want.len() || got.iter().chain(want).any(|v| !v.is_finite()) {
        return f64::INFINITY;
    }
    let den = want.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
    let num = got.iter().zip(want).fold(0.0f64, |m, (&x, &y)| {
        m.max((f64::from(x) - f64::from(y)).abs())
    });
    if den > 0.0 { num / den } else { num }
}

/// The maximum absolute difference of two f32 vectors in f64.
pub(crate) fn max_delta(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
        .fold(0.0, f64::max)
}

/// The cosine similarity of two f32 vectors in f64.
pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

/// The first index of the largest value, as the head's argmax breaks ties.
pub fn argmax(v: &[f32]) -> u32 {
    let best = v
        .iter()
        .enumerate()
        .fold(0usize, |b, (i, &x)| if x > v[b] { i } else { b });
    best as u32
}

/// The runner-up's index: the largest value other than at `top`.
pub fn second(v: &[f32], top: u32) -> u32 {
    let best = v.iter().enumerate().fold(None::<usize>, |b, (i, &x)| {
        if i == top as usize {
            b
        } else {
            match b {
                Some(j) if v[j] >= x => Some(j),
                _ => Some(i),
            }
        }
    });
    best.unwrap_or(0) as u32
}

/// Bit equality of two logits rows.
pub fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// `‖x̂_ik − x‖ / ‖x‖`: how far ik's 8-bit activation of `x` (q8_2,
/// blocks of 32, a bf16 scale) sits from the f32 `x` our kernels read.
pub fn quant_gap(x: &[f32]) -> f64 {
    let whole = x.len() / ik_q8_2::QK * ik_q8_2::QK;
    rel(&ik_q8_2::reconstruct(&x[..whole]), &x[..whole])
}

/// A set's tap `name` in its logical order.
pub fn tap(man: &RefManifest, name: &str) -> Result<Vec<f32>, GateError> {
    Ok(ref_tensor_logical_in(&man.dir, man.tensor(name, 0)?)?)
}

/// A set's tap `name`, occurrence `occ`, in its logical order.
pub fn tap_at(man: &RefManifest, name: &str, occ: u32) -> Result<Vec<f32>, GateError> {
    Ok(ref_tensor_logical_in(&man.dir, man.tensor(name, occ)?)?)
}

/// ik's last logits row of a set.
pub fn ik_last(man: &RefManifest, vocab: usize) -> Result<Vec<f32>, GateError> {
    let ik = tap(man, "result_output")?;
    let at = ik
        .len()
        .checked_sub(vocab)
        .ok_or("result_output holds no row")?;
    Ok(ik[at..].to_vec())
}

/// Each layer's relative distance at every position `taps` holds against
/// ik's `l_out-L` (`None` where ik kept no row for it): layers `0..n_layer`,
/// one streams row `row` values each. A tap shorter than a layer's row is a
/// panic, not a silently dropped position.
pub fn layer_table(
    man: &RefManifest,
    taps: &[Vec<f32>],
    row: usize,
    n_layer: usize,
) -> Result<Vec<Vec<Option<f64>>>, GateError> {
    let n = taps.len();
    (0..n_layer)
        .map(|l| {
            let ik = tap(man, &format!("l_out-{l}"))?;
            let kept = ik.len() / row;
            Ok(taps
                .iter()
                .enumerate()
                .map(|(t, ours)| {
                    let i = (t + kept).checked_sub(n)?;
                    Some(rel(
                        &ours[l * row..(l + 1) * row],
                        &ik[i * row..(i + 1) * row],
                    ))
                })
                .collect())
        })
        .collect()
}

/// Each row's worst entry of a [`layer_table`] and the tap it was met at.
pub fn worst_of(table: &[Vec<Option<f64>>]) -> Vec<(f64, usize)> {
    table
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .filter_map(|(t, e)| e.map(|e| (e, t)))
                .fold((0.0f64, 0usize), |w, x| if x.0 > w.0 { x } else { w })
        })
        .collect()
}

/// Each layer's worst relative distance over the positions `taps` holds
/// against ik's `l_out-L`: [`worst_of`] of its [`layer_table`].
pub fn layer_rels(
    man: &RefManifest,
    taps: &[Vec<f32>],
    row: usize,
    n_layer: usize,
) -> Result<Vec<(f64, usize)>, GateError> {
    Ok(worst_of(&layer_table(man, taps, row, n_layer)?))
}

/// The worst layer and the first of layers `0..held` past `band`, printed;
/// whether every one of those is inside it.
pub fn print_layers(what: &str, rels: &[(f64, usize)], held: usize, band: f64) -> bool {
    for (l, &(e, t)) in rels.iter().enumerate() {
        if l % 8 == 0 || l == rels.len() - 1 || (l < held && e > band) {
            println!("{what} layer={l} l_out_rel={e:.3e} at tap {t}");
        }
    }
    match rels.iter().take(held).position(|&(e, _)| e > band) {
        Some(l) => {
            println!("{what}: first layer past the band {band:.2}: {l}");
            false
        }
        None => true,
    }
}

/// Whether a flip at `(l', t')` lies on the path of layer `l`'s output at
/// position `t`: every layer from `l'` on reads it at `t'`, and every later
/// position through the mixers' stores (`same_position` holds it to `t'`).
pub fn on_path(flips: &[Flip], l: usize, t: usize, same_position: bool) -> bool {
    flips.iter().any(|f| {
        f.layer <= l
            && if same_position {
                f.token == t
            } else {
                f.token <= t
            }
    })
}

/// Every flip's line under `arm`, and whether all are allowed within `cap`.
pub fn flips_report(flips: &[Flip], arm: &str, cap: f64) -> bool {
    for f in flips {
        println!("{}", f.line(arm, cap));
    }
    flips.iter().all(|f| f.allowed(cap))
}

/// A [`layer_table`]'s worst entry off every flip's path ([`on_path`]), with
/// the layer and position it was met at, and how many entries a flip's path
/// covers.
pub fn worst_off_path(
    table: &[Vec<Option<f64>>],
    flips: &[Flip],
    same_position: bool,
) -> (f64, usize, usize, usize) {
    let mut w = (0.0f64, 0usize, 0usize, 0usize);
    for (l, row) in table.iter().enumerate() {
        for (t, e) in row.iter().enumerate() {
            let Some(e) = *e else { continue };
            if on_path(flips, l, t, same_position) {
                w.3 += 1;
            } else if e > w.0 {
                (w.0, w.1, w.2) = (e, l, t);
            }
        }
    }
    w
}

/// By position `0..taps`, the first layer a flip lies on the path of
/// ([`on_path`] through the mixers' stores; `n_layer` where none does).
pub fn first_flip_layers(flips: &[Flip], taps: usize, n_layer: usize) -> Vec<usize> {
    (0..taps)
        .map(|t| {
            (0..n_layer)
                .find(|&l| on_path(flips, l, t, false))
                .unwrap_or(n_layer)
        })
        .collect()
}

/// A step set opened for the (t) clause: its manifest, the step's position
/// and token, its prefill ids, and the layer the band holds to (0 without a
/// band). With `band` — the batch set's tokens and the first layer a flip
/// lies on the path of its last position — the set's prefill and step must
/// be those tokens, which the free arm routed. The set's path is the family's [`Family::path_in`]: a
/// fixture variant's sets carry its tag.
pub fn set_open(
    (name, family): (&str, &Family),
    band: Option<(&[u32], usize)>,
) -> Result<(RefManifest, u32, u32, Vec<u32>, usize), GateError> {
    let man = RefManifest::open(&family.path_in(&data_dir(), name), family)?;
    let (pos, step, prefill) = man.step()?;
    let (pos, step, prefill) = (pos, step.to_vec(), prefill.to_vec());
    let [tok] = step[..] else {
        return Err(format!("{name}: a step of {} tokens, not one", step.len()).into());
    };
    let held = match band {
        Some((toks, first)) => {
            if toks.split_last() != Some((&tok, &prefill[..])) {
                return Err(format!(
                    "{name}: prefill {prefill:?} and step {tok} are not the batch set's {toks:?}"
                )
                .into());
            }
            first
        }
        None => 0,
    };
    Ok((man, pos, tok, prefill, held))
}

/// The step's argmax and ik's, ik's runner-up, ik's own margin between the
/// two, our distance at the two ids, and the logits row's relative distance
/// from ik's: the named-tie rule's numbers.
pub fn tie_numbers(ours: &[f32], ik: &[f32]) -> (u32, u32, u32, f64, f64, f64) {
    let (top, ik_top) = (argmax(ours), argmax(ik));
    let ik_2 = second(ik, ik_top);
    let margin = f64::from(ik[ik_top as usize]) - f64::from(ik[ik_2 as usize]);
    let dist = [ik_top, ik_2]
        .iter()
        .map(|&i| (f64::from(ours[i as usize]) - f64::from(ik[i as usize])).abs())
        .fold(0.0, f64::max);
    (top, ik_top, ik_2, margin, dist, rel(ours, ik))
}

/// One clause's elapsed line when it ends: the module header's code for it
/// and its wall in seconds, the load line's own shape.
pub fn elapsed(what: &str, t: &Instant) {
    println!(
        "clause {what} in {:.1} s (runtime value)",
        t.elapsed().as_secs_f64()
    );
}

/// A clause group's verdict, an error it ended in printed as its FAIL rather
/// than ending the gate; `group` the group's clauses print under.
pub fn clause(group: &str, what: &str, r: Result<bool, GateError>) -> bool {
    r.unwrap_or_else(|e| {
        println!("{group} {what}: ended in error \"{e}\" {}", verdict(false));
        false
    })
}

/// A clause group's verdict with its elapsed line: the work timed inside the
/// closure, then [`clause`]'s FAIL line when it ended in one. Every clause run
/// through here is self-consistency (the engine's arms against each other), so
/// both tiers run it; it is registered as one first.
pub fn clause_timed(
    group: &str,
    what: &str,
    body: impl FnOnce() -> Result<bool, GateError>,
) -> bool {
    let t = Instant::now();
    let r = tier::sc(&format!("{group} {what}")).and_then(|()| body());
    elapsed(what, &t);
    clause(group, what, r)
}

/// A refusal's text for a line: the error, or that the call ran.
pub fn outcome<T>(r: &Result<T, GpuError>) -> String {
    match r {
        Ok(_) => "ran".to_string(),
        Err(e) => format!("\"{e}\""),
    }
}

/// Whether `r` is `what`'s shape refusal whose words hold `says`.
pub fn refused_by<T>(r: &Result<T, GpuError>, what: &str, says: &str) -> bool {
    matches!(r, Err(GpuError::Shape { what: w, detail }) if *w == what && detail.contains(says))
}

/// Whether `r` is `what`'s shape refusal whose words hold every one of
/// `says`.
pub fn refused_saying<T>(r: &Result<T, GpuError>, what: &str, says: &[&str]) -> bool {
    says.iter().all(|s| refused_by(r, what, s))
}

/// The word after `--<flag>`: `None` when the flag is absent, `Some(None)`
/// when it stands at the argv's end — the caller's unknown-word arm refuses
/// that as `None`.
pub fn word_after(flag: &str) -> Option<Option<String>> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == flag)
        .map(|i| args.get(i + 1).cloned())
}

/// Worst ratio and its tap, over a layer's taps.
#[derive(Default)]
pub struct Worst {
    pub ratio: f64,
    pub tap: String,
    pub lines: Vec<String>,
}

impl Worst {
    pub fn add(&mut self, tap: &str, e: f64, gap: f64) {
        let r = e / gap.max(f64::MIN_POSITIVE);
        let r = if r.is_nan() { f64::INFINITY } else { r };
        self.lines
            .push(format!("{tap} rel={e:.3e} gap={gap:.3e} ratio={r:.2}"));
        if self.tap.is_empty() || r > self.ratio {
            self.ratio = r;
            self.tap = tap.to_string();
        }
    }
}

/// The RMS-normed rows of `x` (`k` values each) times `gain`, in f64 then
/// rounded: the input a mixer's projections read, near enough for its gap.
pub fn normed(x: &[f32], gain: &[f32], k: usize) -> Vec<f32> {
    x.chunks(k)
        .flat_map(|r| {
            let ms = r.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / k as f64;
            let s = 1.0 / (ms + 1e-6).sqrt();
            r.iter()
                .zip(gain)
                .map(move |(&v, &g)| (f64::from(v) * s * f64::from(g)) as f32)
        })
        .collect()
}

/// `rows` of `k` values of `v`, concatenated.
pub fn pick(v: &[f32], k: usize, rows: &[usize]) -> Vec<f32> {
    rows.iter()
        .flat_map(|&t| v[t * k..(t + 1) * k].to_vec())
        .collect()
}

/// `a − b`, value by value.
pub fn minus(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(&x, &y)| x - y).collect()
}

/// A step run: the argmax id and the logits after each token.
pub struct Run {
    pub ids: Vec<u32>,
    pub logits: Vec<Vec<f32>>,
}

/// Feed `ids` one token at a time from wherever `m` stands.
pub fn feed<B: bloomery_gpu::model::ChainBody>(
    m: &mut bloomery_gpu::GpuModel<B>,
    ids: &[u32],
) -> Result<Run, GpuError> {
    let mut out = Run {
        ids: Vec::new(),
        logits: Vec::new(),
    };
    for &t in ids {
        out.ids.push(m.step(&[t])?);
        out.logits.push(m.logits()?);
    }
    Ok(out)
}

/// `r`'s run, or — when a step raised the fault word — a line naming the
/// fault and `None`: the clause fails there and the gate goes on to the
/// next one. Any other error ends the gate.
pub fn or_fault<T>(c: &str, w: &str, r: Result<T, GpuError>) -> Result<Option<T>, GateError> {
    match r {
        Ok(x) => Ok(Some(x)),
        Err(GpuError::Fault { fault, .. }) => {
            println!("{c}: {w} raised the fault word: {fault} FAIL");
            Ok(None)
        }
        Err(e) => Err(e.into()),
    }
}

/// Reset, then [`feed`] `ids`, as [`or_fault`] reads it.
pub fn fresh<B: bloomery_gpu::model::ChainBody>(
    c: &str,
    w: &str,
    m: &mut bloomery_gpu::GpuModel<B>,
    ids: &[u32],
) -> Result<Option<Run>, GateError> {
    m.reset()?;
    or_fault(c, w, feed(m, ids))
}

pub fn finite(r: &Run) -> bool {
    r.logits.iter().flatten().all(|v| v.is_finite())
}

pub fn same(a: &Run, b: &Run) -> bool {
    a.ids == b.ids
        && a.logits.len() == b.logits.len()
        && a.logits
            .iter()
            .zip(&b.logits)
            .all(|(x, y)| bits_equal(x, y))
}

/// The kernel entries of `nodes` with their counts.
pub fn histogram(nodes: &[bloomery_gpu::NodeInfo]) -> std::collections::BTreeMap<String, usize> {
    let mut h = std::collections::BTreeMap::new();
    for k in nodes.iter().filter_map(|n| n.kernel.as_ref()) {
        *h.entry(k.name.clone()).or_insert(0) += 1;
    }
    h
}

/// The ids of the `k` largest values, largest first, lower id first on a tie.
pub fn top_ids(v: &[f32], k: usize) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..v.len()).collect();
    ids.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
    ids.truncate(k);
    ids
}

/// The band within which two experts' selection values may cross between two paths: each expert's score moves by
/// `σ'(z) · Δz`, with `σ'(z) = s (1 − s)` at its own score `s` and `Δz` the router-logit error, taken as `eps` (the
/// error relative to the layer's peak logit) times that peak `z_max`; the two errors add.
pub fn cross_band(eps: f32, z_max: f32, a: f32, b: f32) -> f32 {
    eps * z_max * (a * (1.0 - a) + b * (1.0 - b))
}

/// A sigmoid router's selection over one row of logits: the scores `σ(z)`, each expert's selection value
/// (`score + bias`), the experts ranked by it and the layer's peak `|z|` (the `z_max` of [`cross_band`]).
pub struct Selection {
    pub scores: Vec<f32>,
    /// The selection value of each expert.
    pub values: Vec<f32>,
    /// `(expert, value)`, larger first, the larger id first on a tie (the device's rule).
    pub rank: Vec<(u32, f32)>,
    pub z_max: f32,
}

/// The [`Selection`] of `logits` under the layer's selection `bias`, scored by the device's own sigmoid
/// ([`bloomery_gpu::route_core::sigmoid`]); the experts the shorter of the two covers.
pub fn selection_order(bias: &[f32], logits: &[f32]) -> Selection {
    let scores: Vec<f32> = logits.iter().map(|&z| sigmoid(z)).collect();
    let values: Vec<f32> = scores.iter().zip(bias).map(|(s, b)| s + b).collect();
    let mut rank: Vec<(u32, f32)> = (0..).zip(values.iter().copied()).collect();
    rank.sort_by(|a, b| b.1.total_cmp(&a.1).then(b.0.cmp(&a.0)));
    let z_max = logits.iter().fold(0.0f32, |m, z| m.max(z.abs()));
    Selection {
        scores,
        values,
        rank,
        z_max,
    }
}

/// The fixed walk: a multiplicative walk through ids the vocabulary holds.
pub fn walk(n: usize, vocab: usize) -> Vec<u32> {
    (0..n)
        .map(|i| u32::try_from(1_000 + (i * 7_919 + 13) % (vocab - 1_000)).unwrap_or(1))
        .collect()
}

#[cfg(test)]
mod rel_tests {
    use super::{rel, rel_max};

    #[test]
    fn rel_is_zero_on_equal_and_infinite_on_a_hole() {
        let a = [1.0f32, -2.0, 3.0];
        assert_eq!(rel(&a, &a), 0.0);
        assert_eq!(rel_max(&a, &a), 0.0);
        let want = [3.0f32, 4.0];
        assert_eq!(rel(&[0.0, 0.0], &want), 1.0);
        assert!(rel(&a, &a[..2]).is_infinite());
        assert!(rel_max(&a, &a[..2]).is_infinite());
        assert!(rel(&a[..2], &a).is_infinite());
        for bad in [f32::NAN, f32::INFINITY] {
            let got = [1.0f32, bad, 3.0];
            assert!(rel(&got, &a).is_infinite());
            assert!(rel(&a, &got).is_infinite());
            assert!(rel_max(&got, &a).is_infinite());
            assert!(rel_max(&a, &got).is_infinite());
        }
    }

    #[test]
    fn rel_max_is_the_max_norm_over_want() {
        // max|got - want| = 1.5, max|want| = 4.
        assert_eq!(rel_max(&[1.0, 2.5], &[1.0, 4.0]), 0.375);
        // An all-zero want leaves the numerator.
        assert_eq!(rel_max(&[0.25, -0.5], &[0.0, 0.0]), 0.5);
        // The L2 form normalises by want's L2 norm: sqrt(0.25) / 5.
        assert!((rel(&[3.5, 4.0], &[3.0, 4.0]) - 0.1).abs() < 1e-12);
    }
}
