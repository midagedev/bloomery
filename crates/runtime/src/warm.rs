//! The predictor of the step port's L3 warm (host only, no card): which of a
//! layer's host experts the pass about to be served will route to, from what
//! the passes before it routed.
//!
//! The table is `counts[l-1][a][b]`: over the kept rows it has folded, how many
//! times expert `b` of layer `l` was routed by a token that also routed expert
//! `a` of layer `l-1`, and `marg[l-1][a]` how many such tokens routed `a`. A
//! row of the table halves, counts and marginal together, when its marginal
//! reaches [`HALVE_AT`], so each row is a recency window of between half
//! of that and all of it and a count never passes it (a `u8` holds it). A
//! prompt call starts a request: [`Predictor::reset`] makes every row empty
//! at once, by an epoch each row is stamped with, not by clearing the table.
//!
//! The score of expert `b` for layer `l` is the sum, over the layer-`l-1` ids
//! `a` the open pass routed, of `counts[a][b] / marg[a]`, plus one for each
//! kept row of the pass before that routed `b` at layer `l`. [`Predictor::pick`]
//! lists the host ids whose score is above zero, highest score first, a tie to
//! the lower id, up to a cap. Scores are integers in units of
//! `1 / lcm(1..=HALVE_AT)`, so every quotient is exact and no float compare
//! decides a tie.
//!
//! A pass's routed ids ([`Predictor::note`], one expert a call) wait in a store
//! of the pass until [`Predictor::pass_end`] says which rows the pass kept; the
//! kept rows' (layer `l-1`, layer `l`) pairs are folded into the table at the
//! next pass's service of layer `l` ([`Predictor::fold`]), once a layer a pass.
//! A row that was not kept leaves no trace in the table.
//!
//! Every call after [`Predictor::new`] allocates nothing: the table, the two
//! pass stores and the scoring scratch are sized by it.

use std::fmt;

use crate::swaprule::KeptRows;

/// The marginal at which a table row halves.
pub const HALVE_AT: u8 = 16;

/// The most routed ids (rows × slots a row) one pass may note: a score's
/// largest value stays inside a `u32`.
pub const MAX_CELLS: usize = 4096;

/// `lcm(1..=HALVE_AT)`: a score's unit is its reciprocal, which every marginal
/// a row can hold divides.
const SCALE: u32 = scale();

const fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

const fn scale() -> u32 {
    let mut lcm = 1u32;
    let mut m = 1u32;
    while m <= HALVE_AT as u32 {
        lcm = lcm / gcd(lcm, m) * m;
        m += 1;
    }
    lcm
}

/// Bytes the table of `layers` layers of `experts` experts holds: the counts
/// of each pair of consecutive layers, and each row's marginal and epoch
/// stamp. The pass stores and the scoring scratch are a few kilobytes beside
/// it.
#[must_use]
pub fn table_bytes(layers: usize, experts: usize) -> usize {
    let pairs = layers.saturating_sub(1);
    pairs * experts * (experts + 1 + size_of::<u32>())
}

/// A call the predictor refuses, by name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WarmError {
    /// A shape outside what the predictor is made for.
    Shape { what: &'static str, detail: String },
    /// A routed id noted past its shape: the layer, row or slot, or an
    /// expert past the layer's.
    Note {
        layer: usize,
        row: usize,
        k: usize,
        id: u32,
        detail: &'static str,
    },
}

impl fmt::Display for WarmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WarmError::Shape { what, detail } => write!(f, "{what}: {detail}"),
            WarmError::Note {
                layer,
                row,
                k,
                id,
                detail,
            } => write!(
                f,
                "Predictor::note: layer {layer} row {row} slot {k} id {id}: {detail}"
            ),
        }
    }
}

impl std::error::Error for WarmError {}

/// The routed ids of one pass: per (row, layer) cell, `top_k` ids, the pass
/// that noted them and how many of them are in.
struct Store {
    ids: Vec<u32>,
    pass: Vec<u32>,
    noted: Vec<u8>,
}

impl Store {
    fn new(cells: usize, top_k: usize) -> Store {
        Store {
            ids: vec![0; cells * top_k],
            pass: vec![0; cells],
            noted: vec![0; cells],
        }
    }

    /// The ids of cell `cell` when pass `pass` noted all `top_k` of them.
    fn whole(&self, cell: usize, top_k: usize, pass: u32) -> Option<&[u32]> {
        (self.pass[cell] == pass && usize::from(self.noted[cell]) == top_k)
            .then(|| &self.ids[cell * top_k..][..top_k])
    }
}

/// The step port's warm predictor: see the module doc.
pub struct Predictor {
    layers: usize,
    experts: usize,
    rows: usize,
    top_k: usize,
    /// `[(layers - 1) * experts * experts]`: the pair `(l-1, l)`'s counts of
    /// `b` given `a`, row `(l-1) * experts + a`.
    counts: Vec<u8>,
    /// Per table row, its marginal and the epoch it was last written in.
    marg: Vec<u8>,
    stamp: Vec<u32>,
    epoch: u32,
    /// The open pass's store and the one before it, which of the two the open
    /// pass writes, and the passes' numbers (the open pass's, from 1).
    stores: [Store; 2],
    cur: usize,
    pass: u32,
    /// The rows the pass before the open one kept, and whether there is one.
    prev_kept: KeptRows,
    prev_valid: bool,
    /// Per layer, the pass whose fold of that layer ran.
    folded: Vec<u32>,
    score: Vec<u32>,
    cand: Vec<(u32, u32)>,
}

impl Predictor {
    /// A predictor for `layers` layers of `experts` experts, passes of up to
    /// `rows` rows routing `top_k` experts a layer. Refused by name: a shape
    /// of zero, more rows than a [`KeptRows`] mask holds, and a pass larger
    /// than [`MAX_CELLS`] ids.
    pub fn new(
        layers: usize,
        experts: usize,
        rows: usize,
        top_k: usize,
    ) -> Result<Predictor, WarmError> {
        const WHAT: &str = "Predictor::new";
        let shape = |detail: String| WarmError::Shape { what: WHAT, detail };
        if layers == 0 || experts == 0 || rows == 0 || top_k == 0 {
            return Err(shape(format!(
                "{layers} layers of {experts} experts, {rows} rows of {top_k} slots: no part of \
                 the shape may be 0"
            )));
        }
        if rows > 64 {
            return Err(shape(format!("{rows} rows: a kept-rows mask holds 64")));
        }
        if u32::try_from(experts).is_err() || top_k > usize::from(u8::MAX) {
            return Err(shape(format!(
                "{experts} experts, {top_k} slots: past what the counters and ids hold"
            )));
        }
        if rows * top_k > MAX_CELLS {
            return Err(shape(format!(
                "{rows} rows of {top_k} slots is {} ids a pass, past {MAX_CELLS}: a score would \
                 overflow",
                rows * top_k
            )));
        }
        let pairs = layers - 1;
        let cells = rows * layers;
        Ok(Predictor {
            layers,
            experts,
            rows,
            top_k,
            counts: vec![0; pairs * experts * experts],
            marg: vec![0; pairs * experts],
            stamp: vec![0; pairs * experts],
            epoch: 1,
            stores: [Store::new(cells, top_k), Store::new(cells, top_k)],
            cur: 0,
            pass: 1,
            prev_kept: KeptRows::prefix(0),
            prev_valid: false,
            folded: vec![0; layers],
            score: vec![0; experts],
            cand: Vec::with_capacity(experts),
        })
    }

    /// Layers the predictor covers.
    #[must_use]
    pub fn layers(&self) -> usize {
        self.layers
    }

    fn cell(&self, layer: usize, row: usize) -> usize {
        row * self.layers + layer
    }

    /// Slot `k` of row `row`'s routing at layer `layer` of the open pass is
    /// `id`; the slots of a cell come in order from 0, and slot 0 starts the
    /// cell over. Refused by name: a layer, row or slot past the shape, an
    /// expert past the layer's, and a slot out of order.
    pub fn note(&mut self, layer: usize, row: usize, k: usize, id: u32) -> Result<(), WarmError> {
        let bad = |detail: &'static str| WarmError::Note {
            layer,
            row,
            k,
            id,
            detail,
        };
        if layer >= self.layers {
            return Err(bad("a layer past the predictor's"));
        }
        if row >= self.rows {
            return Err(bad("a row past the predictor's"));
        }
        if k >= self.top_k {
            return Err(bad("a slot past the predictor's"));
        }
        if id as usize >= self.experts {
            return Err(bad("an expert past the layer's"));
        }
        let cell = self.cell(layer, row);
        let store = &mut self.stores[self.cur];
        if k == 0 {
            store.pass[cell] = self.pass;
            store.noted[cell] = 0;
        } else if store.pass[cell] != self.pass || usize::from(store.noted[cell]) != k {
            return Err(bad("a slot out of order"));
        }
        store.ids[cell * self.top_k + k] = id;
        store.noted[cell] += 1;
        Ok(())
    }

    /// The open pass ended keeping `kept` rows: the next pass folds those
    /// rows' pairs, and counts their routing at a layer as that layer's
    /// recent history.
    pub fn pass_end(&mut self, kept: KeptRows) {
        self.prev_kept = kept;
        self.prev_valid = true;
        self.cur ^= 1;
        self.pass += 1;
    }

    /// A new request: every table row is empty and no pass is behind the open
    /// one; the ids noted in the open pass are dropped. O(1).
    pub fn reset(&mut self) {
        self.epoch = self.epoch.checked_add(1).expect("a predictor's epoch");
        self.prev_kept = KeptRows::prefix(0);
        self.prev_valid = false;
        self.pass += 1;
    }

    /// The table row of expert `a` at layer `layer - 1` (the pair of `layer`
    /// and the layer before): its counts of the experts of `layer` and its
    /// marginal; `None` for a row that is empty (stamped by no write of this
    /// request) or a `layer` with no pair.
    #[must_use]
    pub fn row(&self, layer: usize, a: u32) -> Option<(&[u8], u8)> {
        let li = layer.checked_sub(1).filter(|&l| l < self.layers - 1)?;
        let a = usize::try_from(a).ok().filter(|&a| a < self.experts)?;
        let at = li * self.experts + a;
        (self.stamp[at] == self.epoch).then(|| {
            (
                &self.counts[at * self.experts..][..self.experts],
                self.marg[at],
            )
        })
    }

    /// Fold the pairs of layer `layer - 1` and `layer` the pass before the
    /// open one routed, for each row it kept that routed both whole, into the
    /// table: once a layer a pass (a second call of the pass does nothing), and
    /// nothing for layer 0, a layer past the predictor's or a predictor with no
    /// pass behind the open one.
    pub fn fold(&mut self, layer: usize) {
        if layer == 0 || layer >= self.layers || !self.prev_valid || self.folded[layer] == self.pass
        {
            return;
        }
        self.folded[layer] = self.pass;
        let prev_pass = self.pass - 1;
        let Predictor {
            experts,
            rows,
            top_k,
            layers,
            counts,
            marg,
            stamp,
            epoch,
            stores,
            cur,
            prev_kept,
            ..
        } = self;
        let prev = &stores[*cur ^ 1];
        let (e, top_k, li) = (*experts, *top_k, layer - 1);
        for row in 0..*rows {
            if !prev_kept.keeps(row) {
                continue;
            }
            let (Some(a_ids), Some(b_ids)) = (
                prev.whole(row * *layers + li, top_k, prev_pass),
                prev.whole(row * *layers + layer, top_k, prev_pass),
            ) else {
                continue;
            };
            for &a in a_ids {
                let at = li * e + a as usize;
                if stamp[at] != *epoch {
                    counts[at * e..][..e].fill(0);
                    marg[at] = 0;
                    stamp[at] = *epoch;
                }
                let row_counts = &mut counts[at * e..][..e];
                // An expert a cell routes twice adds twice and saturates: the
                // table is a hint, and a cell of distinct ids never reaches it.
                for &b in b_ids {
                    let c = &mut row_counts[b as usize];
                    *c = c.saturating_add(1);
                }
                marg[at] += 1;
            }
            for &a in a_ids {
                let at = li * e + a as usize;
                if marg[at] >= HALVE_AT {
                    for c in &mut counts[at * e..][..e] {
                        *c /= 2;
                    }
                    marg[at] /= 2;
                }
            }
        }
    }

    /// The host experts of layer `layer` the pass is most likely to route,
    /// into `out` (cleared first), highest score first, a tie to the lower id:
    /// at most `cap`, each with a score above zero and `is_host(id)` true.
    /// `out` is the caller's, made for `cap` ids; the call allocates nothing.
    /// Refused by name: a layer past the predictor's.
    pub fn pick(
        &mut self,
        layer: usize,
        is_host: impl Fn(u32) -> bool,
        cap: usize,
        out: &mut Vec<u32>,
    ) -> Result<(), WarmError> {
        out.clear();
        if layer >= self.layers {
            return Err(WarmError::Shape {
                what: "Predictor::pick",
                detail: format!("layer {layer} of {}", self.layers),
            });
        }
        let Predictor {
            experts,
            rows,
            top_k,
            layers,
            counts,
            marg,
            stamp,
            epoch,
            stores,
            cur,
            pass,
            prev_kept,
            prev_valid,
            score,
            cand,
            ..
        } = self;
        let (e, top_k) = (*experts, *top_k);
        score.fill(0);
        if layer >= 1 {
            let li = layer - 1;
            let open = &stores[*cur];
            for row in 0..*rows {
                let Some(a_ids) = open.whole(row * *layers + li, top_k, *pass) else {
                    continue;
                };
                for &a in a_ids {
                    let at = li * e + a as usize;
                    if stamp[at] != *epoch || marg[at] == 0 {
                        continue;
                    }
                    let w = SCALE / u32::from(marg[at]);
                    for (s, &c) in score.iter_mut().zip(&counts[at * e..][..e]) {
                        *s += w * u32::from(c);
                    }
                }
            }
        }
        if *prev_valid {
            let prev = &stores[*cur ^ 1];
            for row in 0..*rows {
                if !prev_kept.keeps(row) {
                    continue;
                }
                if let Some(b_ids) = prev.whole(row * *layers + layer, top_k, *pass - 1) {
                    for &b in b_ids {
                        score[b as usize] += SCALE;
                    }
                }
            }
        }
        cand.clear();
        if cap == 0 {
            return Ok(());
        }
        for (b, &s) in (0u32..).zip(score.iter()) {
            if s > 0 && is_host(b) {
                cand.push((s, b));
            }
        }
        let order = |x: &(u32, u32), y: &(u32, u32)| y.0.cmp(&x.0).then(x.1.cmp(&y.1));
        if cand.len() > cap {
            cand.select_nth_unstable_by(cap - 1, order);
            cand.truncate(cap);
        }
        cand.sort_unstable_by(order);
        out.extend(cand.iter().map(|&(_, b)| b));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{HALVE_AT, Predictor, WarmError, table_bytes};
    use crate::swaprule::KeptRows;
    use std::collections::BTreeMap;

    /// `lcm(1..=16)`, written out: the unit of a score.
    const U: u64 = 720_720;

    /// What a pass routed: `[row][layer]` → the row's ids at the layer.
    type Routing = Vec<Vec<Vec<u32>>>;

    /// A routing from its rows, each a list of layers' ids.
    fn routing<const L: usize>(rows: &[[&[u32]; L]]) -> Routing {
        rows.iter()
            .map(|row| row.iter().map(|ids| ids.to_vec()).collect())
            .collect()
    }

    /// One pass over the predictor as the step port drives it: each layer
    /// folds, picks, then its rows' ids are noted; the pass ends keeping
    /// `kept`. The picks, a list a layer.
    fn pass(
        p: &mut Predictor,
        routing: &Routing,
        kept: KeptRows,
        host: impl Fn(u32) -> bool + Copy,
        cap: usize,
    ) -> Vec<Vec<u32>> {
        let mut picks = Vec::new();
        let mut out = Vec::with_capacity(cap);
        for l in 0..p.layers() {
            p.fold(l);
            p.pick(l, host, cap, &mut out).unwrap();
            picks.push(out.clone());
            for (row, r) in routing.iter().enumerate() {
                for (k, &id) in r[l].iter().enumerate() {
                    p.note(l, row, k, id).unwrap();
                }
            }
        }
        p.pass_end(kept);
        picks
    }

    /// The rule as the replay states it (the table fed the kept rows of a pass
    /// when it ends, every layer pair of a row at once, then the next pass's
    /// score over its layer `l-1` ids plus the pass before's kept layer-`l`
    /// ids), written with maps and no epoch, stores or lazy fold: what
    /// [`Predictor`] must equal.
    #[derive(Default)]
    struct Reference {
        c: BTreeMap<(usize, u32), BTreeMap<u32, u64>>,
        m: BTreeMap<(usize, u32), u64>,
        last: Option<(Routing, KeptRows)>,
        halvings: usize,
    }

    impl Reference {
        fn end_pass(&mut self, routing: &Routing, kept: KeptRows) {
            for (row, r) in routing.iter().enumerate() {
                if !kept.keeps(row) {
                    continue;
                }
                for l in 1..r.len() {
                    let (a_, b_) = (&r[l - 1], &r[l]);
                    for &a in a_ {
                        *self.m.entry((l - 1, a)).or_default() += 1;
                    }
                    for &a in a_ {
                        for &b in b_ {
                            *self.c.entry((l - 1, a)).or_default().entry(b).or_default() += 1;
                        }
                    }
                    for &a in a_ {
                        if self.m[&(l - 1, a)] >= u64::from(HALVE_AT) {
                            let half = self.c[&(l - 1, a)]
                                .iter()
                                .map(|(&b, &n)| (b, n / 2))
                                .collect();
                            self.c.insert((l - 1, a), half);
                            *self.m.get_mut(&(l - 1, a)).unwrap() /= 2;
                            self.halvings += 1;
                        }
                    }
                }
            }
            self.last = Some((routing.clone(), kept));
        }

        fn prompt(&mut self) {
            *self = Reference {
                halvings: self.halvings,
                ..Reference::default()
            };
        }

        fn pick(
            &self,
            l: usize,
            open: &[Vec<u32>],
            host: impl Fn(u32) -> bool,
            cap: usize,
        ) -> Vec<u32> {
            let mut score: BTreeMap<u32, u64> = BTreeMap::new();
            if l >= 1 {
                for ids in open {
                    for &a in ids {
                        let m = self.m.get(&(l - 1, a)).copied().unwrap_or(0);
                        if m == 0 {
                            continue;
                        }
                        for (&b, &n) in self.c.get(&(l - 1, a)).into_iter().flatten() {
                            *score.entry(b).or_default() += n * (U / m);
                        }
                    }
                }
            }
            if let Some((r, kept)) = &self.last {
                for (row, rr) in r.iter().enumerate() {
                    if kept.keeps(row) {
                        for &b in &rr[l] {
                            *score.entry(b).or_default() += U;
                        }
                    }
                }
            }
            let mut v: Vec<(u64, u32)> = score
                .into_iter()
                .filter(|&(b, s)| s > 0 && host(b))
                .map(|(b, s)| (s, b))
                .collect();
            v.sort_by(|x, y| y.0.cmp(&x.0).then(x.1.cmp(&y.1)));
            v.truncate(cap);
            v.into_iter().map(|(_, b)| b).collect()
        }
    }

    /// Three layers, six services (two passes of three layers) and the passes
    /// after, hand-derived from the rule: the picks of every service, with
    /// the ties (lower id first), the host filter (expert 7 is never listed),
    /// the cap (3) and the kept rows (pass 1's row 1 routes experts no pick
    /// ever lists; pass 3 keeps nothing, so its pairs never fold) each in it.
    #[test]
    fn three_layers_six_services_golden() {
        let mut p = Predictor::new(3, 8, 2, 2).unwrap();
        let host = |id: u32| id != 7;
        // Pass 1: nothing behind it, nothing listed. Row 1 is not kept.
        let p1 = routing(&[[&[0, 1], &[2, 3], &[4, 5]], [&[0, 2], &[3, 4], &[6, 7]]]);
        assert_eq!(
            pass(&mut p, &p1, KeptRows::of([0]), host, 3),
            [vec![], vec![], vec![]]
        );
        // Pass 2: layer 0 has only the kept row's layer-0 ids to go by (0 and
        // 1, a tie, the lower first); layer 1 adds the table's pair (0 and 1 →
        // 2 and 3, once from each open id) to the kept layer-1 ids; layer 2
        // the same one layer on.
        let p2 = routing(&[[&[0, 1], &[2, 6], &[4, 7]], [&[2, 3], &[3, 5], &[1, 2]]]);
        assert_eq!(
            pass(&mut p, &p2, KeptRows::of([0, 1]), host, 3),
            [vec![0, 1], vec![2, 3], vec![4, 5]]
        );
        // Pass 3 keeps no row. Layer 1: 3 scores 4U, 2 and 5 3U each (a tie),
        // 6 2U (cut by the cap); layer 2: 4 4U, 7 would be 2.5U but is not a
        // host, 1 and 2 2U each (a tie), 5 1.5U (cut).
        let p3 = routing(&[[&[0, 2], &[3, 6], &[4, 7]], [&[1, 2], &[2, 3], &[5, 6]]]);
        assert_eq!(
            pass(&mut p, &p3, KeptRows::prefix(0), host, 3),
            [vec![0, 1, 2], vec![3, 2, 5], vec![4, 1, 2]]
        );
        // Pass 4: the pass before kept nothing, so no kept layer-0 id and no
        // pair of pass 3 folded; the table alone lists layers 1 and 2.
        let p4 = routing(&[[&[0, 1], &[2, 3], &[4, 5]]]);
        assert_eq!(
            pass(&mut p, &p4, KeptRows::prefix(1), host, 3),
            [vec![], vec![2, 3, 6], vec![4, 5, 1]]
        );
        // The table's row of layer 0's id 0: two kept pairs (pass 1's row 0
        // and pass 2's row 0), none of pass 1's row 1.
        let (counts, marg) = p.row(1, 0).unwrap();
        assert_eq!(marg, 2);
        assert_eq!((counts[2], counts[3], counts[4], counts[6]), (2, 1, 0, 1));
    }

    /// A row of the table halves when its marginal reaches [`HALVE_AT`], the
    /// counts and the marginal both, floor: one id's successor 8 passes, then
    /// another's 16 (a pass's fold is of the pass before, so the marginal is
    /// the pass index until it halves).
    #[test]
    fn a_row_halves_at_sixteen() {
        let mut p = Predictor::new(2, 8, 1, 1).unwrap();
        let at = |p: &Predictor| {
            let (c, m) = p.row(1, 0).unwrap();
            (c[1], c[2], m)
        };
        for n in 0..=24usize {
            let b = if n < 8 { 1 } else { 2 };
            pass(
                &mut p,
                &routing(&[[&[0], &[b]]]),
                KeptRows::prefix(1),
                |_| true,
                3,
            );
            match n {
                15 => assert_eq!(at(&p), (8, 7, 15), "one short of the halving"),
                16 => assert_eq!(at(&p), (4, 4, 8), "the 16th fold halves"),
                23 => assert_eq!(at(&p), (4, 11, 15)),
                24 => assert_eq!(at(&p), (2, 6, 8), "the second halving"),
                _ => {}
            }
        }
    }

    /// A tie in score goes to the lower id, whatever order the ids were
    /// routed in, at the cap's cut too: experts 2, 5 and 6 score 4U each.
    #[test]
    fn ties_go_to_the_lower_id() {
        let go = |cap: usize| {
            let mut p = Predictor::new(2, 8, 1, 3).unwrap();
            let both = routing(&[[&[0, 1, 3], &[5, 2, 6]]]);
            pass(&mut p, &both, KeptRows::prefix(1), |_| true, cap);
            pass(&mut p, &both, KeptRows::prefix(1), |_| true, cap)
        };
        assert_eq!(go(3)[1], [2, 5, 6]);
        assert_eq!(go(2)[1], [2, 5], "the cap cuts the highest of a tie");
        assert_eq!(go(1)[1], [2]);
        assert!(go(0)[1].is_empty());
    }

    /// Only the rows a pass kept fold: a row that was not kept leaves no row
    /// in the table.
    #[test]
    fn only_the_kept_rows_fold() {
        let mut p = Predictor::new(2, 8, 2, 1).unwrap();
        let two = routing(&[[&[0], &[1]], [&[2], &[3]]]);
        pass(&mut p, &two, KeptRows::of([1]), |_| true, 3);
        pass(&mut p, &two, KeptRows::of([1]), |_| true, 3);
        assert!(p.row(1, 0).is_none(), "row 0 was not kept");
        let (counts, marg) = p.row(1, 2).unwrap();
        assert_eq!((counts[3], marg), (1, 1), "row 1's pair, folded once");
        assert_eq!(counts.iter().map(|&c| u32::from(c)).sum::<u32>(), 1);
    }

    /// A prompt call starts a request: every row of the table is empty at once
    /// and what the pass before kept is no history; the next request's folds
    /// start from zero, not from what the table held.
    #[test]
    fn a_prompt_resets_the_table() {
        let mut p = Predictor::new(2, 8, 1, 1).unwrap();
        let go = routing(&[[&[0], &[1]]]);
        for _ in 0..3 {
            pass(&mut p, &go, KeptRows::prefix(1), |_| true, 3);
        }
        assert_eq!(p.row(1, 0).map(|(_, m)| m), Some(2));
        p.reset();
        assert!(p.row(1, 0).is_none(), "every row is empty after the reset");
        let picks = pass(&mut p, &go, KeptRows::prefix(1), |_| true, 3);
        assert_eq!(
            picks,
            [vec![], vec![]],
            "no pair and no kept pass behind it"
        );
        pass(&mut p, &go, KeptRows::prefix(1), |_| true, 3);
        let (counts, marg) = p.row(1, 0).unwrap();
        assert_eq!(
            (counts[1], marg),
            (1, 1),
            "the new request's first fold starts from an empty row"
        );
    }

    /// The predictor equals the rule over a long seeded run of passes of one to
    /// three rows, random kept rows, prompts that reset, and a host filter and
    /// a cap that cut: every service's pick, 600 passes, rows halving
    /// throughout.
    #[test]
    fn it_equals_the_rule_over_a_long_run() {
        let (layers, experts, top_k) = (4usize, 12u32, 3usize);
        let host = |id: u32| id % 4 != 3;
        let mut p = Predictor::new(layers, experts as usize, 3, top_k).unwrap();
        let mut reference = Reference::default();
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move |n: u64| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) % n
        };
        let mut picked = 0usize;
        for n in 0..600 {
            if n % 97 == 96 {
                p.reset();
                reference.prompt();
                continue;
            }
            let rows = 1 + next(3) as usize;
            let route: Routing = (0..rows)
                .map(|_| {
                    (0..layers)
                        .map(|_| {
                            let mut ids = Vec::new();
                            while ids.len() < top_k {
                                let id = next(u64::from(experts)) as u32;
                                if !ids.contains(&id) {
                                    ids.push(id);
                                }
                            }
                            ids
                        })
                        .collect()
                })
                .collect();
            let kept = KeptRows::of((0..rows).filter(|_| next(3) != 0));
            let got = pass(&mut p, &route, kept, host, 5);
            for (l, got) in got.iter().enumerate() {
                let open: Vec<Vec<u32>> = route
                    .iter()
                    .map(|row| row[l.saturating_sub(1)].clone())
                    .collect();
                let want = reference.pick(l, &open, host, 5);
                assert_eq!(*got, want, "pass {n} layer {l}");
                picked += want.len();
            }
            reference.end_pass(&route, kept);
        }
        assert!(
            reference.halvings > 50 && picked > 1000,
            "the run exercised halving ({}) and picks ({picked})",
            reference.halvings
        );
    }

    /// A pick fills the caller's buffer in place: given room for `cap` ids it
    /// never moves or grows it, whatever the history, so a service that
    /// reuses the buffer allocates nothing for it (`gate_hybrid`'s warm arm
    /// counts the allocator calls of a whole service).
    #[test]
    fn a_pick_stays_in_the_callers_buffer() {
        let (layers, experts, cap) = (4usize, 12u32, 5usize);
        let mut p = Predictor::new(layers, experts as usize, 1, 3).unwrap();
        let mut out: Vec<u32> = Vec::with_capacity(cap);
        let at = out.as_ptr();
        let mut picked = 0usize;
        for n in 0..40u32 {
            for l in 0..layers {
                p.fold(l);
                p.pick(l, |_| true, cap, &mut out).unwrap();
                assert_eq!(
                    (out.as_ptr(), out.capacity()),
                    (at, cap),
                    "pass {n} layer {l}"
                );
                assert!(out.len() <= cap);
                picked += out.len();
                for k in 0..3u32 {
                    p.note(l, 0, k as usize, (n + 4 * l as u32 + k) % experts)
                        .unwrap();
                }
            }
            p.pass_end(KeptRows::prefix(1));
        }
        assert!(picked > 0, "the run picked nothing");
    }

    /// A routed id past the shape, a slot out of order, a layer past the
    /// predictor's and a shape it is not made for are refused by name; a cell
    /// that is not whole is no history.
    #[test]
    fn undefined_input_is_a_named_error() {
        let mut p = Predictor::new(2, 4, 2, 2).unwrap();
        let note = |p: &mut Predictor, l, row, k, id| p.note(l, row, k, id);
        assert!(matches!(
            note(&mut p, 2, 0, 0, 0),
            Err(WarmError::Note { .. })
        ));
        assert!(matches!(
            note(&mut p, 0, 2, 0, 0),
            Err(WarmError::Note { .. })
        ));
        assert!(matches!(
            note(&mut p, 0, 0, 2, 0),
            Err(WarmError::Note { .. })
        ));
        assert!(matches!(
            note(&mut p, 0, 0, 0, 4),
            Err(WarmError::Note { .. })
        ));
        let skipped = note(&mut p, 0, 0, 1, 1).unwrap_err().to_string();
        assert!(skipped.contains("out of order"), "{skipped}");
        let mut out = Vec::new();
        let past = p.pick(2, |_| true, 3, &mut out).unwrap_err().to_string();
        assert!(
            past.contains("Predictor::pick") && past.contains("layer 2 of 2"),
            "{past}"
        );
        for shape in [
            (0, 4, 2, 2),
            (2, 0, 2, 2),
            (2, 4, 0, 2),
            (2, 4, 2, 0),
            (2, 4, 65, 2),
        ] {
            let (l, e, r, k) = shape;
            assert!(Predictor::new(l, e, r, k).is_err(), "{shape:?}");
        }
        assert!(
            Predictor::new(2, 4, 64, 65).is_err(),
            "a pass past MAX_CELLS ids"
        );
        // A cell with one of its two slots noted is no history.
        let mut p = Predictor::new(2, 4, 1, 2).unwrap();
        p.note(0, 0, 0, 1).unwrap();
        p.note(0, 0, 1, 2).unwrap();
        p.note(1, 0, 0, 3).unwrap();
        p.pass_end(KeptRows::prefix(1));
        p.fold(1);
        assert!(p.row(1, 1).is_none(), "layer 1's cell was not whole");
    }

    /// The table's size is the formula's, and the formula is the allocation's.
    #[test]
    fn table_bytes_is_the_tables_allocation() {
        let p = Predictor::new(5, 7, 1, 2).unwrap();
        assert_eq!(
            table_bytes(5, 7),
            p.counts.len() + p.marg.len() + 4 * p.stamp.len()
        );
        assert_eq!(table_bytes(48, 512), 47 * 512 * (512 + 1 + 4));
        assert_eq!(table_bytes(1, 512), 0, "one layer has no pair");
    }
}
