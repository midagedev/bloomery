//! The adaptive expert residency rule: which routed experts each layer keeps
//! on the card, as a pure function of the routing history.
//!
//! The rule counts every kept row's routed ids per layer. Every
//! [`SwapParams::every`] passes it pairs each layer's most-used experts off the
//! card with its least-used ones on it and admits a pair when the admitted
//! count clears [`SwapParams::min_count`] and the victim's count by
//! [`SwapParams::margin`]; at most [`SwapParams::cap`] pairs a planning pass
//! over all layers, largest gain first, then every count decays.
//!
//! A pair is a [`Flip`]: admit-then-flip into one of the layer's
//! [`SwapParams::spares`] spare slots. The victim stays on the card until the
//! flip goes live, `delay` passes after the boundary that made it, and both
//! change at that boundary; a layer never has more flips in flight than it has
//! spares. [`SwapRule::open`] is the other way a map moves: an in-place
//! relayout at a quiet boundary from a whole prompt's counts. A layer's first
//! pinned seed experts ([`SwapRule::new_pinned`]) stay on the card through
//! both: they are never a victim. A layer's away experts
//! ([`SwapRule::new_placed`]) run on another device for the rule's life: never
//! admitted, never a victim, never in its card set.
//!
//! One pass is driven as `observe`* → [`SwapRule::end_pass`] →
//! [`SwapRule::plan`] at the boundary it ends at; anything else is refused by
//! name. Flips are made in (gain descending, layer ascending, rank ascending)
//! order, and each layer ranks its candidates and victims by a total order
//! (see [`SwapRule::plan`] and [`SwapRule::open`]), so the same history gives
//! the same flips in the same order.

use std::cmp::{Ordering, Reverse};
use std::fmt;

/// The rule's constants.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwapParams {
    /// A planning pass at every boundary that is a multiple of this, 1 or more.
    pub every: u64,
    /// The most flips one planning pass makes over all layers, 1 or more.
    pub cap: usize,
    /// How far an admitted count must clear its victim's; finite, 0 or more.
    pub margin: f64,
    /// The least count an admitted expert has; finite, 0 or more.
    pub min_count: f64,
    /// What every count is multiplied by after a planning pass, 0 to 1.
    pub decay: f64,
    /// Spare slots per layer: the most flips a layer has in flight, 1 or more.
    pub spares: usize,
    /// Passes from the boundary that makes a flip to the one it goes live at.
    pub delay: u64,
}

impl SwapParams {
    /// The parameters the replay chose (every 4, cap 24, margin 3, min count
    /// 2, decay 0.9, one spare) at a model's live delay.
    #[must_use]
    pub const fn mid(delay: u64) -> Self {
        SwapParams {
            every: 4,
            cap: 24,
            margin: 3.0,
            min_count: 2.0,
            decay: 0.9,
            spares: 1,
            delay,
        }
    }
}

/// The routing a rule observes: experts per layer, ids a row routes, and the
/// most rows one pass runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub experts: usize,
    pub top_k: usize,
    pub max_rows: usize,
}

/// One expert admitted to a layer's card set in place of another, live from
/// boundary `live_at` on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Flip {
    pub layer: usize,
    pub admit: u32,
    pub evict: u32,
    pub live_at: u64,
}

/// A call the rule refuses, by name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SwapRuleError {
    /// A parameter outside its range.
    Param {
        name: &'static str,
        range: &'static str,
    },
    /// A shape outside its range, or a seed and a capacity list of another
    /// length.
    Shape {
        what: &'static str,
        value: usize,
    },
    /// A layer's seed list naming fewer experts than its card slots: the rule
    /// swaps and never fills an empty slot.
    SeedUnderCapacity {
        layer: usize,
        seed: usize,
        capacity: usize,
    },
    /// A layer pinning more of its seed list than its card slots.
    PinnedOverCapacity {
        layer: usize,
        pinned: usize,
        capacity: usize,
    },
    /// A layer's seed list naming an expert twice.
    SeedDuplicate {
        layer: usize,
        id: u32,
    },
    /// An away expert its layer's seed list also ranks.
    AwayInSeed {
        layer: usize,
        id: u32,
    },
    /// A layer's away list naming an expert twice.
    AwayDuplicate {
        layer: usize,
        id: u32,
    },
    LayerOutOfRange {
        layer: usize,
        layers: usize,
    },
    ExpertOutOfRange {
        layer: usize,
        id: u32,
        experts: usize,
    },
    RowOutOfRange {
        row: usize,
        max_rows: usize,
    },
    /// A row with another count of ids than the shape's `top_k`.
    RowLength {
        layer: usize,
        row: usize,
        len: usize,
        top_k: usize,
    },
    /// A row routing one expert twice.
    DuplicateId {
        layer: usize,
        row: usize,
        id: u32,
    },
    /// A (layer, row) observed twice in one pass.
    RowObservedTwice {
        layer: usize,
        row: usize,
    },
    /// An end of pass keeping more rows than the pass observed.
    KeptPastRows {
        kept: usize,
        rows: usize,
    },
    /// A kept row a layer never observed.
    RowMissing {
        layer: usize,
        row: usize,
    },
    /// An end of pass while the boundary the last pass ended at is unplanned.
    PlanSkipped {
        boundary: u64,
    },
    /// A plan with no pass ended since the last one.
    PassNotEnded {
        boundary: u64,
    },
    /// A plan at another boundary than the one the last pass ended at.
    Boundary {
        got: u64,
        expected: u64,
    },
    /// An opening relayout while observed rows wait for their end of pass.
    RowsPending {
        rows: usize,
    },
    /// Opening counts of another length than layers × experts.
    CountsShape {
        len: usize,
        want: usize,
    },
}

impl fmt::Display for SwapRuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            SwapRuleError::Param { name, range } => {
                write!(f, "swap rule parameter {name} must be {range}")
            }
            SwapRuleError::Shape { what, value } => {
                write!(f, "swap rule shape: {what} (got {value})")
            }
            SwapRuleError::SeedUnderCapacity {
                layer,
                seed,
                capacity,
            } => write!(
                f,
                "layer {layer}'s seed list names {seed} experts, fewer than its {capacity} card \
                 slots: the rule never fills an empty slot"
            ),
            SwapRuleError::PinnedOverCapacity {
                layer,
                pinned,
                capacity,
            } => write!(
                f,
                "layer {layer} pins {pinned} seed experts, more than its {capacity} card slots"
            ),
            SwapRuleError::SeedDuplicate { layer, id } => {
                write!(f, "layer {layer}'s seed list names expert {id} twice")
            }
            SwapRuleError::AwayInSeed { layer, id } => write!(
                f,
                "layer {layer}'s expert {id} is away (on another device) and in its seed list"
            ),
            SwapRuleError::AwayDuplicate { layer, id } => {
                write!(f, "layer {layer}'s away list names expert {id} twice")
            }
            SwapRuleError::LayerOutOfRange { layer, layers } => {
                write!(f, "layer {layer} of a rule over {layers} layers")
            }
            SwapRuleError::ExpertOutOfRange { layer, id, experts } => {
                write!(f, "expert {id} at layer {layer} of {experts} experts")
            }
            SwapRuleError::RowOutOfRange { row, max_rows } => {
                write!(f, "row {row} of a pass of at most {max_rows} rows")
            }
            SwapRuleError::RowLength {
                layer,
                row,
                len,
                top_k,
            } => write!(
                f,
                "row {row} at layer {layer} routes {len} ids, not the {top_k} a row routes"
            ),
            SwapRuleError::DuplicateId { layer, row, id } => {
                write!(f, "row {row} at layer {layer} routes expert {id} twice")
            }
            SwapRuleError::RowObservedTwice { layer, row } => {
                write!(f, "row {row} at layer {layer} observed twice in one pass")
            }
            SwapRuleError::KeptPastRows { kept, rows } => {
                write!(f, "a pass keeping {kept} rows of the {rows} it observed")
            }
            SwapRuleError::RowMissing { layer, row } => {
                write!(f, "kept row {row} was never observed at layer {layer}")
            }
            SwapRuleError::PlanSkipped { boundary } => write!(
                f,
                "a pass ended while boundary {boundary} is unplanned: plan it first"
            ),
            SwapRuleError::PassNotEnded { boundary } => write!(
                f,
                "a plan at boundary {boundary} with no pass ended since the last plan"
            ),
            SwapRuleError::Boundary { got, expected } => write!(
                f,
                "a plan at boundary {got}: the last pass ended at boundary {expected}"
            ),
            SwapRuleError::RowsPending { rows } => write!(
                f,
                "an opening relayout while {rows} observed rows wait for their end of pass"
            ),
            SwapRuleError::CountsShape { len, want } => {
                write!(f, "{len} opening counts, not layers × experts = {want}")
            }
        }
    }
}

impl std::error::Error for SwapRuleError {}

/// Where an expert stands with respect to its layer's card set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    Off,
    Live,
    /// Live from the seed on and never a victim.
    Pinned,
    /// On another device for the rule's life: never admitted, never a victim,
    /// not in the card set.
    Away,
    /// Admitted by a flip in flight: not on the card yet.
    Filling,
    /// The victim of a flip in flight: still on the card.
    Leaving,
}

/// One admissible pair of a planning pass.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Pair {
    gain: f64,
    layer: usize,
    rank: usize,
    admit: u32,
    evict: u32,
}

/// The adaptive residency rule over a model's card layers.
///
/// Every buffer the steady path touches is sized at [`SwapRule::new`]:
/// observing, ending a pass and planning allocate nothing. Two rules are equal
/// when everything their next calls read is: scratch and unobserved row cells
/// do not count.
#[derive(Clone, Debug)]
pub struct SwapRule {
    params: SwapParams,
    shape: Shape,
    layers: usize,
    /// `layer * experts + id`: the expert's place in its layer's seed list,
    /// [`UNRANKED`] when the list does not name it.
    rank: Vec<u32>,
    /// Each layer's card slots: the first this many of its seed list.
    capacity: Vec<usize>,
    /// Each layer's pinned experts: the first this many of its seed list,
    /// never a victim.
    pinned: Vec<usize>,
    /// `layer * experts + id`: the expert is away ([`SwapRule::new_placed`]).
    away: Vec<bool>,
    /// `layer * experts + id`.
    slot: Vec<Slot>,
    counts: Vec<f64>,
    in_flight: Vec<usize>,
    pending: Vec<Flip>,
    /// `(row * layers + layer) * top_k`.
    rows: Vec<u32>,
    seen: Vec<bool>,
    /// One past the last row observed in this pass.
    pass_rows: usize,
    /// Passes ended since the seed.
    passes: u64,
    /// The last boundary planned.
    planned: u64,
    cand: Vec<(f64, u32)>,
    vict: Vec<(f64, u32)>,
    pairs: Vec<Pair>,
    flips: Vec<Flip>,
}

impl PartialEq for SwapRule {
    fn eq(&self, o: &Self) -> bool {
        self.params == o.params
            && self.shape == o.shape
            && self.layers == o.layers
            && self.rank == o.rank
            && self.capacity == o.capacity
            && self.pinned == o.pinned
            && self.away == o.away
            && self.slot == o.slot
            && self.counts == o.counts
            && self.in_flight == o.in_flight
            && self.pending == o.pending
            && self.pass_rows == o.pass_rows
            && self.passes == o.passes
            && self.planned == o.planned
            && self.flips == o.flips
            && self.observed().eq(o.observed())
    }
}

/// The seed rank of an expert its layer's seed list does not name: after every
/// named one.
const UNRANKED: u32 = u32::MAX;

/// Whether candidate `a` ranks before `b`: more uses first, then the lower id.
fn cand_before(a: (f64, u32), b: (f64, u32)) -> bool {
    match a.0.total_cmp(&b.0) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => a.1 < b.1,
    }
}

/// Whether victim `a` ranks before `b`: fewer uses first, then the lower id.
fn vict_before(a: (f64, u32), b: (f64, u32)) -> bool {
    match a.0.total_cmp(&b.0) {
        Ordering::Less => true,
        Ordering::Greater => false,
        Ordering::Equal => a.1 < b.1,
    }
}

/// Keep in `best` the first `q` entries by `before`, ranked. Entries arrive in
/// ascending id, so an equal one never displaces an earlier one.
fn keep_best(
    best: &mut Vec<(f64, u32)>,
    q: usize,
    e: (f64, u32),
    before: fn((f64, u32), (f64, u32)) -> bool,
) {
    let at = best
        .iter()
        .position(|&b| before(e, b))
        .unwrap_or(best.len());
    if at < q {
        if best.len() == q {
            best.pop();
        }
        best.insert(at, e);
    }
}

/// The order flips are made in: gain descending, then layer, then rank.
fn pair_order(a: &Pair, b: &Pair) -> Ordering {
    b.gain
        .total_cmp(&a.gain)
        .then(a.layer.cmp(&b.layer))
        .then(a.rank.cmp(&b.rank))
}

fn check_params(p: &SwapParams) -> Result<(), SwapRuleError> {
    let param = |name, range| Err(SwapRuleError::Param { name, range });
    if p.every == 0 {
        return param("every", "1 or more");
    }
    if p.cap == 0 {
        return param("cap", "1 or more");
    }
    if p.spares == 0 {
        return param("spares", "1 or more");
    }
    if !(p.margin.is_finite() && p.margin >= 0.0) {
        return param("margin", "finite, 0 or more");
    }
    if !(p.min_count.is_finite() && p.min_count >= 0.0) {
        return param("min_count", "finite, 0 or more");
    }
    if !(0.0..=1.0).contains(&p.decay) {
        return param("decay", "0 to 1");
    }
    Ok(())
}

impl SwapRule {
    /// A rule at its seed. Layer `l`'s seed list `seed[l]` ranks its experts,
    /// best first; its first `capacity[l]` are the layer's card set, and an
    /// expert the list does not name ranks after every one it does.
    pub fn new(
        params: SwapParams,
        shape: Shape,
        seed: &[&[u32]],
        capacity: &[usize],
    ) -> Result<Self, SwapRuleError> {
        SwapRule::new_pinned(params, shape, seed, capacity, &vec![0; capacity.len()])
    }

    /// [`SwapRule::new`] with layer `l`'s first `pinned[l]` seed experts
    /// pinned: on the card from the seed on, never a victim of a planning pass
    /// or of an opening relayout. A pinned count past the layer's capacity, or
    /// a list of another length, is refused by name.
    pub fn new_pinned(
        params: SwapParams,
        shape: Shape,
        seed: &[&[u32]],
        capacity: &[usize],
        pinned: &[usize],
    ) -> Result<Self, SwapRuleError> {
        let away = vec![&[][..]; seed.len()];
        SwapRule::new_placed(params, shape, seed, capacity, pinned, &away)
    }

    /// [`SwapRule::new_pinned`] with layer `l`'s experts `away[l]` on another
    /// device for the rule's life: counted when routed, never admitted, never
    /// a victim, never in the card set. An away list of another length, an
    /// away expert its seed list ranks, one named twice and one past the
    /// experts are refused by name.
    pub fn new_placed(
        params: SwapParams,
        shape: Shape,
        seed: &[&[u32]],
        capacity: &[usize],
        pinned: &[usize],
        away: &[&[u32]],
    ) -> Result<Self, SwapRuleError> {
        check_params(&params)?;
        let shape_err = |what, value| Err(SwapRuleError::Shape { what, value });
        if shape.experts == 0 || u32::try_from(shape.experts).is_err() {
            return shape_err("experts must be 1 to u32::MAX", shape.experts);
        }
        if shape.top_k == 0 || shape.top_k > shape.experts {
            return shape_err("top_k must be 1 to experts", shape.top_k);
        }
        if shape.max_rows == 0 {
            return shape_err("max_rows must be 1 or more", shape.max_rows);
        }
        let layers = seed.len();
        if layers == 0 {
            return shape_err("a rule needs 1 or more layers", layers);
        }
        if capacity.len() != layers {
            return shape_err(
                "capacity must list one entry per seed layer",
                capacity.len(),
            );
        }
        if pinned.len() != layers {
            return shape_err("pinned must list one entry per seed layer", pinned.len());
        }
        for (layer, (&p, &cap)) in pinned.iter().zip(capacity).enumerate() {
            if p > cap {
                return Err(SwapRuleError::PinnedOverCapacity {
                    layer,
                    pinned: p,
                    capacity: cap,
                });
            }
        }
        let e = shape.experts;
        let mut slot = vec![Slot::Off; layers * e];
        let mut rank = vec![UNRANKED; layers * e];
        for (layer, (&list, &cap)) in seed.iter().zip(capacity).enumerate() {
            if list.len() < cap {
                return Err(SwapRuleError::SeedUnderCapacity {
                    layer,
                    seed: list.len(),
                    capacity: cap,
                });
            }
            for (r, &id) in list.iter().enumerate() {
                let i = Self::index_of(layer, id, e)?;
                if rank[i] != UNRANKED {
                    return Err(SwapRuleError::SeedDuplicate { layer, id });
                }
                rank[i] = u32::try_from(r)
                    .expect("a list naming no expert twice is at most experts <= u32::MAX long");
                if r < pinned[layer] {
                    slot[i] = Slot::Pinned;
                } else if r < cap {
                    slot[i] = Slot::Live;
                }
            }
        }
        if away.len() != layers {
            return shape_err("away must list one entry per seed layer", away.len());
        }
        let mut is_away = vec![false; layers * e];
        for (layer, &list) in away.iter().enumerate() {
            for &id in list {
                let i = Self::index_of(layer, id, e)?;
                if rank[i] != UNRANKED {
                    return Err(SwapRuleError::AwayInSeed { layer, id });
                }
                if is_away[i] {
                    return Err(SwapRuleError::AwayDuplicate { layer, id });
                }
                is_away[i] = true;
                slot[i] = Slot::Away;
            }
        }
        let most = layers * params.spares;
        Ok(SwapRule {
            params,
            shape,
            layers,
            rank,
            capacity: capacity.to_vec(),
            pinned: pinned.to_vec(),
            away: is_away,
            slot,
            counts: vec![0.0; layers * e],
            in_flight: vec![0; layers],
            pending: Vec::with_capacity(most),
            rows: vec![0; shape.max_rows * layers * shape.top_k],
            seen: vec![false; shape.max_rows * layers],
            pass_rows: 0,
            passes: 0,
            planned: 0,
            cand: Vec::with_capacity(params.spares),
            vict: Vec::with_capacity(params.spares),
            pairs: Vec::with_capacity(most),
            flips: Vec::with_capacity(most.min(params.cap)),
        })
    }

    /// This pass's observed cells and their ids.
    fn observed(&self) -> impl Iterator<Item = (usize, &[u32])> {
        let k = self.shape.top_k;
        self.seen
            .iter()
            .enumerate()
            .filter(|(_, s)| **s)
            .map(move |(cell, _)| (cell, &self.rows[cell * k..(cell + 1) * k]))
    }

    fn index_of(layer: usize, id: u32, experts: usize) -> Result<usize, SwapRuleError> {
        let i = usize::try_from(id).map_err(|_| SwapRuleError::ExpertOutOfRange {
            layer,
            id,
            experts,
        })?;
        if i >= experts {
            return Err(SwapRuleError::ExpertOutOfRange { layer, id, experts });
        }
        Ok(layer * experts + i)
    }

    fn check_layer(&self, layer: usize) -> Result<(), SwapRuleError> {
        if layer >= self.layers {
            return Err(SwapRuleError::LayerOutOfRange {
                layer,
                layers: self.layers,
            });
        }
        Ok(())
    }

    /// The layers the rule covers.
    #[must_use]
    pub fn layers(&self) -> usize {
        self.layers
    }

    /// Layer `layer`'s card set at the seed, in seed order: the first
    /// capacity entries of its seed list, what [`SwapRule::reset`] returns
    /// to.
    pub fn seed(&self, layer: usize) -> Result<Vec<u32>, SwapRuleError> {
        self.check_layer(layer)?;
        let e = self.shape.experts;
        let cap = self.capacity[layer];
        let mut ranked: Vec<(u32, u32)> = self.rank[layer * e..(layer + 1) * e]
            .iter()
            .zip(0u32..)
            .filter(|&(&r, _)| (r as usize) < cap)
            .map(|(&r, id)| (r, id))
            .collect();
        ranked.sort_unstable();
        Ok(ranked.into_iter().map(|(_, id)| id).collect())
    }

    /// Layer `layer`'s pinned count: the first this many of its seed are
    /// never a victim.
    pub fn pinned(&self, layer: usize) -> Result<usize, SwapRuleError> {
        self.check_layer(layer)?;
        Ok(self.pinned[layer])
    }

    /// Passes ended since the seed: the boundary the next pass starts at.
    #[must_use]
    pub fn passes(&self) -> u64 {
        self.passes
    }

    /// Whether expert `id` is on layer `layer`'s card for the pass that starts
    /// at the current boundary: live, or the victim of a flip in flight.
    pub fn is_live(&self, layer: usize, id: u32) -> Result<bool, SwapRuleError> {
        self.check_layer(layer)?;
        let i = Self::index_of(layer, id, self.shape.experts)?;
        Ok(matches!(
            self.slot[i],
            Slot::Live | Slot::Pinned | Slot::Leaving
        ))
    }

    /// Layer `layer`'s card set, ascending ids.
    pub fn live(&self, layer: usize) -> Result<impl Iterator<Item = u32> + '_, SwapRuleError> {
        self.check_layer(layer)?;
        let e = self.shape.experts;
        Ok(self.slot[layer * e..(layer + 1) * e]
            .iter()
            .zip(0u32..)
            .filter(|(s, _)| matches!(s, Slot::Live | Slot::Pinned | Slot::Leaving))
            .map(|(_, id)| id))
    }

    /// The flips made and not yet live, in the order they were made.
    #[must_use]
    pub fn in_flight(&self) -> &[Flip] {
        &self.pending
    }

    /// Buffer row `row`'s routed `ids` at layer `layer` for this pass.
    pub fn observe(&mut self, layer: usize, row: usize, ids: &[u32]) -> Result<(), SwapRuleError> {
        self.check_layer(layer)?;
        let Shape {
            experts,
            top_k,
            max_rows,
        } = self.shape;
        if row >= max_rows {
            return Err(SwapRuleError::RowOutOfRange { row, max_rows });
        }
        if ids.len() != top_k {
            return Err(SwapRuleError::RowLength {
                layer,
                row,
                len: ids.len(),
                top_k,
            });
        }
        for (k, &id) in ids.iter().enumerate() {
            Self::index_of(layer, id, experts)?;
            if ids[..k].contains(&id) {
                return Err(SwapRuleError::DuplicateId { layer, row, id });
            }
        }
        let cell = row * self.layers + layer;
        if self.seen[cell] {
            return Err(SwapRuleError::RowObservedTwice { layer, row });
        }
        self.seen[cell] = true;
        self.rows[cell * top_k..(cell + 1) * top_k].copy_from_slice(ids);
        self.pass_rows = self.pass_rows.max(row + 1);
        Ok(())
    }

    /// End the pass: fold its first `kept` rows into the counts and drop the
    /// rest, so a rejected row leaves no trace. Every layer must have observed
    /// every kept row.
    pub fn end_pass(&mut self, kept: usize) -> Result<(), SwapRuleError> {
        if self.planned != self.passes {
            return Err(SwapRuleError::PlanSkipped {
                boundary: self.passes,
            });
        }
        if kept > self.pass_rows {
            return Err(SwapRuleError::KeptPastRows {
                kept,
                rows: self.pass_rows,
            });
        }
        let (l_n, top_k, e) = (self.layers, self.shape.top_k, self.shape.experts);
        for row in 0..kept {
            for layer in 0..l_n {
                if !self.seen[row * l_n + layer] {
                    return Err(SwapRuleError::RowMissing { layer, row });
                }
            }
        }
        for cell in 0..kept * l_n {
            let layer = cell % l_n;
            for &id in &self.rows[cell * top_k..(cell + 1) * top_k] {
                self.counts[layer * e + id as usize] += 1.0;
            }
        }
        self.seen[..self.pass_rows * l_n].fill(false);
        self.pass_rows = 0;
        self.passes += 1;
        Ok(())
    }

    /// The boundary `boundary` the last pass ended at: the flips going live
    /// there land, and at a multiple of [`SwapParams::every`] a planning pass
    /// makes new flips, live at `boundary + delay`, then every count decays.
    /// Returns the flips made here, in the order they were made.
    ///
    /// Ties: candidates by (count descending, id ascending), victims by (count ascending, id ascending).
    pub fn plan(&mut self, boundary: u64) -> Result<&[Flip], SwapRuleError> {
        let expected = self.planned + 1;
        if self.passes < expected {
            return Err(SwapRuleError::PassNotEnded { boundary });
        }
        if boundary != expected {
            return Err(SwapRuleError::Boundary {
                got: boundary,
                expected,
            });
        }
        self.land(boundary);
        self.flips.clear();
        if boundary.is_multiple_of(self.params.every) {
            self.make_pairs();
            self.issue(boundary);
            let decay = self.params.decay;
            self.counts.iter_mut().for_each(|c| *c *= decay);
            if self.params.delay == 0 {
                self.land(boundary);
            }
        }
        self.planned = boundary;
        Ok(&self.flips)
    }

    /// Apply the flips live at or before `boundary`.
    fn land(&mut self, boundary: u64) {
        let SwapRule {
            pending,
            slot,
            in_flight,
            shape,
            ..
        } = self;
        let e = shape.experts;
        pending.retain(|f| {
            if f.live_at > boundary {
                return true;
            }
            slot[f.layer * e + f.admit as usize] = Slot::Live;
            slot[f.layer * e + f.evict as usize] = Slot::Off;
            in_flight[f.layer] -= 1;
            false
        });
    }

    /// Each layer's admissible pairs, as many as its free spares take, into
    /// `pairs` in the order flips are made.
    fn make_pairs(&mut self) {
        let SwapParams {
            margin,
            min_count,
            spares,
            ..
        } = self.params;
        let e = self.shape.experts;
        self.pairs.clear();
        for layer in 0..self.layers {
            let q = spares - self.in_flight[layer];
            if q == 0 {
                continue;
            }
            self.cand.clear();
            self.vict.clear();
            let slots = &self.slot[layer * e..(layer + 1) * e];
            let counts = &self.counts[layer * e..(layer + 1) * e];
            for ((&s, &c), id) in slots.iter().zip(counts).zip(0u32..) {
                match s {
                    Slot::Off => keep_best(&mut self.cand, q, (c, id), cand_before),
                    Slot::Live => keep_best(&mut self.vict, q, (c, id), vict_before),
                    Slot::Pinned | Slot::Away | Slot::Filling | Slot::Leaving => {}
                }
            }
            for (rank, (&(c, admit), &(v, evict))) in self.cand.iter().zip(&self.vict).enumerate() {
                if !(c >= min_count && c >= v + margin) {
                    break;
                }
                self.pairs.push(Pair {
                    gain: c - v,
                    layer,
                    rank,
                    admit,
                    evict,
                });
            }
        }
        self.pairs.sort_unstable_by(pair_order);
    }

    /// Make the first [`SwapParams::cap`] pairs flips live at `boundary + delay`.
    fn issue(&mut self, boundary: u64) {
        let live_at = boundary
            .checked_add(self.params.delay)
            .expect("a boundary plus the live delay stays under 2^64 passes");
        let e = self.shape.experts;
        for p in self.pairs.iter().take(self.params.cap) {
            let f = Flip {
                layer: p.layer,
                admit: p.admit,
                evict: p.evict,
                live_at,
            };
            self.slot[p.layer * e + p.admit as usize] = Slot::Filling;
            self.slot[p.layer * e + p.evict as usize] = Slot::Leaving;
            self.in_flight[p.layer] += 1;
            self.pending.push(f);
            self.flips.push(f);
        }
    }

    /// The opening relayout at a quiet boundary, from the undecayed routing
    /// `counts` of a whole prompt (`layer * experts + id`): per layer, the
    /// most-used experts off the card against the least-used on it, kept
    /// while the admitted count is higher, at most `m` over all layers in the
    /// order flips are made. The flips are live at once (in place, no spare);
    /// flips in flight keep their experts out of it. The rule's own counts do
    /// not move.
    ///
    /// Ties: candidates by (count descending, seed rank ascending, id ascending), victims by
    /// (count ascending, seed rank descending, id ascending): the better-ranked seed expert
    /// enters or stays first.
    pub fn open(&mut self, counts: &[u32], m: usize) -> Result<Vec<Flip>, SwapRuleError> {
        if self.pass_rows > 0 {
            return Err(SwapRuleError::RowsPending {
                rows: self.pass_rows,
            });
        }
        let e = self.shape.experts;
        let want = self.layers * e;
        if counts.len() != want {
            return Err(SwapRuleError::CountsShape {
                len: counts.len(),
                want,
            });
        }
        let mut pairs = Vec::new();
        let mut cand = Vec::with_capacity(e);
        let mut vict = Vec::with_capacity(e);
        for layer in 0..self.layers {
            cand.clear();
            vict.clear();
            let slots = &self.slot[layer * e..(layer + 1) * e];
            let ranks = &self.rank[layer * e..(layer + 1) * e];
            let layer_counts = &counts[layer * e..(layer + 1) * e];
            for (((&s, &c), &r), id) in slots.iter().zip(layer_counts).zip(ranks).zip(0u32..) {
                match s {
                    Slot::Off => cand.push((c, r, id)),
                    Slot::Live => vict.push((c, r, id)),
                    Slot::Pinned | Slot::Away | Slot::Filling | Slot::Leaving => {}
                }
            }
            cand.sort_unstable_by_key(|&(c, r, id)| (Reverse(c), r, id));
            vict.sort_unstable_by_key(|&(c, r, id)| (c, Reverse(r), id));
            for (rank, (&(c, _, admit), &(v, _, evict))) in cand.iter().zip(&vict).enumerate() {
                if c <= v {
                    break;
                }
                pairs.push(Pair {
                    gain: f64::from(c - v),
                    layer,
                    rank,
                    admit,
                    evict,
                });
            }
        }
        pairs.sort_by(pair_order);
        pairs.truncate(m);
        let flips = pairs
            .iter()
            .map(|p| Flip {
                layer: p.layer,
                admit: p.admit,
                evict: p.evict,
                live_at: self.passes,
            })
            .collect::<Vec<_>>();
        for f in &flips {
            self.slot[f.layer * e + f.admit as usize] = Slot::Live;
            self.slot[f.layer * e + f.evict as usize] = Slot::Off;
        }
        Ok(flips)
    }

    /// Back to the seed: every layer's card set, its away experts away, no
    /// count, no flip in flight, no observed row, boundary 0.
    pub fn reset(&mut self) {
        let e = self.shape.experts;
        self.slot.fill(Slot::Off);
        for layer in 0..self.layers {
            let (cap, pin) = (self.capacity[layer], self.pinned[layer]);
            for i in layer * e..(layer + 1) * e {
                let r = self.rank[i] as usize;
                if self.away[i] {
                    self.slot[i] = Slot::Away;
                } else if r < pin {
                    self.slot[i] = Slot::Pinned;
                } else if r < cap {
                    self.slot[i] = Slot::Live;
                }
            }
        }
        self.counts.fill(0.0);
        self.in_flight.fill(0);
        self.pending.clear();
        self.seen.fill(false);
        self.pass_rows = 0;
        self.passes = 0;
        self.planned = 0;
        self.cand.clear();
        self.vict.clear();
        self.pairs.clear();
        self.flips.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::{Flip, Shape, SwapParams, SwapRule, SwapRuleError};

    // The residency rule's hand-computed cases, its protocol, and its refusals.
    //
    // The first seven traces are the adaptive policy's hand cases of the replay's
    // self-test, each at the parameters its call ran with (the policy's defaults
    // where the call named none: margin 1.5, min count 2, cap 96); every layer
    // there holds one card slot, so one spare stands in for no per-layer limit,
    // and the replay's link delay is the live delay `d`.
    fn params(every: u64, cap: usize, margin: f64, min_count: f64, decay: f64) -> SwapParams {
        SwapParams {
            every,
            cap,
            margin,
            min_count,
            decay,
            spares: 1,
            delay: 0,
        }
    }

    fn rule(p: SwapParams, experts: usize, top_k: usize, seed: &[&[u32]]) -> SwapRule {
        let cap: Vec<usize> = seed.iter().map(|s| s.len()).collect();
        let shape = Shape {
            experts,
            top_k,
            max_rows: 4,
        };
        SwapRule::new(p, shape, seed, &cap).expect("a valid rule")
    }

    /// One row a pass: hits per layer counted on the map each pass starts with,
    /// then observe, end the pass, plan its boundary. Returns hits per layer and
    /// every flip made.
    fn replay(r: &mut SwapRule, x: &[Vec<Vec<u32>>]) -> (Vec<usize>, Vec<Flip>) {
        let mut hits = vec![0; r.layers()];
        let mut flips = Vec::new();
        for (s, row) in x.iter().enumerate() {
            for (layer, ids) in row.iter().enumerate() {
                hits[layer] += ids
                    .iter()
                    .filter(|&&id| r.is_live(layer, id).unwrap())
                    .count();
                r.observe(layer, 0, ids).unwrap();
            }
            r.end_pass(1).unwrap();
            flips.extend_from_slice(r.plan(s as u64 + 1).unwrap());
        }
        (hits, flips)
    }

    /// Steps of one-layer rows, each row the ids given.
    fn steps(rows: &[&[u32]]) -> Vec<Vec<Vec<u32>>> {
        rows.iter().map(|r| vec![r.to_vec()]).collect()
    }

    /// `n` one-layer steps routing `id` alone.
    fn repeat(id: u32, n: usize) -> Vec<Vec<Vec<u32>>> {
        vec![vec![vec![id]]; n]
    }

    /// Steps of two-layer rows of one id each.
    fn steps2(rows: &[(u32, u32)]) -> Vec<Vec<Vec<u32>>> {
        rows.iter().map(|&(a, b)| vec![vec![a], vec![b]]).collect()
    }

    fn flip(layer: usize, admit: u32, evict: u32, live_at: u64) -> Flip {
        Flip {
            layer,
            admit,
            evict,
            live_at,
        }
    }

    /// Case 1: expert 2 used every step clears min count 2 and margin 1.5 over the
    /// unused resident at the first pass (boundary 2) and serves steps 2..6.
    #[test]
    fn a_hot_expert_is_admitted_at_the_first_pass() {
        let mut r = rule(params(2, 96, 1.5, 2.0, 1.0), 4, 1, &[&[0]]);
        let (hits, flips) = replay(&mut r, &repeat(2, 6));
        assert_eq!(hits, [4]);
        assert_eq!(flips, [flip(0, 2, 0, 2)]);
    }

    /// Case 2: two experts used once a pass each, with every count dropped at the
    /// pass, never reach min count 2.
    #[test]
    fn decay_keeps_alternating_experts_under_min_count() {
        let mut r = rule(params(2, 96, 1.5, 2.0, 0.0), 4, 1, &[&[0]]);
        let (hits, flips) = replay(&mut r, &steps(&[&[2], &[3], &[2], &[3], &[2], &[3]]));
        assert_eq!(hits, [0]);
        assert_eq!(flips, []);
    }

    /// Case 3: the replay's link (a copy of two and a half steps, issued after the
    /// second step) serves from step 5: made at boundary 2, live at 5, so d = 3;
    /// the victim stays on the card until then and the admitted expert is not on
    /// it before.
    #[test]
    fn a_flip_goes_live_after_the_delay() {
        let mut p = params(2, 96, 1.5, 2.0, 1.0);
        p.delay = 3;
        let mut r = rule(p, 4, 1, &[&[0]]);
        let (hits, flips) = replay(&mut r, &repeat(2, 8));
        assert_eq!(hits, [3], "steps 5, 6, 7");
        assert_eq!(flips, [flip(0, 2, 0, 5)]);

        let mut r = rule(p, 4, 1, &[&[0]]);
        let (_, flips) = replay(&mut r, &repeat(2, 4));
        assert_eq!(flips, [flip(0, 2, 0, 5)]);
        assert_eq!(r.in_flight(), [flip(0, 2, 0, 5)]);
        assert!(r.is_live(0, 0).unwrap() && !r.is_live(0, 2).unwrap());
    }

    /// Cases 4 and 5: at the pass after 4 steps expert 2 has 3 uses and the
    /// resident 0 one; 3 >= 1 + 1.5 admits it (inclusive), 3 >= 1 + 3 does not.
    #[test]
    fn the_margin_is_over_the_victims_count() {
        let x = steps(&[&[2], &[2], &[2], &[0], &[2], &[2]]);
        let mut r = rule(params(4, 96, 1.5, 2.0, 1.0), 4, 1, &[&[0]]);
        let (hits, flips) = replay(&mut r, &x);
        assert_eq!(hits, [3], "step 3 (0) and steps 4-5 (2)");
        assert_eq!(flips, [flip(0, 2, 0, 4)]);

        let mut r = rule(params(4, 96, 3.0, 2.0, 1.0), 4, 1, &[&[0]]);
        let (hits, flips) = replay(&mut r, &x);
        assert_eq!(hits, [1]);
        assert_eq!(flips, []);
    }

    /// Case 6: two uses clear margin 1.5 over an unused resident but not a
    /// minimum count of 3.
    #[test]
    fn min_count_holds_back_a_small_count() {
        let mut r = rule(params(2, 96, 1.5, 3.0, 1.0), 4, 1, &[&[0]]);
        let (_, flips) = replay(&mut r, &repeat(2, 3));
        assert_eq!(flips, []);
    }

    /// Case 7: two layers both want a flip at boundary 3; a cap of 1 over all
    /// layers admits the larger gain only (layer 1: 3 uses against layer 0's 2).
    #[test]
    fn the_cap_is_over_all_layers() {
        let p = params(3, 1, 1.5, 2.0, 1.0);
        let mut r = rule(p, 4, 1, &[&[0], &[0]]);
        let (hits, flips) = replay(&mut r, &steps2(&[(2, 2), (2, 2), (1, 2), (3, 3)]));
        assert_eq!(hits, [0, 0]);
        assert_eq!(flips, [flip(1, 2, 0, 3)]);

        let mut r = rule(p, 4, 1, &[&[0], &[0]]);
        let (hits, flips) = replay(&mut r, &steps2(&[(2, 2), (2, 2), (1, 2), (3, 2)]));
        assert_eq!(hits, [0, 1]);
        assert_eq!(flips, [flip(1, 2, 0, 3)]);
    }

    /// One layer with three spares wants three flips; a cap of 2 takes the first
    /// two by rank.
    #[test]
    fn the_cap_binds_inside_one_layer() {
        let mut p = params(2, 2, 1.5, 2.0, 1.0);
        p.spares = 3;
        let mut r = rule(p, 8, 3, &[&[0, 1, 2]]);
        let (_, flips) = replay(&mut r, &steps(&[&[5, 6, 7], &[5, 6, 7]]));
        assert_eq!(flips, [flip(0, 5, 0, 2), flip(0, 6, 1, 2)]);
    }

    /// Equal counts rank the lower id first: the candidate 3 before 5, the victim
    /// 1 before 2 (both unused). Equal gains across layers go to the lower layer
    /// first.
    #[test]
    fn ties_break_to_the_lower_id_then_the_lower_layer() {
        let mut r = rule(params(2, 96, 1.5, 2.0, 1.0), 8, 2, &[&[2, 1], &[2, 1]]);
        let row = vec![vec![5, 3], vec![3, 5]];
        let (_, flips) = replay(&mut r, &[row.clone(), row]);
        assert_eq!(flips, [flip(0, 3, 1, 2), flip(1, 3, 1, 2)]);
    }

    /// A verify pass's rejected rows never count: a history of 3-row passes that
    /// keep row 0 equals the history of those rows alone, flips and state.
    #[test]
    fn rejected_rows_leave_no_trace() {
        let p = params(1, 96, 1.5, 2.0, 1.0);
        let mut a = rule(p, 8, 1, &[&[0]]);
        let mut b = rule(p, 8, 1, &[&[0]]);
        let mut fa = Vec::new();
        let mut fb = Vec::new();
        for s in 1..=2u64 {
            a.observe(0, 0, &[2]).unwrap();
            a.observe(0, 1, &[3]).unwrap();
            a.observe(0, 2, &[4]).unwrap();
            a.end_pass(1).unwrap();
            fa.extend_from_slice(a.plan(s).unwrap());
            b.observe(0, 0, &[2]).unwrap();
            b.end_pass(1).unwrap();
            fb.extend_from_slice(b.plan(s).unwrap());
        }
        assert_eq!(fa, [flip(0, 2, 0, 2)]);
        assert_eq!(fa, fb);
        assert_eq!(a, b);
    }

    /// A small deterministic history: 3 layers of 16 experts, top-2, passes of 1
    /// to 3 rows keeping 1 to all of them.
    fn history(r: &mut SwapRule, passes: u64) -> Vec<Flip> {
        let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut flips = Vec::new();
        for b in 1..=passes {
            let rows = (next() % 3 + 1) as usize;
            for row in 0..rows {
                for layer in 0..3 {
                    let a = (next() % 6 + 4 * layer as u64 % 10) as u32;
                    let c = (a + 1 + (next() % 5) as u32) % 16;
                    r.observe(layer, row, &[a, c]).unwrap();
                }
            }
            let kept = (next() as usize) % rows + 1;
            r.end_pass(kept).unwrap();
            flips.extend_from_slice(r.plan(b).unwrap());
        }
        flips
    }

    fn history_rule() -> SwapRule {
        let seed: [&[u32]; 3] = [&[0, 1, 2, 3], &[4, 5, 6, 7], &[8, 9, 10, 11]];
        let shape = Shape {
            experts: 16,
            top_k: 2,
            max_rows: 3,
        };
        SwapRule::new(SwapParams::mid(2), shape, &seed, &[4, 4, 4]).unwrap()
    }

    #[test]
    fn the_same_history_gives_the_same_flips() {
        let mut a = history_rule();
        let mut b = history_rule();
        let fa = history(&mut a, 200);
        assert!(!fa.is_empty(), "the history makes flips");
        assert_eq!(fa, history(&mut b, 200));
        assert_eq!(a, b);
    }

    #[test]
    fn reset_is_a_fresh_rule() {
        let mut a = history_rule();
        let first = history(&mut a, 200);
        a.reset();
        assert_eq!(a, history_rule());
        assert_eq!(history(&mut a, 200), first);
    }

    /// Flips made at a boundary where earlier flips go live see the map after
    /// them: at d = every the victim of the second flip is the first's admit.
    #[test]
    fn flips_land_before_the_pass_plans() {
        let mut p = params(2, 96, 1.5, 2.0, 0.0);
        p.delay = 2;
        let mut r = rule(p, 4, 1, &[&[0]]);
        let (_, flips) = replay(&mut r, &steps(&[&[2], &[2], &[3], &[3]]));
        assert_eq!(flips, [flip(0, 2, 0, 4), flip(0, 3, 2, 6)]);
    }

    /// Every 4 steps a new expert, 10, 11, 12, …, takes all a layer's routing.
    fn a_new_hot_expert_every_pass() -> Vec<Vec<Vec<u32>>> {
        (0..24u32).map(|s| vec![vec![10 + s / 4]]).collect()
    }

    /// A layer never has more flips in flight than spares: at the mid rule with
    /// d = 8 = 2 × every and one spare, a layer flips at every other planning
    /// pass.
    #[test]
    fn one_spare_holds_one_flip_in_flight() {
        let mut r = rule(SwapParams::mid(8), 32, 1, &[&[0, 1, 2, 3]]);
        let (_, flips) = replay(&mut r, &a_new_hot_expert_every_pass());
        assert_eq!(
            flips,
            [flip(0, 10, 0, 12), flip(0, 12, 1, 20), flip(0, 14, 2, 28)]
        );
    }

    /// Two spares at d = 8 flip at every planning pass while the victims'
    /// counts leave the margin, and never hold a third flip.
    #[test]
    fn two_spares_hold_two_flips_in_flight() {
        let mut p = SwapParams::mid(8);
        p.spares = 2;
        let mut r = rule(p, 32, 1, &[&[0, 1, 2, 3]]);
        let (_, flips) = replay(&mut r, &a_new_hot_expert_every_pass());
        assert_eq!(
            flips,
            [
                flip(0, 10, 0, 12),
                flip(0, 11, 1, 16),
                flip(0, 12, 2, 20),
                flip(0, 13, 3, 24)
            ]
        );
    }

    /// The opening relayout: per layer the most-used off-card experts against
    /// the least-used on it while the admitted count is higher; gains across
    /// layers largest first, ties to the lower layer; live at once. Layer 1's
    /// residents tie at 1 use: the worse-ranked seed expert 5 leaves first.
    #[test]
    fn open_relays_out_from_whole_prompt_counts() {
        let seed: [&[u32]; 2] = [&[0, 1], &[4, 5]];
        let fresh = || rule(params(4, 24, 3.0, 2.0, 0.9), 6, 1, &seed);
        let counts = [1, 5, 3, 3, 0, 2, 4, 4, 0, 0, 1, 1];
        let mut r = fresh();
        assert_eq!(
            r.open(&counts, 2).unwrap(),
            [flip(1, 0, 5, 0), flip(1, 1, 4, 0)]
        );
        assert_eq!(r.live(1).unwrap().collect::<Vec<_>>(), [0, 1]);
        assert_eq!(r.live(0).unwrap().collect::<Vec<_>>(), [0, 1]);
        let mut r = fresh();
        assert_eq!(
            r.open(&counts, usize::MAX).unwrap(),
            [flip(1, 0, 5, 0), flip(1, 1, 4, 0), flip(0, 2, 0, 0)]
        );
        assert_eq!(r.live(0).unwrap().collect::<Vec<_>>(), [1, 2]);
    }

    /// A 512-row prompt that routes only 5 and 6 leaves the residents 3, 1, 2
    /// and 0 tied at 0 uses: the worst-ranked resident leaves first (0, then 2),
    /// and of the tied 5 and 6 the better-ranked 6 enters first; by id alone it
    /// would be 5 for 0, then 6 for 1.
    #[test]
    fn open_breaks_raw_zero_ties_by_seed_rank() {
        let seed: [&[u32]; 1] = [&[3, 1, 2, 0, 6, 5]];
        let shape = Shape {
            experts: 8,
            top_k: 2,
            max_rows: 1,
        };
        let mut r = SwapRule::new(SwapParams::mid(0), shape, &seed, &[4]).unwrap();
        let mut counts = [0u32; 8];
        for _ in 0..512 {
            for id in [5, 6] {
                counts[id] += 1;
            }
        }
        assert_eq!(
            r.open(&counts, usize::MAX).unwrap(),
            [flip(0, 6, 0, 0), flip(0, 5, 2, 0)]
        );
        assert_eq!(r.live(0).unwrap().collect::<Vec<_>>(), [1, 3, 5, 6]);
    }

    /// Pinned seed experts are never a victim: an opening relayout that would
    /// swap out every resident leaves the pinned one, and a reset brings the
    /// pinned rule back to its own fresh state, not the unpinned one's.
    #[test]
    fn a_pinned_expert_is_never_a_victim() {
        let seed: [&[u32]; 1] = [&[0, 1, 2, 3]];
        let shape = Shape {
            experts: 8,
            top_k: 1,
            max_rows: 1,
        };
        let pinned =
            |p: usize| SwapRule::new_pinned(SwapParams::mid(0), shape, &seed, &[3], &[p]).unwrap();
        let counts = [0, 0, 0, 0, 9, 9, 9, 9];
        let mut r = pinned(1);
        assert_eq!(
            r.open(&counts, usize::MAX).unwrap(),
            [flip(0, 4, 2, 0), flip(0, 5, 1, 0)]
        );
        assert_eq!(r.live(0).unwrap().collect::<Vec<_>>(), [0, 4, 5]);
        r.reset();
        assert_eq!(r, pinned(1));
        assert_ne!(r, pinned(0));

        let mut r = pinned(2);
        let (_, flips) = replay(&mut r, &repeat(6, 8));
        assert_eq!(flips, [flip(0, 6, 2, 4)], "the unpinned resident 2 leaves");
        let mut r = pinned(3);
        let (_, flips) = replay(&mut r, &repeat(6, 8));
        assert_eq!(flips, [], "every resident pinned: no victim");
        assert_eq!(
            SwapRule::new_pinned(SwapParams::mid(0), shape, &seed, &[3], &[4]).unwrap_err(),
            SwapRuleError::PinnedOverCapacity {
                layer: 0,
                pinned: 4,
                capacity: 3
            }
        );
        assert!(matches!(
            SwapRule::new_pinned(SwapParams::mid(0), shape, &seed, &[3], &[1, 1]).unwrap_err(),
            SwapRuleError::Shape { .. }
        ));
    }

    /// An away expert is never admitted, however hot, and never in the card
    /// set: the opening relayout and a planning pass take the next candidate,
    /// a reset keeps it away, and the seed and pinned accessors read the
    /// rule's own lists.
    #[test]
    fn an_away_expert_is_never_admitted() {
        let seed: [&[u32]; 1] = [&[0, 1, 2, 3]];
        let shape = Shape {
            experts: 8,
            top_k: 1,
            max_rows: 1,
        };
        let placed = |away: &[u32]| {
            SwapRule::new_placed(SwapParams::mid(0), shape, &seed, &[3], &[1], &[away]).unwrap()
        };
        let counts = [0, 0, 0, 0, 9, 8, 0, 0];
        let mut r = placed(&[4]);
        assert_eq!(
            r.open(&counts, usize::MAX).unwrap(),
            [flip(0, 5, 2, 0)],
            "4 is away: 5 enters, and only one victim goes"
        );
        assert_eq!(r.live(0).unwrap().collect::<Vec<_>>(), [0, 1, 5]);
        assert!(!r.is_live(0, 4).unwrap());
        r.reset();
        assert_eq!(r, placed(&[4]));
        assert_ne!(r, placed(&[]));
        assert_eq!(r.seed(0).unwrap(), [0, 1, 2]);
        assert_eq!(r.pinned(0).unwrap(), 1);
        assert!(r.seed(1).is_err() && r.pinned(1).is_err());

        let mut r = placed(&[6]);
        let (_, flips) = replay(&mut r, &repeat(6, 12));
        assert_eq!(flips, [], "the only routed expert is away");
        assert!(r.live(0).unwrap().all(|id| id != 6));
        let mut r = placed(&[]);
        let (_, flips) = replay(&mut r, &repeat(6, 12));
        assert!(
            !flips.is_empty(),
            "the same trace admits 6 when it is not away"
        );

        let refused = |away: &[u32]| {
            SwapRule::new_placed(SwapParams::mid(0), shape, &seed, &[3], &[0], &[away]).unwrap_err()
        };
        assert_eq!(refused(&[3]), SwapRuleError::AwayInSeed { layer: 0, id: 3 });
        assert_eq!(
            refused(&[5, 5]),
            SwapRuleError::AwayDuplicate { layer: 0, id: 5 }
        );
        assert!(matches!(
            refused(&[8]),
            SwapRuleError::ExpertOutOfRange { .. }
        ));
        assert!(matches!(
            SwapRule::new_placed(SwapParams::mid(0), shape, &seed, &[3], &[0], &[&[], &[]])
                .unwrap_err(),
            SwapRuleError::Shape { .. }
        ));
    }

    #[test]
    fn undefined_input_is_refused_by_name() {
        use SwapRuleError as E;
        let shape = Shape {
            experts: 4,
            top_k: 2,
            max_rows: 2,
        };
        let p = params(2, 4, 1.5, 2.0, 0.9);
        let new =
            |p, seed: &[&[u32]], cap: &[usize]| SwapRule::new(p, shape, seed, cap).unwrap_err();
        let ranked = SwapRule::new(p, shape, &[&[2, 0, 3]], &[2]).unwrap();
        assert_eq!(ranked.live(0).unwrap().collect::<Vec<_>>(), [0, 2]);
        assert_eq!(
            new(p, &[&[0]], &[2]),
            E::SeedUnderCapacity {
                layer: 0,
                seed: 1,
                capacity: 2
            }
        );
        assert_eq!(
            new(p, &[&[1, 1]], &[2]),
            E::SeedDuplicate { layer: 0, id: 1 }
        );
        assert_eq!(
            new(p, &[&[0, 4]], &[2]),
            E::ExpertOutOfRange {
                layer: 0,
                id: 4,
                experts: 4
            }
        );
        let mut bad = p;
        bad.decay = 1.5;
        assert!(matches!(
            new(bad, &[&[0]], &[1]),
            E::Param { name: "decay", .. }
        ));
        let mut bad = p;
        bad.margin = f64::NAN;
        assert!(matches!(
            new(bad, &[&[0]], &[1]),
            E::Param { name: "margin", .. }
        ));
        let mut bad = p;
        bad.spares = 0;
        assert!(matches!(
            new(bad, &[&[0]], &[1]),
            E::Param { name: "spares", .. }
        ));

        let mut r = SwapRule::new(p, shape, &[&[0], &[1]], &[1, 1]).unwrap();
        assert_eq!(
            r.observe(2, 0, &[0, 1]),
            Err(E::LayerOutOfRange {
                layer: 2,
                layers: 2
            })
        );
        assert_eq!(
            r.observe(0, 2, &[0, 1]),
            Err(E::RowOutOfRange {
                row: 2,
                max_rows: 2
            })
        );
        assert_eq!(
            r.observe(0, 0, &[0]),
            Err(E::RowLength {
                layer: 0,
                row: 0,
                len: 1,
                top_k: 2
            })
        );
        assert_eq!(
            r.observe(0, 0, &[0, 7]),
            Err(E::ExpertOutOfRange {
                layer: 0,
                id: 7,
                experts: 4
            })
        );
        assert_eq!(
            r.observe(0, 0, &[3, 3]),
            Err(E::DuplicateId {
                layer: 0,
                row: 0,
                id: 3
            })
        );
        assert_eq!(
            r.is_live(0, 4),
            Err(E::ExpertOutOfRange {
                layer: 0,
                id: 4,
                experts: 4
            })
        );
        assert_eq!(r.plan(1), Err(E::PassNotEnded { boundary: 1 }));
        r.observe(0, 0, &[0, 1]).unwrap();
        assert_eq!(
            r.observe(0, 0, &[2, 3]),
            Err(E::RowObservedTwice { layer: 0, row: 0 })
        );
        assert_eq!(r.open(&[0; 8], 1), Err(E::RowsPending { rows: 1 }));
        assert_eq!(r.end_pass(2), Err(E::KeptPastRows { kept: 2, rows: 1 }));
        assert_eq!(r.end_pass(1), Err(E::RowMissing { layer: 1, row: 0 }));
        r.observe(1, 0, &[0, 1]).unwrap();
        r.end_pass(1).unwrap();
        assert_eq!(r.end_pass(0), Err(E::PlanSkipped { boundary: 1 }));
        assert_eq!(
            r.plan(2),
            Err(E::Boundary {
                got: 2,
                expected: 1
            })
        );
        assert_eq!(r.plan(1).unwrap(), []);
        assert_eq!(r.open(&[0; 7], 1), Err(E::CountsShape { len: 7, want: 8 }));
    }

    mod fixture {
        use super::super::{Flip, Shape, SwapParams, SwapRule};

        // The residency rule against the fixtures of `tools/ref/router-residency.py`
        // (`tests/data/swaprule-fixture.json` for `plan`, `swaprule-open-fixture.json`
        // for `open`, `-pinned-` and `-away-` for those states; each
        // `header.command` names the command that wrote it).
        /// Just enough JSON for the fixture: objects, arrays, numbers, strings.
        #[derive(Debug)]
        enum Json {
            Num(f64),
            Str(String),
            Arr(Vec<Json>),
            Obj(Vec<(String, Json)>),
        }

        struct Parser<'a> {
            s: &'a [u8],
            at: usize,
        }

        impl Parser<'_> {
            fn ws(&mut self) {
                while self.s.get(self.at).is_some_and(u8::is_ascii_whitespace) {
                    self.at += 1;
                }
            }

            fn eat(&mut self, c: u8) {
                self.ws();
                assert_eq!(
                    self.s.get(self.at),
                    Some(&c),
                    "fixture: '{}' at byte {}",
                    c as char,
                    self.at
                );
                self.at += 1;
            }

            fn value(&mut self) -> Json {
                self.ws();
                match self.s[self.at] {
                    b'{' => {
                        self.at += 1;
                        let mut kv = Vec::new();
                        self.ws();
                        if self.s[self.at] == b'}' {
                            self.at += 1;
                            return Json::Obj(kv);
                        }
                        loop {
                            let Json::Str(k) = self.value() else {
                                panic!("fixture: a key at byte {}", self.at)
                            };
                            self.eat(b':');
                            kv.push((k, self.value()));
                            self.ws();
                            self.at += 1;
                            match self.s[self.at - 1] {
                                b',' => {}
                                b'}' => return Json::Obj(kv),
                                c => panic!("fixture: '{}' at byte {}", c as char, self.at - 1),
                            }
                        }
                    }
                    b'[' => {
                        self.at += 1;
                        let mut v = Vec::new();
                        self.ws();
                        if self.s[self.at] == b']' {
                            self.at += 1;
                            return Json::Arr(v);
                        }
                        loop {
                            v.push(self.value());
                            self.ws();
                            self.at += 1;
                            match self.s[self.at - 1] {
                                b',' => {}
                                b']' => return Json::Arr(v),
                                c => panic!("fixture: '{}' at byte {}", c as char, self.at - 1),
                            }
                        }
                    }
                    b'"' => {
                        let start = self.at + 1;
                        let len = self.s[start..]
                            .iter()
                            .position(|&c| c == b'"')
                            .expect("fixture: an unterminated string");
                        assert!(
                            !self.s[start..start + len].contains(&b'\\'),
                            "fixture: an escaped string"
                        );
                        self.at = start + len + 1;
                        Json::Str(String::from_utf8(self.s[start..start + len].to_vec()).unwrap())
                    }
                    _ => {
                        let start = self.at;
                        while self
                            .s
                            .get(self.at)
                            .is_some_and(|c| c.is_ascii_digit() || b"+-.eE".contains(c))
                        {
                            self.at += 1;
                        }
                        let text = std::str::from_utf8(&self.s[start..self.at]).unwrap();
                        Json::Num(
                            text.parse()
                                .unwrap_or_else(|_| panic!("fixture: number {text:?}")),
                        )
                    }
                }
            }
        }

        impl Json {
            fn find(&self, key: &str) -> Option<&Json> {
                let Json::Obj(kv) = self else {
                    panic!("fixture: {key} of a non-object")
                };
                kv.iter().find(|(k, _)| k == key).map(|(_, v)| v)
            }

            fn get(&self, key: &str) -> &Json {
                self.find(key)
                    .unwrap_or_else(|| panic!("fixture: no key {key}"))
            }

            fn num(&self) -> f64 {
                let Json::Num(n) = *self else {
                    panic!("fixture: a number, got {self:?}")
                };
                n
            }

            fn int(&self) -> u64 {
                let n = self.num();
                assert!(n >= 0.0 && n.fract() == 0.0, "fixture: an integer, got {n}");
                n as u64
            }

            fn arr(&self) -> &[Json] {
                let Json::Arr(v) = self else {
                    panic!("fixture: an array, got {self:?}")
                };
                v
            }

            fn ints(&self) -> Vec<u64> {
                self.arr().iter().map(Json::int).collect()
            }
        }

        fn parse(text: &[u8]) -> Json {
            let mut p = Parser { s: text, at: 0 };
            let v = p.value();
            p.ws();
            assert_eq!(p.at, text.len(), "fixture: bytes after the value");
            v
        }

        fn fixture() -> Json {
            parse(include_bytes!("../tests/data/swaprule-fixture.json"))
        }

        fn pinned_fixture() -> Json {
            parse(include_bytes!("../tests/data/swaprule-pinned-fixture.json"))
        }

        fn away_fixture() -> Json {
            parse(include_bytes!("../tests/data/swaprule-away-fixture.json"))
        }

        fn open_fixture() -> Json {
            parse(include_bytes!("../tests/data/swaprule-open-fixture.json"))
        }

        fn ints_u32(v: &Json) -> Vec<u32> {
            v.ints().into_iter().map(|x| x as u32).collect()
        }

        fn usizes(v: &Json) -> Vec<usize> {
            v.ints().into_iter().map(|x| x as usize).collect()
        }

        /// Each layer's seed list, ranked best first.
        fn seed_of(f: &Json) -> Vec<Vec<u32>> {
            f.get("seed").arr().iter().map(ints_u32).collect()
        }

        /// One plan case: passes of 1 to 3 rows, the first `kept[p]` of pass
        /// `p` counted, every flip in the order the rule makes it.
        fn plan_case(f: &Json) {
            let p = f.get("params");
            let params = SwapParams {
                every: p.get("every").int(),
                cap: p.get("cap").int() as usize,
                margin: p.get("margin").num(),
                min_count: p.get("min_count").num(),
                decay: p.get("decay").num(),
                spares: p.get("spares").int() as usize,
                delay: p.get("d").int(),
            };
            let trace = f.get("trace").arr();
            let kept = usizes(f.get("kept"));
            let layers = trace.len();
            let max_rows = trace
                .iter()
                .flat_map(|l| l.arr().iter().map(|pass| pass.arr().len()))
                .max()
                .unwrap();
            let shape = Shape {
                experts: p.get("n_expert").int() as usize,
                top_k: p.get("top_k").int() as usize,
                max_rows,
            };
            let seed = seed_of(f);
            let seed_refs: Vec<&[u32]> = seed.iter().map(Vec::as_slice).collect();
            let capacity = usizes(p.get("capacity"));
            let pinned = p
                .find("pinned")
                .map_or_else(|| vec![0; capacity.len()], usizes);
            let away: Vec<Vec<u32>> = p.find("away").map_or_else(
                || vec![Vec::new(); capacity.len()],
                |a| a.arr().iter().map(ints_u32).collect(),
            );
            let away_refs: Vec<&[u32]> = away.iter().map(Vec::as_slice).collect();
            let mut r =
                SwapRule::new_placed(params, shape, &seed_refs, &capacity, &pinned, &away_refs)
                    .unwrap();
            let mut got = Vec::new();
            for (pass, &k) in kept.iter().enumerate() {
                for (layer, l) in trace.iter().enumerate() {
                    for (row, ids) in l.arr()[pass].arr().iter().enumerate() {
                        r.observe(layer, row, &ints_u32(ids)).unwrap();
                    }
                }
                r.end_pass(k).unwrap();
                let b = pass as u64 + 1;
                got.extend(r.plan(b).unwrap().iter().map(|&fl| (b, fl)));
            }
            assert_eq!(layers, capacity.len());
            let want: Vec<(u64, Flip)> = f
                .get("flips")
                .arr()
                .iter()
                .map(|x| {
                    let flip = Flip {
                        layer: x.get("layer").int() as usize,
                        admit: x.get("in").int() as u32,
                        evict: x.get("out").int() as u32,
                        live_at: x.get("live_at").int(),
                    };
                    (x.get("boundary").int(), flip)
                })
                .collect();
            assert!(!want.is_empty());
            let first = got.iter().zip(&want).position(|(g, w)| g != w);
            assert!(
                first.is_none() && got.len() == want.len(),
                "the first differing flip, index {first:?}: rule {:?}, fixture {:?} ({} flips against {})",
                first.map(|j| got[j]),
                first.map(|j| want[j]),
                got.len(),
                want.len()
            );
        }

        /// The plan fixture's main case: d = 8, one spare, in-flight flips
        /// blocking a layer.
        #[test]
        fn the_plan_replay() {
            plan_case(&fixture());
        }

        /// The case where the cap over all layers binds, at two spares.
        #[test]
        fn the_cap_binding_replay() {
            let f = fixture();
            let case = f.get("cap_case");
            assert!(
                case.get("header").get("cap_bound").int() > 0,
                "the cap binds somewhere"
            );
            assert_eq!(case.get("params").get("spares").int(), 2);
            plan_case(case);
        }

        /// The pinned fixture: the plan fixture's seed and trace with every
        /// layer's first 8 seed experts pinned, in both its cases; the file
        /// moves flips against the unpinned rule (its header's `pinned_moved`).
        #[test]
        fn the_pinned_replay() {
            let f = pinned_fixture();
            assert!(
                f.get("header").get("pinned_moved").int() > 0,
                "pinning moves some flip"
            );
            plan_case(&f);
            plan_case(f.get("cap_case"));
        }

        /// The away fixture: the plan fixture's seed and trace with one id a layer
        /// away (on another device), in both its cases; the file moves flips
        /// against the rule with none away (its header's `away_moved`), and no
        /// flip admits an away id.
        #[test]
        fn the_away_replay() {
            let f = away_fixture();
            assert!(
                f.get("header").get("away_moved").int() > 0,
                "an away id moves some flip"
            );
            for case in [&f, f.get("cap_case")] {
                let away: Vec<Vec<u32>> = case
                    .get("params")
                    .get("away")
                    .arr()
                    .iter()
                    .map(ints_u32)
                    .collect();
                assert!(case.get("flips").arr().iter().all(|x| {
                    !away[x.get("layer").int() as usize].contains(&(x.get("in").int() as u32))
                }));
                plan_case(case);
            }
        }

        /// The open fixture: a prompt's whole counts, ties to the seed rank; the
        /// first `m` flips, then every flip on a fresh rule.
        #[test]
        fn the_opening_replay() {
            let f = open_fixture();
            let p = f.get("params");
            let e = p.get("n_expert").int() as usize;
            let shape = Shape {
                experts: e,
                top_k: p.get("top_k").int() as usize,
                max_rows: 1,
            };
            let seed = seed_of(&f);
            let seed_refs: Vec<&[u32]> = seed.iter().map(Vec::as_slice).collect();
            let capacity = usizes(p.get("capacity"));
            let fresh = || SwapRule::new(SwapParams::mid(0), shape, &seed_refs, &capacity).unwrap();
            let counts: Vec<u32> = f.get("counts").arr().iter().flat_map(ints_u32).collect();
            let mut tally = vec![0u32; counts.len()];
            for (layer, rows) in f.get("prompt").arr().iter().enumerate() {
                for ids in rows.arr() {
                    for id in ints_u32(ids) {
                        tally[layer * e + id as usize] += 1;
                    }
                }
            }
            assert_eq!(tally, counts, "the fixture's counts are its prompt's");
            let flips = |key: &str| -> Vec<Flip> {
                f.get(key)
                    .arr()
                    .iter()
                    .map(|x| Flip {
                        layer: x.get("layer").int() as usize,
                        admit: x.get("in").int() as u32,
                        evict: x.get("out").int() as u32,
                        live_at: 0,
                    })
                    .collect()
            };
            let m = p.get("m").int() as usize;
            let want = flips("flips");
            assert_eq!(want.len(), m);
            assert_eq!(fresh().open(&counts, m).unwrap(), want);
            let all = flips("flips_all");
            assert!(all.len() > m, "m cuts the pairs");
            assert_eq!(fresh().open(&counts, usize::MAX).unwrap(), all);
        }
    }
}
