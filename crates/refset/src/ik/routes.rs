//! ik's routing of the prefill, as `dump_ref --prefill-routes` writes it
//! (`tools/ref/dump_ref.cpp`): the nodes `<stem>-<layer>` of each ubatch of the
//! quiet prefill are `tensor` rows named `prefill.<stem>-<layer>`, one
//! occurrence a ubatch in prefill order, their positions following in order
//! from 0. [`RefManifest::prefill_routes`] reads the three stems a family's
//! profile names (the picks, the gathered weights and the final weights) into
//! one [`PrefillRoutes`]; each family maps that into its own plant format.

use super::{Layout, RefManifest, RefRow, RowKind, ref_ints, ref_tensor_logical_in};
use crate::RefError;
use std::collections::BTreeMap;
use std::fmt::Display;
use std::path::PathBuf;

/// One routed layer's rows over the whole prefill. Position `p` of the prefill
/// holds `n_used` values at `p * n_used` in each vector.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteLayer {
    pub layer: u32,
    /// The experts ik's router picked, in ggml's top-k order; the `n_used`
    /// ids of a position are distinct.
    pub picks: Vec<u32>,
    /// The picked experts' gathered router weights, as dumped.
    pub raw: Vec<f32>,
    /// The last node of ik's weight chain, as dumped: the weights the experts'
    /// outputs are summed with.
    pub last: Vec<f32>,
}

/// The routing of a whole prefill ([`RefManifest::prefill_routes`]).
#[derive(Debug, Clone, PartialEq)]
pub struct PrefillRoutes {
    set: PathBuf,
    /// The experts picked a position.
    pub n_used: usize,
    /// The prefill's positions, `# prefill`.
    pub positions: usize,
    /// The ubatches the prefill ran in.
    pub ubatches: usize,
    /// The routed layers, ascending.
    pub layers: Vec<RouteLayer>,
}

impl PrefillRoutes {
    /// Every pick is an expert below `n_expert`, the model's expert count,
    /// which the set does not state; the first one that is not is an error
    /// naming its layer and position.
    pub fn check_experts(&self, n_expert: u32) -> Result<(), RefError> {
        for l in &self.layers {
            if let Some(i) = l.picks.iter().position(|&id| id >= n_expert) {
                return Err(RefError::malformed(
                    format!("prefill_routes: {}", self.set.display()),
                    format!(
                        "layer {} position {} picks expert {}, the model has {n_expert}",
                        l.layer,
                        i / self.n_used,
                        l.picks[i]
                    ),
                ));
            }
        }
        Ok(())
    }
}

/// The row a stem names at `layer`.
fn route_name(stem: &str, layer: u32) -> String {
    format!("prefill.{stem}-{layer}")
}

/// `(n_used, positions)` of a picks row: ik's top-k view is `[n_used,
/// positions]`.
fn picks_ne(row: &RefRow) -> Option<(u64, u64)> {
    match row.ne {
        [k, n, 1, 1] if k > 0 && n > 0 => Some((k, n)),
        _ => None,
    }
}

/// The positions of a weights row of `n_used` weights a position: ik's
/// normalised weights are `[n_used, positions]`, its gathered and scaled ones
/// `[1, n_used, positions]`.
fn weights_positions(row: &RefRow, n_used: u64) -> Option<u64> {
    match row.ne {
        [k, n, 1, 1] | [1, k, n, 1] if k == n_used => Some(n),
        _ => None,
    }
}

impl RefManifest {
    /// ik's routing of the prefill, from the rows `prefill.<stem>-<layer>` of
    /// the three stems: `picks` (the top-k view, read from its logical `i32`
    /// twin, ggml index order), `raw` (the gathered router weights) and `last`
    /// (the last node of ik's weight chain, which the engine's step
    /// multiplies the experts' outputs by). The weights are read as dumped and
    /// recomputed by nothing.
    ///
    /// A layer's vectors hold the ubatches in occurrence order, each ubatch's
    /// positions in order, so position `p` of the prefill (from 0, `# prefill`
    /// positions in all) holds its `n_used` values at `p * n_used`, picks and
    /// weights in the same order.
    ///
    /// Each of these is an error naming the set and the row: a set dumped
    /// without `--prefill-routes` (no `# prefill_routes` line) or with no
    /// `# prefill` line, a stem with no row, stems naming different layers,
    /// occurrences that do not run 0..U the same for every row, ne that do
    /// not count the same positions at one occurrence and the same `n_used`
    /// throughout, ubatches that do not sum to `# prefill`, a pick that is
    /// repeated within a position or is no `u32`, and a weight that is not
    /// finite.
    pub fn prefill_routes(
        &self,
        picks: &str,
        raw: &str,
        last: &str,
    ) -> Result<PrefillRoutes, RefError> {
        let stems = [picks, raw, last];
        let no = |what: &str| {
            RefError::missing(
                self.dir.join("MANIFEST.tsv"),
                format!("prefill_routes: {} has no {what}", self.dir.display()),
            )
        };
        if !self
            .header
            .other
            .iter()
            .any(|l| l.starts_with("# prefill_routes\t"))
        {
            return Err(no(
                "# prefill_routes line: the set was dumped without --prefill-routes",
            ));
        }
        let Some(prefill) = self.header.prefill else {
            return Err(no("# prefill line"));
        };
        let mut found = Vec::with_capacity(stems.len());
        for stem in stems {
            found.push(self.route_rows(stem)?);
        }
        let layers: Vec<u32> = found[0].keys().copied().collect();
        for (stem, rows) in stems.iter().zip(&found).skip(1) {
            if !rows.keys().eq(&layers) {
                return Err(self.bad(format!(
                    "the stems name different layers: {picks} has {layers:?}, {stem} has {:?}",
                    rows.keys().collect::<Vec<_>>()
                )));
            }
        }
        let occs0 = &found[0][&layers[0]];
        for (stem, rows) in stems.iter().zip(&found) {
            for (layer, occs) in rows {
                if !occs.iter().zip(0u32..).all(|(&o, i)| o == i) {
                    return Err(self.bad(format!(
                        "{} has occurrences {occs:?}, want 0..{}",
                        route_name(stem, *layer),
                        occs.len()
                    )));
                }
                if occs.len() != occs0.len() {
                    return Err(self.bad(format!(
                        "{} has {} occurrences, {} has {}",
                        route_name(stem, *layer),
                        occs.len(),
                        route_name(picks, layers[0]),
                        occs0.len()
                    )));
                }
            }
        }
        let (n_used, widths) = self.route_shape(stems, &layers, occs0, prefill)?;
        let mut out = Vec::with_capacity(layers.len());
        for &layer in &layers {
            out.push(self.route_layer(stems, layer, n_used, &widths)?);
        }
        Ok(PrefillRoutes {
            set: self.dir.clone(),
            n_used,
            positions: prefill as usize,
            ubatches: occs0.len(),
            layers: out,
        })
    }

    /// The layers and occurrences of the rows named `prefill.<stem>-<layer>`,
    /// the whole name as the dumper matches a routing node: a stem, a dash and
    /// digits. A stem with no row is an error.
    fn route_rows(&self, stem: &str) -> Result<BTreeMap<u32, Vec<u32>>, RefError> {
        let head = format!("prefill.{stem}-");
        let mut layers: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for r in &self.tensors {
            let Some(digits) = r.name.strip_prefix(&head) else {
                continue;
            };
            if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                continue;
            }
            let layer = digits
                .parse()
                .map_err(|e| self.bad(format!("{}: layer {digits}: {e}", r.name)))?;
            layers.entry(layer).or_default().push(r.occurrence);
        }
        if layers.is_empty() {
            return Err(RefError::missing(
                &self.dir,
                format!(
                    "prefill_routes: {} has no tensor row {head}<layer>",
                    self.dir.display()
                ),
            ));
        }
        for occs in layers.values_mut() {
            occs.sort_unstable();
        }
        Ok(layers)
    }

    /// `n_used` and the positions of each ubatch, from the ne of every row:
    /// the three stems count the same positions at an occurrence, and every
    /// layer has the same `n_used` and the same ubatch widths, which sum to
    /// `# prefill`.
    fn route_shape(
        &self,
        [picks, raw, last]: [&str; 3],
        layers: &[u32],
        occs: &[u32],
        prefill: u32,
    ) -> Result<(usize, Vec<usize>), RefError> {
        let mut n_used = None;
        let mut widths: Vec<u64> = Vec::with_capacity(occs.len());
        for (at, &layer) in layers.iter().enumerate() {
            for (ub, &u) in occs.iter().enumerate() {
                let prow = self.tensor(&route_name(picks, layer), u)?;
                let Some((k, n)) = picks_ne(prow) else {
                    return Err(self.bad(format!(
                        "{}/{u} has ne {:?}, want [n_used, positions, 1, 1] with both above 0",
                        prow.name, prow.ne
                    )));
                };
                for stem in [raw, last] {
                    let row = self.tensor(&route_name(stem, layer), u)?;
                    match weights_positions(row, k) {
                        Some(m) if m == n => {}
                        Some(m) => {
                            return Err(self.bad(format!(
                                "{}/{u} counts {m} positions where {}/{u} counts {n}",
                                row.name, prow.name
                            )));
                        }
                        None => {
                            return Err(self.bad(format!(
                                "{}/{u} has ne {:?}, want [{k}, positions, 1, 1] or [1, {k}, positions, 1]",
                                row.name, row.ne
                            )));
                        }
                    }
                }
                if *n_used.get_or_insert(k) != k {
                    return Err(self.bad(format!(
                        "{}/{u} picks {k} experts a position, an earlier row {}",
                        prow.name,
                        n_used.unwrap_or(0)
                    )));
                }
                if at == 0 {
                    widths.push(n);
                } else if widths[ub] != n {
                    return Err(self.bad(format!(
                        "{}/{u} counts {n} positions, layer {} counts {}",
                        prow.name, layers[0], widths[ub]
                    )));
                }
            }
        }
        if widths.iter().try_fold(0u64, |a, &w| a.checked_add(w)) != Some(u64::from(prefill)) {
            return Err(self.bad(format!(
                "the ubatches of {picks} count {widths:?} positions, # prefill says {prefill}"
            )));
        }
        Ok((
            n_used.unwrap_or(0) as usize,
            widths.iter().map(|&w| w as usize).collect(),
        ))
    }

    /// One layer's picks and weights over every ubatch, each file read through
    /// the per-row readers; a pick must be a `u32` and distinct within its
    /// position.
    fn route_layer(
        &self,
        [picks, raw, last]: [&str; 3],
        layer: u32,
        n_used: usize,
        widths: &[usize],
    ) -> Result<RouteLayer, RefError> {
        let mut out = RouteLayer {
            layer,
            picks: Vec::new(),
            raw: Vec::new(),
            last: Vec::new(),
        };
        let mut first = 0;
        for (u, &width) in (0u32..).zip(widths) {
            let prow = self.tensor(&route_name(picks, layer), u)?;
            let ids = ref_ints(self, &prow.name, u, RowKind::Tensor, Layout::Logical)
                .map_err(|e| self.route_err(prow, e))?;
            if ids.len() as u64 != prow.count() {
                return Err(self.bad(format!(
                    "{}/{u} has {} picks, its integer twin {}",
                    prow.name,
                    prow.count(),
                    ids.len()
                )));
            }
            for (p, ids_p) in ids.chunks_exact(n_used).enumerate() {
                let at = out.picks.len();
                for &id in ids_p {
                    out.picks.push(u32::try_from(id).map_err(|_| {
                        self.bad(format!(
                            "{}/{u} position {} picks expert {id}, not a u32",
                            prow.name,
                            first + p
                        ))
                    })?);
                }
                let picked = &out.picks[at..];
                if let Some((j, id)) = picked
                    .iter()
                    .enumerate()
                    .find(|&(j, id)| picked[..j].contains(id))
                {
                    return Err(self.bad(format!(
                        "{}/{u} position {} picks expert {id} twice (slot {j})",
                        prow.name,
                        first + p
                    )));
                }
            }
            for (stem, dst) in [(raw, &mut out.raw), (last, &mut out.last)] {
                let row = self.tensor(&route_name(stem, layer), u)?;
                dst.extend(
                    ref_tensor_logical_in(&self.dir, row).map_err(|e| self.route_err(row, e))?,
                );
            }
            first += width;
        }
        Ok(out)
    }

    /// An error of `prefill_routes` naming this set.
    fn bad(&self, what: impl Display) -> RefError {
        RefError::malformed(format!("prefill_routes: {}", self.dir.display()), what)
    }

    /// A per-row reader's refusal, naming the set and the row it read; the
    /// `Missing` ones already name the file's path.
    fn route_err(&self, row: &RefRow, e: RefError) -> RefError {
        match e {
            RefError::Malformed { at, what } => {
                self.bad(format!("{}/{}: {at}: {what}", row.name, row.occurrence))
            }
            e => e,
        }
    }
}
