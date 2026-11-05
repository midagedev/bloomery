//! A router rank: for each routed layer, every expert id once, hottest first
//! (`BLOOMERY_EXPERT_RANK`, written by `tools/ref/router-rank.py` from router
//! sets). The planner reads it as an order over the ids of a layer: the stage
//! card keeps the layer's hottest `n_l`, each expert tier card a block from
//! the coldest end, the host the middle ([`ExpertRank::split`]); the NVMe
//! expert tier, paging what the host cannot hold, takes the coldest of the
//! host's own ids ([`ExpertRank::cold_split`]).
//!
//! The file is one TSV: a header line
//! `# router-rank<TAB>n_expert=<N><TAB>n_used=<U><TAB>sets=<a,b><TAB>positions=<total>`,
//! then one row `<layer><TAB><id>,<id>,...` per routed layer. A rank that is
//! not exactly that is refused by name ([`PlacementError::ExpertRank`]): a
//! missing final newline (a cut file), a layer twice, an id twice or past
//! `n_expert`, a layer with other than `n_expert` ids. Nothing is padded or
//! defaulted.

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use super::{ExpertList, PlacementError};

/// The first field of a rank file's header line.
const HEADER: &str = "# router-rank";

/// Every routed layer's experts in router order, hottest first, and what the
/// file says about where the order came from.
#[derive(Clone, PartialEq, Eq)]
pub struct ExpertRank {
    path: String,
    n_expert: u64,
    sets: String,
    positions: u64,
    layers: BTreeMap<usize, Box<[u32]>>,
}

impl fmt::Debug for ExpertRank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExpertRank")
            .field("path", &self.path)
            .field("n_expert", &self.n_expert)
            .field("sets", &self.sets)
            .field("positions", &self.positions)
            .field("layers", &self.layers.len())
            .finish()
    }
}

impl ExpertRank {
    /// The rank in the file at `path`; refused by name when the file cannot
    /// be read or is not a rank ([`ExpertRank::parse`]).
    pub fn read(path: &Path) -> Result<ExpertRank, PlacementError> {
        let name = path.display().to_string();
        let text = std::fs::read_to_string(path).map_err(|e| PlacementError::ExpertRank {
            file: name.clone(),
            detail: format!("cannot be read: {e}"),
        })?;
        ExpertRank::parse(&name, &text)
    }

    /// The rank `text` holds, `file` its name in refusals.
    pub fn parse(file: &str, text: &str) -> Result<ExpertRank, PlacementError> {
        let refuse = |detail: String| PlacementError::ExpertRank {
            file: file.to_string(),
            detail,
        };
        let body = text
            .strip_suffix('\n')
            .ok_or_else(|| refuse("has no final newline: the file is cut short".into()))?;
        let mut lines = body.split('\n');
        let head = lines.next().unwrap_or_default();
        let mut fields = head.split('\t');
        if fields.next() != Some(HEADER) {
            return Err(refuse(format!(
                "line 1 is not a `{HEADER}` header: {head:?}"
            )));
        }
        let mut found: BTreeMap<&str, &str> = BTreeMap::new();
        for field in fields {
            let (key, value) = field
                .split_once('=')
                .ok_or_else(|| refuse(format!("header field {field:?} is not key=value")))?;
            if !matches!(key, "n_expert" | "n_used" | "sets" | "positions") {
                return Err(refuse(format!("header field {key} is not one a rank has")));
            }
            if found.insert(key, value).is_some() {
                return Err(refuse(format!("header field {key} is given twice")));
            }
        }
        let number = |key: &str| -> Result<u64, PlacementError> {
            let value = found
                .get(key)
                .ok_or_else(|| refuse(format!("the header has no {key}")))?;
            value
                .parse::<u64>()
                .map_err(|_| refuse(format!("header {key}={value} is not a count")))
        };
        let n_expert = number("n_expert")?;
        let n_used = number("n_used")?;
        let positions = number("positions")?;
        let sets = found
            .get("sets")
            .filter(|s| !s.is_empty())
            .ok_or_else(|| refuse("the header has no sets".into()))?
            .to_string();
        if n_expert == 0 || n_expert > u64::from(u32::MAX) || n_used == 0 || n_used > n_expert {
            return Err(refuse(format!(
                "header n_expert={n_expert} n_used={n_used}: an order over experts needs \
                 0 < n_used <= n_expert, ids fitting u32"
            )));
        }
        if positions == 0 {
            return Err(refuse(
                "header positions=0: an order learned from no position".into(),
            ));
        }
        let width =
            usize::try_from(n_expert).map_err(|_| refuse("n_expert passes usize".into()))?;
        let mut layers: BTreeMap<usize, Box<[u32]>> = BTreeMap::new();
        for (at, row) in lines.enumerate() {
            let line = at + 2;
            let (layer, ids) = row
                .split_once('\t')
                .ok_or_else(|| refuse(format!("line {line} is not <layer><TAB><ids>: {row:?}")))?;
            let layer: usize = layer
                .parse()
                .map_err(|_| refuse(format!("line {line}: layer {layer:?} is not a layer")))?;
            let mut order = Vec::with_capacity(width);
            let mut seen = vec![false; width];
            for id in ids.split(',') {
                let e: u32 = id.parse().map_err(|_| {
                    refuse(format!("line {line}: layer {layer} id {id:?} is not an id"))
                })?;
                let slot = seen.get_mut(e as usize).ok_or_else(|| {
                    refuse(format!(
                        "line {line}: layer {layer} id {e} is not below n_expert {n_expert}"
                    ))
                })?;
                if std::mem::replace(slot, true) {
                    return Err(refuse(format!(
                        "line {line}: layer {layer} lists id {e} twice"
                    )));
                }
                order.push(e);
            }
            if order.len() != width {
                return Err(refuse(format!(
                    "line {line}: layer {layer} lists {} ids, not n_expert {n_expert}",
                    order.len()
                )));
            }
            if layers.insert(layer, order.into_boxed_slice()).is_some() {
                return Err(refuse(format!("line {line}: layer {layer} is given twice")));
            }
        }
        if layers.is_empty() {
            return Err(refuse("holds no layer".into()));
        }
        Ok(ExpertRank {
            path: file.to_string(),
            n_expert,
            sets,
            positions,
            layers,
        })
    }

    /// The file the rank was read from, as the plan line names it.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The router sets the order was learned from, as the file's header names
    /// them.
    #[must_use]
    pub fn sets(&self) -> &str {
        &self.sets
    }

    /// The positions the order was learned over, all sets together.
    #[must_use]
    pub fn positions(&self) -> u64 {
        self.positions
    }

    /// The layers the rank orders, ascending.
    pub fn layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.layers.keys().copied()
    }

    /// `layer`'s experts, hottest first; `None` for a layer the rank does not
    /// order.
    #[must_use]
    pub fn ranked(&self, layer: usize) -> Option<&[u32]> {
        self.layers.get(&layer).map(|ids| &ids[..])
    }

    /// Refuse by name unless the rank orders exactly the routed layers the
    /// model has, each over the experts of its stacks: `routed` is each such
    /// layer with its stacks' expert count.
    pub(super) fn check(&self, routed: &[(usize, u64)]) -> Result<(), PlacementError> {
        let refuse = |detail: String| PlacementError::ExpertRank {
            file: self.path.clone(),
            detail,
        };
        for &(layer, experts) in routed {
            if self.ranked(layer).is_none() {
                return Err(refuse(format!(
                    "layer {layer} holds routed experts in the model and the rank orders none of \
                     its experts"
                )));
            }
            if experts != self.n_expert {
                return Err(refuse(format!(
                    "layer {layer} has {experts} experts in the model and the rank orders \
                     {} a layer",
                    self.n_expert
                )));
            }
        }
        if let Some(layer) = self
            .layers()
            .find(|l| !routed.iter().any(|&(r, _)| r == *l))
        {
            return Err(refuse(format!(
                "orders layer {layer}, which holds no routed experts in the model"
            )));
        }
        Ok(())
    }

    /// `layer`'s expert lists under the rank, each list ascending as the
    /// slot order needs it ([`ExpertList`]): the stage card's the layer's
    /// `stage` hottest ids, and one list per entry of `tiers` in tier order
    /// from the coldest end — tier 0 the coldest `tiers[0]` ids, tier 1 the
    /// next-coldest `tiers[1]`, and so on. The host holds the middle: the ids
    /// of no list. Refused by name when the rank does not order `layer` or the
    /// counts pass the layer's `experts`.
    pub(super) fn split(
        &self,
        layer: usize,
        experts: u64,
        stage: u64,
        tiers: &[u64],
    ) -> Result<(ExpertList, Vec<ExpertList>), PlacementError> {
        let refuse = |detail: String| PlacementError::ExpertRank {
            file: self.path.clone(),
            detail,
        };
        let ids = self
            .ranked(layer)
            .filter(|ids| ids.len() as u64 == experts)
            .ok_or_else(|| {
                refuse(format!(
                    "layer {layer} is not ordered over its {experts} experts"
                ))
            })?;
        let taken = tiers
            .iter()
            .try_fold(stage, |sum, &n| sum.checked_add(n))
            .filter(|&sum| sum <= experts)
            .ok_or_else(|| {
                refuse(format!(
                    "layer {layer}: the stage card's {stage} experts and the tiers' {tiers:?} pass \
                     its {experts}"
                ))
            })?;
        debug_assert!(taken <= experts);
        // `taken <= experts == ids.len()`, so every cut below is in range.
        let cut =
            |from: u64, to: u64| ExpertList::new(ids[from as usize..to as usize].to_vec(), experts);
        let stage_list = cut(0, stage)?;
        let mut end = experts;
        let mut tier_lists = Vec::with_capacity(tiers.len());
        for &own in tiers {
            tier_lists.push(cut(end - own, end)?);
            end -= own;
        }
        Ok((stage_list, tier_lists))
    }

    /// The layer's ids `of` — the host leg's, ascending — split by the rank
    /// at its cold end: the coldest `n` of them, which the NVMe expert tier
    /// pages ([`super::room::expert_nvme_tier`]), and the rest, which the
    /// host keeps; each list ascending as the slot order needs it
    /// ([`ExpertList`]). The counts are the caller's, so the bytes the split
    /// moves are the same whatever ids they are. Refused by name when the
    /// rank does not order `layer` over its `experts` experts, the refusal
    /// [`ExpertRank::split`] gives.
    pub(super) fn cold_split(
        &self,
        layer: usize,
        experts: u64,
        of: &[u32],
        n: u64,
    ) -> Result<(Vec<u32>, Vec<u32>), PlacementError> {
        let refuse = |detail: String| PlacementError::ExpertRank {
            file: self.path.clone(),
            detail,
        };
        let ids = self
            .ranked(layer)
            .filter(|ids| ids.len() as u64 == experts)
            .ok_or_else(|| {
                refuse(format!(
                    "layer {layer} is not ordered over its {experts} experts"
                ))
            })?;
        // `of` holds ids below `experts` ([`ExpertList`]) and the rank's row
        // every id of the layer once, so the row names every id of `of` and
        // the two lists below partition it.
        let mut paged = Vec::new();
        let mut kept = Vec::new();
        for &id in ids.iter().rev() {
            if of.binary_search(&id).is_err() {
                continue;
            }
            if u64::try_from(paged.len()).is_ok_and(|len| len < n) {
                paged.push(id);
            } else {
                kept.push(id);
            }
        }
        paged.sort_unstable();
        kept.sort_unstable();
        Ok((kept, paged))
    }
}
