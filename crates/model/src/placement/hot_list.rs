//! A hot list: per layer, routed expert ids in rank order, hottest first —
//! which experts a card keeps when the plan's count for the layer is `n_l`
//! (its first `n_l`). The plan decides how many, the file decides which.
//!
//! The file is text, written by `tools/ref/router-hotlist.py`: `# key<TAB>value`
//! header lines, then one line per layer, `layer<TAB>id,id,…`. `# n_expert`
//! must be the model's expert count and `# order` must be `rank`; every other
//! header line is provenance and is kept only for printing. Within a layer an
//! id is below `n_expert` and appears once; a layer appears at most once.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::OnceLock;

use super::{ExpertList, ModelTensors, PlacementError};

/// The lever: a path to a hot list, read once per process.
pub const LEVER: &str = "BLOOMERY_HOT_LIST";

/// A parsed hot list file.
#[derive(Clone, Debug)]
pub struct HotList {
    path: String,
    n_expert: u64,
    /// Per layer, its ids in rank order.
    layers: BTreeMap<usize, Vec<u32>>,
    /// The header lines other than `n_expert` and `order`, in file order.
    provenance: Vec<(String, String)>,
}

impl HotList {
    /// The list `BLOOMERY_HOT_LIST` names, read and checked on the first
    /// call of the process; `None` when the variable is not set. Held in a
    /// `OnceLock` because it is a lever: every plan of the process places by
    /// the same file.
    pub fn from_env() -> Result<Option<&'static HotList>, PlacementError> {
        static READ: OnceLock<Result<Option<HotList>, (String, String)>> = OnceLock::new();
        let read = READ.get_or_init(|| match std::env::var(LEVER) {
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(e) => Err((LEVER.to_string(), e.to_string())),
            Ok(path) => HotList::parse_file(&path).map(Some),
        });
        match read {
            Ok(list) => Ok(list.as_ref()),
            Err((path, detail)) => Err(PlacementError::HotList {
                path: path.clone(),
                detail: detail.clone(),
            }),
        }
    }

    /// Read and check the file at `path`.
    pub fn read(path: &Path) -> Result<HotList, PlacementError> {
        let p = path.display().to_string();
        HotList::parse_file(&p).map_err(|(path, detail)| PlacementError::HotList { path, detail })
    }

    fn parse_file(path: &str) -> Result<HotList, (String, String)> {
        let refuse = |detail: String| (path.to_string(), detail);
        if path.is_empty() {
            return Err(refuse("an empty path".to_string()));
        }
        let text = std::fs::read_to_string(path).map_err(|e| refuse(e.to_string()))?;
        HotList::parse(path, &text).map_err(refuse)
    }

    /// Parse the text of the file at `path`; the error is the first line it
    /// refuses and why.
    pub fn parse(path: &str, text: &str) -> Result<HotList, String> {
        let (mut n_expert, mut order) = (None, None);
        let mut provenance = Vec::new();
        let mut layers = BTreeMap::new();
        for (no, line) in text.lines().enumerate() {
            let at = |detail: String| format!("line {}: {detail}", no + 1);
            if line.trim().is_empty() {
                continue;
            }
            if let Some(h) = line.strip_prefix('#') {
                let (key, value) = h.trim_start().split_once('\t').unwrap_or((h.trim(), ""));
                match key {
                    "n_expert" => {
                        let n = value.trim().parse::<u64>();
                        n_expert = Some(n.map_err(|e| at(format!("n_expert {value:?}: {e}")))?);
                    }
                    "order" => order = Some(value.trim().to_string()),
                    _ => provenance.push((key.to_string(), value.to_string())),
                }
                continue;
            }
            let n_expert =
                n_expert.ok_or_else(|| at("a layer line before `# n_expert`".to_string()))?;
            let (layer, ids) = line
                .split_once('\t')
                .ok_or_else(|| at("not `layer<TAB>ids`".to_string()))?;
            let layer = layer
                .trim()
                .parse::<usize>()
                .map_err(|e| at(format!("layer {layer:?}: {e}")))?;
            let ids = ids
                .split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| {
                    s.trim()
                        .parse::<u32>()
                        .map_err(|e| format!("id {s:?}: {e}"))
                })
                .collect::<Result<Vec<u32>, String>>()
                .map_err(at)?;
            ExpertList::new(ids.clone(), n_expert)
                .map_err(|e| at(format!("layer {layer}: {e}")))?;
            if layers.insert(layer, ids).is_some() {
                return Err(at(format!("layer {layer} appears twice")));
            }
        }
        let n_expert = n_expert.ok_or("no `# n_expert` line")?;
        match order.as_deref() {
            Some("rank") => {}
            other => return Err(format!("`# order` is {other:?}, not \"rank\"")),
        }
        Ok(HotList {
            path: path.to_string(),
            n_expert,
            layers,
            provenance,
        })
    }

    /// Refused unless the list is a list of `model`: the same expert count,
    /// and no layer past the model's.
    pub fn check_model(&self, model: &ModelTensors) -> Result<(), PlacementError> {
        if self.n_expert != model.experts {
            return Err(self.refuse(format!(
                "n_expert {} is not the model's {}",
                self.n_expert, model.experts
            )));
        }
        if let Some((&l, _)) = self.layers.range(model.layers..).next() {
            return Err(self.refuse(format!("layer {l} is past the model's {}", model.layers)));
        }
        Ok(())
    }

    /// The experts layer `layer`'s card keeps when the plan's count is `n`:
    /// the layer's first `n` ids, as a list of a stack of `experts`. Refused
    /// when the file holds fewer than `n` for the layer.
    pub fn card_list(
        &self,
        layer: usize,
        n: u64,
        experts: u64,
    ) -> Result<ExpertList, PlacementError> {
        let ids = self.layers.get(&layer).map_or(&[][..], Vec::as_slice);
        let take = usize::try_from(n)
            .ok()
            .filter(|&n| n <= ids.len())
            .ok_or_else(|| {
                self.refuse(format!(
                    "layer {layer} lists {} experts, the plan keeps {n} on its card",
                    ids.len()
                ))
            })?;
        ExpertList::new(ids[..take].to_vec(), experts)
    }

    /// The file's path.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The header lines other than `n_expert` and `order`.
    #[must_use]
    pub fn provenance(&self) -> &[(String, String)] {
        &self.provenance
    }

    /// Layer `layer`'s ids in rank order; empty for a layer the file lacks.
    #[must_use]
    pub fn ranked(&self, layer: usize) -> &[u32] {
        self.layers.get(&layer).map_or(&[], Vec::as_slice)
    }

    fn refuse(&self, detail: String) -> PlacementError {
        PlacementError::HotList {
            path: self.path.clone(),
            detail,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::ExpertList;
    use super::HotList;

    const TEXT: &str =
        "# router-hotlist\n# sets\ta,b\n# n_expert\t8\n# order\trank\n2\t5,1,7\n3\t0,4\n";

    /// A layer's first `n` ranked ids become its sorted card list; asking for
    /// more than the layer lists, or a layer the file lacks with n > 0, is
    /// refused.
    #[test]
    fn card_list_takes_the_ranked_head() {
        let h = HotList::parse("t", TEXT).expect("the text is a hot list");
        assert_eq!(h.card_list(2, 2, 8).expect("2 of 3").ids(), &[1, 5]);
        assert_eq!(h.card_list(3, 0, 8).expect("none").len(), 0);
        assert!(h.card_list(2, 4, 8).is_err());
        assert!(h.card_list(9, 1, 8).is_err());
        assert_eq!(h.ranked(2), &[5, 1, 7]);
    }

    /// The parser refuses a repeated id, an id past n_expert, a repeated
    /// layer, and a file without `# order rank`.
    #[test]
    fn parse_refuses_what_is_not_a_hot_list() {
        let head = "# n_expert\t8\n# order\trank\n";
        for bad in ["2\t1,1\n", "2\t8\n", "2\t1\n2\t3\n"] {
            assert!(
                HotList::parse("t", &format!("{head}{bad}")).is_err(),
                "{bad:?}"
            );
        }
        assert!(HotList::parse("t", "# n_expert\t8\n2\t1\n").is_err());
    }

    /// The list type itself: sorted on construction, duplicates and ids past
    /// the stack refused, the complement and runs of a scattered list, and a
    /// prefix recognised as one.
    #[test]
    fn expert_list_shapes() {
        let l = ExpertList::new(vec![6, 1, 2], 8).expect("a list");
        assert_eq!(l.ids(), &[1, 2, 6]);
        assert_eq!(l.runs(), vec![1..3, 6..7]);
        assert_eq!(l.to_string(), "1..3,6..7");
        assert_eq!(l.complement(8).expect("fits").ids(), &[0, 3, 4, 5, 7]);
        assert_eq!(l.slot_of(6), Some(2));
        assert_eq!(l.as_prefix(), None);
        assert!(ExpertList::new(vec![1, 1], 8).is_err());
        assert!(ExpertList::new(vec![8], 8).is_err());
        let p = ExpertList::prefix(3).expect("a prefix");
        assert_eq!(
            (p.as_prefix(), p.to_string()),
            (Some(3), "0..3".to_string())
        );
    }
}
