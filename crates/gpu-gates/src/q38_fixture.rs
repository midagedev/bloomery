//! What the three Qwen3.8 whole-model gates (`gate_qwen4exp_e2e`, `gate_qwen4exp_mtp`,
//! `gate_qwen38_twocard`) read from the file they open instead of carrying the real file's
//! numbers: its layer kinds from the file's description and the captured graphs' node counts from
//! those kinds. The tier the gates run in, the card a plan is made on and the plan's card budget
//! belong to `crate::tier`; a gate prints each derived value beside the literal it replaced there
//! (`tier::witness`).

use crate::GateError;

/// The live raw keys of Qwen3.8's unfinished pool, or None for a malformed ring.
#[cfg(feature = "gpu")]
pub fn key_ring_live(ring: &[u16], pos: usize) -> Option<Vec<u16>> {
    use bloomery_gpu::qsa::{DIM, POOL, RING_ROWS};
    (ring.len() == RING_ROWS * DIM).then(|| {
        (pos / POOL * POOL..pos)
            .flat_map(|p| {
                let s = p % RING_ROWS * DIM;
                ring[s..s + DIM].iter().copied()
            })
            .collect()
    })
}

/// Launches of the embedding row.
const EMBED_LAUNCHES: usize = 1;
/// Launches of a delta layer: its two mixes' 3 each, the mixer's 7 (q·k·v, z, β·α, the conv, the
/// delta step, the gated norm, the output projection) and the block's 9 (the router, the handoff,
/// the go, the shared expert's four, the wait, the gated sum).
const GDN_LAYER_LAUNCHES: usize = 6 + 7 + 9;
/// Launches of a selecting layer: the same 6 and 9 around the mixer's 13.
/// PIN(2026-10-30): the selected flash's merge applies the out gate, so the mixer runs 13 launches
/// (14 before).
const QSA_LAYER_LAUNCHES: usize = 6 + 13 + 9;
/// Launches of the PLE site: the combine, key and value, the gate and the conv.
const PLE_SITE_LAUNCHES: usize = 5;
/// Launches of the head: its mix's three, the gemv and the argmax; a K-quant `output` quantizes
/// its input rows to q8_1 first, and a Q5_K one copies its token-major logits past one row
/// ([`Shape::head`]), one more each.
const HEAD_LAUNCHES: usize = 5;
/// Nodes a verify's PLE rows' arrival adds at the PLE site: the wait on the host's flag (a batch
/// memory operation) and the copy of the rows from the host's staging ([`Shape::arrivals`]).
const ARRIVAL_NODES: usize = 2;
/// Launches the card leg adds on a card layer: the normed rows' q8_1, the gate·up, the card
/// slots' q8_1, the down and the card sum.
const CARD_LEG_LAUNCHES: usize = 5;

/// The layer kinds a file's description names: what a gate's structure clause is written against,
/// and the only source it reads them from (never the loaded body).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    /// The trunk's layers.
    pub n_layer: usize,
    /// The selecting layers (the description's attention layers), in order.
    pub qsa: Vec<usize>,
    /// `ple.layers`: the layer the PLE site sits on; `None` when the file lists no site.
    pub ple_layer: Option<usize>,
    /// `ple.image_token_id`: the placeholder the file's PLE hash refuses as an input.
    pub image_token: Option<u32>,
    /// The launches the head's `output` type adds to a captured walk: the one tensor type, with
    /// the dense sites' ([`Shape::by_name`]), the node count depends on. [`Shape::with_head`]
    /// states it, which the header's values alone do not.
    pub head: Head,
    /// The launches the dense sites' q8_1 projections add to a captured walk (a K-quant site
    /// quantizes its input rows first, and past one row a row-major type copies its result
    /// token-major): at one row and at more than one. [`Shape::with_by_name`] states them, which
    /// the header's values alone do not; zero on a file whose dense sites read the f32 rows or q8
    /// blocks of 32.
    pub by_name: ByName,
}

/// The extra launches of a file's by-name dense projections ([`Shape::by_name`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ByName {
    /// A captured step's, over every layer.
    pub one: usize,
    /// A captured verify's at two to four rows, over every layer.
    pub rows: usize,
}

/// The extra launches of a file's head over a projection of f32 rows ([`Shape::head`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Head {
    /// A K-quant `output` quantizes its input rows to q8_1 first: one more launch at any row count.
    pub quantizes: bool,
    /// A Q5_K `output` writes token-major logits, which a copy interleaves past one row: one more
    /// launch at more than one row.
    pub copies_rows: bool,
}

impl Shape {
    /// The shape of a description's layers: `n_layer` of them, `qsa` of which select.
    ///
    /// # Errors
    /// A file with no layer, a selecting layer past the layers or listed out of order, or a PLE
    /// site past the layers.
    pub fn from_layers(
        n_layer: usize,
        qsa: Vec<usize>,
        ple_layer: Option<usize>,
        image_token: Option<u32>,
    ) -> Result<Shape, GateError> {
        if n_layer == 0 {
            return Err("the description holds no layer".into());
        }
        if qsa.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(format!("the selecting layers {qsa:?} are not in order").into());
        }
        if qsa.last().is_some_and(|&l| l >= n_layer) {
            return Err(
                format!("the selecting layers {qsa:?} are past the {n_layer} layers").into(),
            );
        }
        if ple_layer.is_some_and(|l| l >= n_layer) {
            return Err(format!("the PLE site {ple_layer:?} is past the {n_layer} layers").into());
        }
        Ok(Shape {
            n_layer,
            qsa,
            ple_layer,
            image_token,
            head: Head::default(),
            by_name: ByName::default(),
        })
    }

    /// `self` for a file whose dense sites add `by_name` launches.
    #[must_use]
    pub fn with_by_name(self, by_name: ByName) -> Shape {
        Shape { by_name, ..self }
    }

    /// `self` for a file whose head adds `head` launches.
    #[must_use]
    pub fn with_head(self, head: Head) -> Shape {
        Shape { head, ..self }
    }

    /// Selecting layers.
    #[must_use]
    pub fn n_qsa(&self) -> usize {
        self.qsa.len()
    }

    /// Delta layers.
    #[must_use]
    pub fn n_gdn(&self) -> usize {
        self.n_layer - self.qsa.len()
    }

    /// The captured decode step's nodes: the embedding row, each layer's launches, the PLE site's
    /// and the head's.
    #[must_use]
    pub fn nodes_decode(&self) -> usize {
        EMBED_LAUNCHES
            + self.n_gdn() * GDN_LAYER_LAUNCHES
            + self.n_qsa() * QSA_LAYER_LAUNCHES
            + self.ple_layer.map_or(0, |_| PLE_SITE_LAUNCHES)
            + HEAD_LAUNCHES
            + usize::from(self.head.quantizes)
            + self.by_name.one
    }

    /// The stream memory-operation batches of a captured step: each layer's go and wait.
    #[must_use]
    pub fn memops(&self) -> usize {
        2 * self.n_layer
    }

    /// The stream memory-operation batches of a captured step or verify under a card plan that
    /// puts every routed expert of `hostless` layers on the card, on a load with no residency
    /// machine and no route trace: each other layer's go and wait (the `hostless` layers run
    /// without the host leg).
    #[must_use]
    pub fn memops_card(&self, hostless: usize) -> usize {
        assert!(
            hostless <= self.n_layer,
            "{hostless} hostless layers of {} layers",
            self.n_layer
        );
        2 * (self.n_layer - hostless)
    }

    /// The captured verify's nodes at 2, 3 and 4 rows: a layer's launches are the step's (a
    /// delta layer's joined β·α projection is one launch at every row count), past the by-name
    /// launches of more than one row, the head's row copy and the PLE rows' arrival.
    #[must_use]
    pub fn nodes_verify(&self) -> usize {
        self.nodes_decode() - self.by_name.one
            + self.by_name.rows
            + usize::from(self.head.copies_rows)
            + self.arrivals() * ARRIVAL_NODES
    }

    /// The PLE rows' arrivals of a captured verify: one on a file with a PLE site. Each is one
    /// batch memory operation and one copy among the verify's nodes.
    #[must_use]
    pub fn arrivals(&self) -> usize {
        usize::from(self.ple_layer.is_some())
    }

    /// The nodes one more busy slot adds to a captured pass of several slots' rows over an f16
    /// cache: the embedding row, the PLE conv and the parked rows' copy. Every layer's other
    /// launches bound to a sequence (a delta layer's conv and delta step, a selecting layer's
    /// pool, key append, selection, q/k append and selected flash) and the indexer key's
    /// projection run once over every slot's rows.
    #[must_use]
    pub fn nodes_slot(&self) -> usize {
        EMBED_LAUNCHES + usize::from(self.ple_layer.is_some()) + 1
    }

    /// The captured step's nodes under a card plan that puts routed experts on `card_layers`
    /// layers.
    #[must_use]
    pub fn nodes_decode_card(&self, card_layers: usize) -> usize {
        self.nodes_decode() + CARD_LEG_LAUNCHES * card_layers
    }

    /// The captured verify's nodes under that plan.
    #[must_use]
    pub fn nodes_verify_card(&self, card_layers: usize) -> usize {
        self.nodes_verify() + CARD_LEG_LAUNCHES * card_layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real file's description (48 layers, every fourth one selecting, the PLE site on layer 1)
    /// gives the literals the gates carried; the fixture's (4 layers, every second one selecting,
    /// the site on layer 2) gives the counts its derivation names. Mutant: any launch constant or
    /// a count off by one.
    #[test]
    fn the_description_gives_the_real_files_literals_and_the_fixtures_counts() {
        let every_fourth = (0..12).map(|i| 4 * i + 3).collect();
        let real = Shape::from_layers(48, every_fourth, Some(1), Some(248_056)).expect("real");
        assert_eq!((real.n_qsa(), real.n_gdn()), (12, 36));
        assert_eq!(real.nodes_decode(), 1139);
        assert_eq!(real.memops(), 96);
        assert_eq!(real.nodes_verify(), 1141);
        assert_eq!(real.nodes_slot(), 3);
        assert_eq!(real.nodes_decode_card(48), 1379);
        assert_eq!(real.nodes_verify_card(48), 1381);
        assert_eq!((real.memops_card(0), real.memops_card(48)), (96, 0));
        let fix = Shape::from_layers(4, vec![1, 3], Some(2), Some(248_056)).expect("fixture");
        assert_eq!((fix.n_qsa(), fix.n_gdn()), (2, 2));
        assert_eq!(fix.nodes_decode(), 1 + 2 * 22 + 2 * 28 + 5 + 5);
        assert_eq!(fix.nodes_decode(), 111);
        assert_eq!(fix.nodes_verify(), 113);
        assert_eq!(fix.nodes_slot(), 3);
        assert_eq!(fix.nodes_decode_card(4), 131);
        assert_eq!(fix.nodes_verify_card(4), 133);
        let kq = fix.clone().with_head(Head {
            quantizes: true,
            copies_rows: false,
        });
        assert_eq!(kq.nodes_decode(), 112);
        assert_eq!(kq.nodes_verify(), 114);
        assert_eq!(kq.nodes_decode_card(4), 132);
        assert_eq!(kq.nodes_slot(), fix.nodes_slot());
        // A Q5_K head also copies its token-major logits past one row: the decode step at one
        // row is the K-quant head's, the verify one launch more.
        let q5 = fix.clone().with_head(Head {
            quantizes: true,
            copies_rows: true,
        });
        assert_eq!(q5.nodes_decode(), 112);
        assert_eq!(q5.nodes_verify(), 115);
        assert_eq!(q5.nodes_verify_card(4), 135);
        assert_eq!(q5.nodes_slot(), fix.nodes_slot());
        // Dense sites that read q8_1 rows: a delta layer's q·k·v, z and output and the shared
        // expert's gate and up (five sites) and a selecting layer's six and the same two (eight)
        // quantize once a launch; past one row Q4_K and Q6_K sites copy token-major too.
        let by = fix.with_by_name(ByName {
            one: 2 * 5 + 2 * 8,
            rows: 2 * 5 + 2 * 8 + 3 + 4,
        });
        assert_eq!(by.nodes_decode(), 111 + 26);
        assert_eq!(by.nodes_verify(), 113 + 26 + 7);
        assert_eq!(by.nodes_slot(), 3);
        assert_eq!(by.nodes_decode_card(4), 131 + 26);
    }

    /// A description no gate can read a shape from is a named error, and a file with no PLE site
    /// has no site's launches.
    #[test]
    fn a_description_without_a_shape_is_refused_and_no_site_adds_no_launches() {
        assert!(Shape::from_layers(0, Vec::new(), None, None).is_err());
        assert!(Shape::from_layers(4, vec![1, 4], None, None).is_err());
        assert!(Shape::from_layers(4, vec![3, 1], None, None).is_err());
        assert!(Shape::from_layers(4, vec![1, 3], Some(4), None).is_err());
        let bare = Shape::from_layers(4, vec![1, 3], None, None).expect("no site");
        assert_eq!(bare.nodes_decode(), 106);
        assert_eq!(bare.nodes_slot(), 2);
        assert_eq!((bare.nodes_verify(), bare.arrivals()), (106, 0));
    }
}
