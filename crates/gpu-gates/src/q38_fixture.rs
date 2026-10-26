//! What the three Qwen3.8 whole-model gates (`gate_qwen4exp_e2e`, `gate_qwen4exp_mtp`,
//! `gate_qwen38_twocard`) read from the file they open instead of carrying the real file's
//! numbers: its layer kinds from the file's description and the captured graphs' node counts from
//! those kinds. The tier the gates run in, the card a plan is made on and the plan's card budget
//! belong to `crate::tier`; a gate prints each derived value beside the literal it replaced there
//! (`tier::witness`).

use crate::GateError;

/// Launches of the embedding row.
const EMBED_LAUNCHES: usize = 1;
/// Launches of a delta layer: its two mixes' 3 each, the mixer's 7 (q·k·v, z, β·α, the conv, the
/// delta step, the gated norm, the output projection) and the block's 9 (the router, the handoff,
/// the go, the shared expert's four, the wait, the gated sum).
const GDN_LAYER_LAUNCHES: usize = 6 + 7 + 9;
/// Launches of a selecting layer: the same 6 and 9 around the mixer's 14.
const QSA_LAYER_LAUNCHES: usize = 6 + 14 + 9;
/// Launches of the PLE site: the combine, key and value, the gate and the conv.
const PLE_SITE_LAUNCHES: usize = 5;
/// Launches of the head: its mix's three, the q8_0 gemv and the argmax.
const HEAD_LAUNCHES: usize = 5;
/// Nodes a verify of more than one row adds on a delta layer: the joined β·α projection runs as
/// β's rows and α's rows, one launch each. A selecting layer adds none.
/// PIN(2026-10-22): the token-major copies and the indexer's extra launch are gone, so a delta layer
/// adds one launch (two before) and a selecting layer none (one before).
const VERIFY_GDN_EXTRA: usize = 1;
/// Launches bound to one sequence on a selecting layer, and on a delta layer (its conv and delta
/// step); a captured pass of several slots' rows repeats them for each busy slot.
const SLOT_QSA_LAUNCHES: usize = 8;
const SLOT_GDN_LAUNCHES: usize = 2;
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
        })
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

    /// The captured verify's nodes at 2, 3 and 4 rows.
    #[must_use]
    pub fn nodes_verify(&self) -> usize {
        self.nodes_decode() + self.n_gdn() * VERIFY_GDN_EXTRA
    }

    /// The nodes one more busy slot adds to a captured pass of several slots' rows: the
    /// embedding row, the PLE conv, each layer's sequence-bound launches and the parked rows'
    /// copy.
    #[must_use]
    pub fn nodes_slot(&self) -> usize {
        EMBED_LAUNCHES
            + usize::from(self.ple_layer.is_some())
            + self.n_gdn() * SLOT_GDN_LAUNCHES
            + self.n_qsa() * SLOT_QSA_LAUNCHES
            + 1
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
        assert_eq!(real.nodes_decode(), 1151);
        assert_eq!(real.memops(), 96);
        assert_eq!(real.nodes_verify(), 1187);
        assert_eq!(real.nodes_slot(), 171);
        assert_eq!(real.nodes_decode_card(48), 1391);
        assert_eq!(real.nodes_verify_card(48), 1427);
        assert_eq!((real.memops_card(0), real.memops_card(48)), (96, 0));
        let fix = Shape::from_layers(4, vec![1, 3], Some(2), Some(248_056)).expect("fixture");
        assert_eq!((fix.n_qsa(), fix.n_gdn()), (2, 2));
        assert_eq!(fix.nodes_decode(), 1 + 2 * 22 + 2 * 29 + 5 + 5);
        assert_eq!(fix.nodes_decode(), 113);
        assert_eq!(fix.nodes_verify(), 115);
        assert_eq!(fix.nodes_slot(), 23);
        assert_eq!(fix.nodes_decode_card(4), 133);
        assert_eq!(fix.nodes_verify_card(4), 135);
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
        assert_eq!(bare.nodes_decode(), 108);
        assert_eq!(bare.nodes_slot(), 22);
    }
}
