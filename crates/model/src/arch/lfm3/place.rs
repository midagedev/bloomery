//! An `lfm3` file's placement inputs: the family's half of the path from the file's headers to a
//! plan. The tensors' roles, the typed description and each layer's store bytes are read once
//! ([`describe`]; [`read`] refuses a file whose features no program runs, every one listed). No
//! program runs the description yet, so [`read`] refuses every file by the coverage list; the plan
//! itself comes with the program.
//!
//! A sequence's stores are a short-conv layer's ring of the last positions' inputs, which does not
//! grow with the context; a full attention layer's K and V planes, which do; and a sliding layer's
//! ring of rows ([`runtime::stores::ring_rows`]), which is the window and a prompt batch, never
//! past the context.

use gguf::Split;
use models::{Mixer, ModelSpec, RopeMode};
use runtime::stores::{self, Rows};

use super::spec;
use crate::arch::coverage;
use crate::placement::{KvBytes, ModelTensors, PlacementError, Unimplemented};

/// What a plan of an `lfm3` file is made from: every tensor with its role, the typed description
/// the coverage check reads, and each layer's store bytes.
#[derive(Clone, Debug)]
pub struct Inputs {
    /// Every tensor with its role, and the model's layer count.
    pub model: ModelTensors,
    /// The typed model description.
    pub spec: ModelSpec,
    /// Each layer's store bytes.
    pub kv: KvLayout,
}

impl Inputs {
    /// `self`, or, for a file with a feature the engine does not run,
    /// [`PlacementError::Unimplemented`] with every one listed.
    pub fn runnable(self) -> Result<Inputs, PlacementError> {
        let missing = self.unimplemented();
        if missing.is_empty() {
            Ok(self)
        } else {
            Err(PlacementError::Unimplemented(missing))
        }
    }

    /// Every feature of the file the engine does not run, as [`coverage::check`] lists them.
    pub fn unimplemented(&self) -> Vec<Unimplemented> {
        coverage::check(&self.spec, &self.model)
    }
}

/// `split`'s tensors' roles, then its description, the first that fails being the error; then a
/// file with a feature the engine does not run ([`PlacementError::Unimplemented`], every one
/// listed).
pub fn read(split: &Split) -> Result<Inputs, PlacementError> {
    describe(split)?.runnable()
}

/// [`read`] without the refusals: the file as it is, for a gate that describes a file whatever the
/// coverage list holds.
pub fn describe(split: &Split) -> Result<Inputs, PlacementError> {
    let read = spec::read(split)?;
    let kv = KvLayout::of_spec(&read.spec)?;
    Ok(Inputs {
        model: read.tensors,
        spec: read.spec,
        kv,
    })
}

/// What one trunk layer holds per sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Store {
    /// A short-conv layer: a ring of this many bytes, whatever the context.
    Ring(u64),
    /// An attention layer: K and V of `row` bytes a position over its `window` positions when it
    /// has one.
    Attention { row: u64, window: Option<u64> },
}

/// One file's per-layer stores, read off its description: [`KvBytes`] for the plan.
#[derive(Clone, Debug)]
pub struct KvLayout {
    /// Per trunk layer.
    layers: Vec<Store>,
    /// Which row count a window layer uses.
    rows: Rows,
    /// The rope rows the body copies to its card, in bytes a position: one table of `dims` f32
    /// values a position for each distinct rope (base and dims) the attention layers turn by.
    rope_row: u64,
}

impl KvLayout {
    /// The layout `spec` describes: each short-conv layer's ring
    /// ([`stores::conv_ring_rows`] rows of the hidden width, f32) and each attention layer's rows
    /// of its key heads' key and value values in f16, a sliding layer's on the ring
    /// ([`Rows::Ring`]).
    ///
    /// # Errors
    ///
    /// A layer whose stores this layout does not size, by layer
    /// ([`PlacementError::Unimplemented`]): a mixer other than a short conv or a GQA layer, and a
    /// GQA layer that selects positions.
    pub fn of_spec(spec: &ModelSpec) -> Result<KvLayout, PlacementError> {
        let mut layers = Vec::with_capacity(spec.layers.len());
        let mut ropes: Vec<(u32, u32)> = Vec::new();
        for (layer, l) in spec.layers.iter().enumerate() {
            let refuse = |what: &str| {
                PlacementError::Unimplemented(vec![Unimplemented {
                    layer: Some(layer),
                    feature: format!(
                        "{what}: this plan sizes the stores of a short-conv layer and a plain GQA \
                         layer"
                    ),
                }])
            };
            layers.push(match &l.mixer {
                Mixer::ShortConv(c) => {
                    let rows = stores::conv_ring_rows(c.taps as usize) as u64;
                    Store::Ring(rows * u64::from(spec.hidden) * stores::F32_BYTES)
                }
                Mixer::Gqa(g) if g.select.is_none() => {
                    if g.rope.mode != RopeMode::None {
                        let rope = (g.rope.base.to_bits(), g.rope.dims);
                        if !ropes.contains(&rope) {
                            ropes.push(rope);
                        }
                    }
                    Store::Attention {
                        row: u64::from(g.kv_heads)
                            * (u64::from(g.head_dim) + u64::from(g.value_dim))
                            * stores::F16_BYTES,
                        window: g.window.map(u64::from),
                    }
                }
                Mixer::Gqa(_) => return Err(refuse("a GQA mixer that selects positions")),
                other => return Err(refuse(other.what())),
            });
        }
        Ok(KvLayout {
            layers,
            rows: Rows::Ring,
            rope_row: ropes
                .iter()
                .map(|&(_, dims)| u64::from(dims) * stores::F32_BYTES)
                .sum(),
        })
    }

    /// The same layout with the window layers on `rows`.
    #[must_use]
    pub fn with_rows(self, rows: Rows) -> KvLayout {
        KvLayout { rows, ..self }
    }
}

impl KvBytes for KvLayout {
    /// A short-conv layer's ring, an attention layer's rows of `ctx_max` positions or of the
    /// window's ring; 0 past the last layer.
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        match self.layers.get(layer) {
            Some(Store::Ring(bytes)) => *bytes,
            Some(Store::Attention { row, window }) => {
                self.rows.count(*window, ctx_max).saturating_mul(*row)
            }
            None => 0,
        }
    }

    fn stage_position_bytes(&self, ctx_max: u64) -> u64 {
        self.rope_row.saturating_mul(ctx_max)
    }
}

#[cfg(test)]
mod tests {
    use runtime::stores::{Rows, ring_rows};

    use super::super::spec::tests::{EMBD, HEAD, KINDS, KV, TAPS, WINDOW, keys, load, tensors};
    use super::{KvLayout, describe};
    use crate::arch::synthetic::{V, header_shaped};
    use crate::placement::{KvBytes, PlacementError};
    use gguf::Split;

    fn inputs(tag: &str) -> super::Inputs {
        let path = header_shaped(
            tag,
            "lfm3",
            &keys(),
            &[("tokenizer.ggml.pre", V::Str("lfm2"))],
            &tensors(),
        );
        let split = Split::open(&path).expect("the synthetic header opens");
        let inputs = describe(&split).expect("the synthetic header describes");
        let _ = std::fs::remove_file(&path);
        inputs
    }

    /// A short-conv ring's bytes: the rows of a pass beside the filter's reach, of the hidden
    /// width, in f32.
    fn ring_bytes() -> u64 {
        (TAPS - 1 + 8) * EMBD * 4
    }

    /// One position's K and V of the synthetic attention layers, f16.
    fn row() -> u64 {
        2 * u64::from(KV) * HEAD * 2
    }

    /// Each kind holds its own bytes: a short-conv layer's ring at every context, a full layer's
    /// plane of every position, a sliding layer's ring of rows that is the plane while the context
    /// is the shorter, and none past the last layer. Mutant: the ring sized by the taps alone, or
    /// the sliding layer on a plane.
    #[test]
    fn each_kind_holds_its_own_bytes() {
        let kv = inputs("lfm3-place-bytes").kv;
        assert_eq!(row(), 64);
        assert_eq!(ring_bytes(), 1792);
        let ring = ring_rows(u64::from(WINDOW), 4096);
        assert_eq!(ring, 768);
        for (layer, kind) in KINDS.chars().enumerate() {
            for ctx in [1, 9, 767, 768, 4096] {
                let want = match kind {
                    'c' => ring_bytes(),
                    'f' => row() * ctx,
                    's' => row() * ring_rows(u64::from(WINDOW), ctx),
                    _ => unreachable!(),
                };
                assert_eq!(kv.layer_bytes(layer, ctx), want, "layer {layer} at {ctx}");
            }
        }
        assert_eq!(kv.layer_bytes(KINDS.len(), 9), 0);
        let plane = kv.clone().with_rows(Rows::Plane);
        assert_eq!(plane.layer_bytes(4, 4096), row() * 4096);
        assert_eq!(plane.layer_bytes(0, 4096), ring_bytes());
    }

    /// The rope tables the body copies are one for each distinct rope the attention layers turn
    /// by: the full layers' θ and the sliding layers' are two, of `dims` f32 values a position.
    /// A layout that counts one, or none, fails this.
    #[test]
    fn two_ropes_are_two_tables() {
        let kv = inputs("lfm3-place-rope").kv;
        assert_eq!(kv.stage_position_bytes(10), 2 * HEAD * 4 * 10);
    }

    /// A mixer this layout does not size is refused by name with its layer: a latent layer, a
    /// GQA layer that selects.
    #[test]
    fn a_mixer_the_layout_does_not_size_is_refused() {
        let mut spec = inputs("lfm3-place-refuse").spec;
        spec.layers[1].mixer = models::Mixer::None;
        match KvLayout::of_spec(&spec) {
            Err(PlacementError::Unimplemented(items)) => {
                assert_eq!(items.len(), 1);
                assert_eq!(items[0].layer, Some(1));
                assert!(items[0].feature.starts_with("no token mixer"), "{items:?}");
            }
            other => panic!("{other:?}, not the layer's refusal"),
        }
    }

    /// No program runs the file: `read` lists what no program runs, and `describe` is the file as
    /// it is. The read of a file with a part no row covers is refused by name.
    #[test]
    fn no_program_runs_the_description_yet() {
        let inputs = inputs("lfm3-place-runnable");
        let listed = inputs.unimplemented();
        assert!(
            listed
                .iter()
                .any(|u| u.feature == "a layer program for lfm3"),
            "{listed:?}"
        );
        match inputs.runnable() {
            Err(PlacementError::Unimplemented(items)) => assert_eq!(items, listed),
            other => panic!("{other:?}, not the coverage list"),
        }
        assert!(load("lfm3-place-registry", keys(), &tensors()).is_ok());
    }
}
