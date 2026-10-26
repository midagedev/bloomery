//! The one piece of `q38_fixture` that reads the Qwen3.8 family's layer kinds off the file's
//! description. It sits beside the gates that read it; the library module keeps the family-free
//! formulas and the tier helpers.

use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::q38_fixture::Shape;
use model::arch::models::{Extra, Mixer, ModelSpec, NgramRule};

/// The shape of a file's description: its layers, the attention layers among them, the PLE
/// site's layer and the image id the site hashes.
///
/// # Errors
/// A file with no layer or no attention layer, or a PLE site past the layers.
pub fn shape_of(spec: &ModelSpec) -> Result<Shape, GateError> {
    let attention: Vec<usize> = spec
        .layers
        .iter()
        .enumerate()
        .filter(|(_, l)| matches!(l.mixer, Mixer::Gqa(_)))
        .map(|(layer, _)| layer)
        .collect();
    if attention.is_empty() {
        return Err("the description holds no attention layer".into());
    }
    let ple_layer = spec
        .layers
        .iter()
        .position(|l| l.extras.contains(&Extra::Ple));
    let image = spec.engram.and_then(|e| match e.rule {
        NgramRule::Ple { image, .. } => image,
        NgramRule::Engram => None,
    });
    Shape::from_layers(spec.layers.len(), attention, ple_layer, image)
}
