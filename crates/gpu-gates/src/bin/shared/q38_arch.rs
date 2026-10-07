//! The one piece of `q38_fixture` that names the Qwen3.8 family's own hyperparameters: the layer
//! kinds a file's header holds. It sits beside the gates that read it; the library module keeps
//! the family-free formulas and the tier helpers.

use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::q38_fixture::Shape;
use model::arch::qwen35moe::hparams::Hparams;

/// The shape of a file's hyperparameters.
///
/// # Errors
/// A file with no layer or an interval of 0.
pub fn shape_of(hp: &Hparams) -> Result<Shape, GateError> {
    let ple = hp.exp.as_ref().and_then(|e| e.ple.as_ref());
    Shape::from_header(
        hp.n_layer,
        hp.interval,
        ple.map(|p| p.layer),
        ple.and_then(|p| p.image),
    )
}
