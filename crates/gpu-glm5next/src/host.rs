//! The GLM host tier's layer finder: the architecture's part of the common
//! host run ([`bloomery_gpu::host::run::HostRun`]), which owns the stacks'
//! storage, the scratch and the [`bloomery_gpu::hybrid::HostExperts`] calls.

use std::ops::Range;

use model::ModelError;
use model::arch::glm5next::host;
use model::arch::glm5next::hparams::Hparams;
use model::moe::HostLayer;
use model::r8file::R8Source;

/// The host views of the routed layers `run`, each from `host::layer`; a
/// layer of the run that does not route (a dense lead layer) is refused by
/// name. Load-time only.
pub(crate) fn routed_layers(
    src: R8Source<'_>,
    hp: &Hparams,
    run: Range<usize>,
) -> Result<Vec<HostLayer>, ModelError> {
    run.map(|l| {
        host::layer(src, hp, l)?.ok_or_else(|| {
            ModelError::MissingTensor(format!(
                "layer {l}'s routed stacks: a layer of the host run does not route"
            ))
        })
    })
    .collect()
}
