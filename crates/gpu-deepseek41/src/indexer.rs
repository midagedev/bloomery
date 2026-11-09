//! V4.1's binding of the lightning indexer ([`bloomery_gpu::indexer`]): the
//! kernels' shape read from the file's hyperparameters.

use bloomery_gpu::GpuError;
use bloomery_gpu::indexer::IndexerKernels;
use cuda_core::CudaContext;
use model::arch::deepseek41::hparams::Hparams;
use std::sync::Arc;

/// The indexer kernels for the model `hp` describes: its indexer must be
/// [`bloomery_gpu::indexer::HEADS`] heads of
/// [`bloomery_gpu::indexer::HEAD_DIM`], and its rope turns an even number of
/// values, at most a head, in pairs of pairs. Load-time only.
pub fn kernels(ctx: &Arc<CudaContext>, hp: &Hparams) -> Result<IndexerKernels, GpuError> {
    IndexerKernels::with_shape(ctx, hp.indexer.n_head, hp.indexer.head_dim, hp.rope_dims)
}
