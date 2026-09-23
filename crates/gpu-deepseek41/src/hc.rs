//! Hyper-connections: the four residual streams around every sub-layer.
//! HC_PRE mixes them into the sub-layer's input and derives `pre`, `post` and
//! the Sinkhorn-normalized `comb` from a 24-value head gemv; HC_POST adds the
//! output back into the streams; the stream fold collapses them before the head.
